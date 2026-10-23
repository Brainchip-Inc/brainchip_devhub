# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# <img src="https://raw.githubusercontent.com/Brainchip-Inc/brainchip_devhub/main/docs/assets/0.-BC-dev-hub-LOGO-flicker.svg" alt="BrainChip Dev Hub" width="200"/>
#
# # Portrait Segmentation (Portrait128) — Akida 2 Training
#
# Run Time: ~7 hours with training included (NVIDIA RTX 2080 Ti) / TBD with training skipped
#
# This notebook walks through the complete pipeline to train, quantize, convert, and evaluate an Akida U-Net model on the **Portrait128** dataset for Akida 2 hardware.
#
# Portrait128 is a binary segmentation task: for each 128 × 128 portrait, the model labels every pixel as **person** or **background**.
#
# The pipeline follows the standard Akida workflow:
# 1. Train a float model
# 2. 8-bit quantization (PTQ), then one epoch of QAT
# 3. Conversion to Akida `.fbz` format
# 4. Evaluation on Akida
#
# The focus is on the **Akida-specific** aspects of the pipeline. The data loading,
# model definition and training loop live in the accompanying Python files: they are
# standard tf_keras code and are imported here rather than repeated.
#
# Segmentation differs from the classification examples in this model zoo in two ways
# that matter on Akida. The model has to **upsample** back to full resolution, which
# needs transposed convolutions, an Akida 2 feature. And its output is a full-size map
# rather than a handful of class scores, so the last layers of the network run at the
# highest resolution. Both come up below.

# %%
# Colab-only setup. Local users: ignore this cell — it does nothing for you.
import sys, os

if 'google.colab' in sys.modules:
    if not os.path.exists('colab_setup.py'):
        !wget -q https://raw.githubusercontent.com/Brainchip-Inc/brainchip_devhub/main/akida2/model_zoo/portrait128/colab_setup.py
    import colab_setup; colab_setup.setup()

# %% [markdown]
# ## Setup
#
# The default dataset path is `./data/portrait128`, a folder holding the two arrays
# `img_uint8.npy` and `msk_uint8.npy`. See the [README](README.md) for where to get
# them and for symlink setup. Update `DATA_PATH` below if you keep them elsewhere.
#
# Setting `RUN_FLOAT_TRAINING` to `False` skips the training run and loads the float
# model saved in `MODELS_DIR` by an earlier run of this notebook or of
# `portrait128_train.sh`.

# %%
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

DATA_PATH = './data/portrait128'
MODELS_DIR = './models'
os.makedirs(MODELS_DIR, exist_ok=True)

RUN_FLOAT_TRAINING = True

SEED = 42

# Must be called before any TF ops to make GPU ops deterministic.
tf.config.experimental.enable_op_determinism()

# %% [markdown]
# ## Dataset
#
# `get_data` returns a training and a validation `tf.data.Dataset`. The full loading
# and augmentation code is in [portrait128_data.py](portrait128_data.py).
#
# A few points that matter for what follows:
#
# * **Images stay uint8**, in their original `[0, 255]` range. The model rescales them
#   itself (see Model below), so the same data feed the float, quantized and Akida
#   models without any change, and a deployed Akida model can take camera pixels
#   directly.
# * **Labels are masks**, of shape `(128, 128, 1)`: 1 for person, 0 for background.
# * **Augmentation is geometric** (flips, shifts and zooms), and each image is stacked
#   with its mask before it is transformed. That way the mask moves with the image;
#   transforming them separately would misalign the labels.
# * **The split is fixed.** The dataset ships as one pool, so it is split 80/20 with a
#   fixed shuffle seed, independent of `SEED`. Every run is evaluated on the same
#   validation images.

# %%
from portrait128_data import get_data

BATCH_SIZE = 32
INPUT_SHAPE = (128, 128, 3)

train_ds, val_ds = get_data(DATA_PATH, INPUT_SHAPE, BATCH_SIZE, seed=SEED)

# %% [markdown]
# ## Model
#
# The model is the **Akida U-Net** from `akida_models`, an encoder–decoder:
#
# * The **encoder** is AkidaNet with width multiplier alpha = 0.5. Its five stride-2
#   stages reduce the 128 × 128 input to a 4 × 4 × 512 feature map: coarse in space
#   but rich in features.
# * The **decoder** is five separable *transposed*-convolution blocks. Each doubles the
#   spatial size, back up to 128 × 128. A transposed convolution with stride 2 is a
#   learned upsampling, and it is the reason this model needs Akida 2: Akida 1 has no
#   upsampling layer. Dropout after each block regularizes the decoder during
#   training; it does nothing at inference and is dropped at conversion.
# * A final **1 × 1 convolution** gives one logit per pixel: positive for person,
#   negative for background. There is no sigmoid in the model. The loss applies it,
#   and a threshold at 0 gives the same mask as a threshold at probability 0.5.
#
# Unlike the classic U-Net there are **no skip connections** from encoder to decoder,
# so all the detail in the output has to pass through the 4 × 4 bottleneck. That keeps
# the graph a simple chain. The cost is softer mask edges than a U-Net with skips
# would give.
#
# The input rescaling is part of the model. The encoder's `input_scaling=(128, -1)`
# adds a layer that maps `[0, 255]` to `[-1, 1)`. Keeping it in the graph means one data
# pipeline serves every version of the model. It also means the conversion step can
# fold the rescaling into the Akida input layer's parameters, so the deployed model
# still takes plain uint8 pixels.
#
# **Encoder initialization.** `base_weights='imagenet'` loads BrainChip's
# ImageNet-trained AkidaNet 0.5 weights into the encoder, by layer name; the decoder
# starts from random weights. The ImageNet model was trained at 224 × 224, but
# convolution weights don't depend on the input size, so they load directly into this
# 128 × 128 encoder. Starting from features that already recognize edges, textures and
# shapes is why the learning rate below can be small.

# %%
from portrait128_model import build_portrait128_model

model = build_portrait128_model(base_weights='imagenet', seed=SEED)
model.summary()

# %% [markdown]
# ## Float Training
#
# The full training code is in [portrait128_train.py](portrait128_train.py). The
# choices that matter here:
#
# * **Loss**: binary cross-entropy on the logits, averaged over every pixel. Each pixel
#   is its own binary classification.
# * **Metrics**: *binary IoU*, the overlap between predicted and true masks averaged
#   over person and background, and *pixel accuracy*. Background covers a large share of
#   every image, so pixel accuracy is high even for poor masks; IoU penalizes a mask of
#   the wrong shape, so it is the one to watch.
# * **Learning rate**: Adam with a one-cycle cosine schedule, stepped every batch. It
#   warms up from a 25th of the peak to the peak `LEARNING_RATE` over the first 30 % of
#   training, then decays. The peak, 3e-5, is low because the encoder starts
#   pretrained: a high rate early on would destroy the ImageNet features before the
#   decoder has learned to use them.

# %%
from portrait128_train import train_portrait128, get_metrics

if RUN_FLOAT_TRAINING:
    LEARNING_RATE = 3e-5
    EPOCHS = 50
    # Freshly build the datasets for reproducibility
    train_ds, val_ds = get_data(DATA_PATH, INPUT_SHAPE, BATCH_SIZE, seed=SEED)

    train_portrait128(model, train_ds, val_ds, EPOCHS, LEARNING_RATE, seed=SEED)

    float_model_path = os.path.join(MODELS_DIR, 'akida_unet_portrait128.h5')
    model.save(float_model_path, include_optimizer=False)
    print(f'Float model saved to {float_model_path}')
else:
    from tf_keras.models import load_model
    print('Training skipped. Loading an existing float model...')
    model = load_model(os.path.join(MODELS_DIR, 'akida_unet_portrait128.h5'))

# %% [markdown]
# ### Evaluate float model

# %%
model.compile(metrics=get_metrics())
float_results = model.evaluate(val_ds, verbose=0, return_dict=True)
float_iou, float_accuracy = float_results['binary_iou'], float_results['accuracy']
print(f'Float binary IoU: {float_iou:.4f}, pixel accuracy: {float_accuracy:.4f}')

# %% [markdown]
# ## Quantization (quantizeml)
#
# Akida runs on integers, so the weights and activations have to be quantized. Akida 2
# quantizes with **quantizeml** (not `cnn2snn.quantize`). We quantize to **8 bits**
# (i8/w8/a8) by post-training quantization (PTQ) only: 256 levels per weight and
# activation track the float values closely enough that no retraining is needed.
# `input_weight_bits` sets the input layer separately, because it runs on Akida's
# dedicated input processor; here it is 8-bit like everything else.
#
# **Why no 4-bit variant?** Other Akida 2 examples also offer a 4-bit model, fine-tuned
# with quantization-aware training (QAT). This one can't. The decoder upsamples with
# transposed depthwise convolutions, and on Akida 2 those layers need 8-bit inputs and
# 8-bit weights. In a model quantized to 4 bits throughout, all five of them would fall
# back to software, so the model would no longer run fully on the hardware.
#
# Quantization with `quantizeml` requires **calibration samples**. Calibration runs real
# inputs through the model to measure the range of every layer's activations, and those
# ranges set the quantization scales. Random data would give the wrong ranges, so the
# samples should be representative images for the task. They are drawn from the
# **training** split, to avoid leaking validation data into the model.

# %%
from quantizeml.models import quantize, QuantizationParams

from portrait128_data import get_samples

NUM_SAMPLES = 1024
samples = get_samples(DATA_PATH, INPUT_SHAPE, num_samples=NUM_SAMPLES)

# --- 8-bit (i8 / w8 / a8), PTQ ---
qparams_8bit = QuantizationParams(input_weight_bits=8, weight_bits=8, activation_bits=8)
model_8bit = quantize(model, qparams=qparams_8bit, samples=samples)

model_8bit.compile(metrics=get_metrics())
ptq_results = model_8bit.evaluate(val_ds, verbose=0, return_dict=True)
iou_ptq, acc_ptq = ptq_results['binary_iou'], ptq_results['accuracy']
print(f'8-bit PTQ binary IoU: {iou_ptq:.4f}, pixel accuracy: {acc_ptq:.4f}')

# %% [markdown]
# ### One epoch of QAT
#
# The 8-bit model then gets one epoch of quantization-aware training (QAT), as the
# model in BrainChip's model zoo does. During QAT the forward pass uses the quantized
# weights and activations, while the gradients update an underlying float copy of the
# weights, so the network adjusts to the rounding that PTQ introduced.
#
# QAT starts from a model that is already close to a good solution, so it uses a lower
# learning rate: a tenth of the float rate here. A higher rate would undo the float
# training rather than refine it.
#
# A model quantized with `quantizeml` is a standard Keras model, so it goes straight
# back into the same training function used for float training. The PTQ model itself
# is not kept; compare its result above with the one after QAT.

# %%
QAT_EPOCHS = 1
QAT_LR = 3e-6
train_ds, val_ds = get_data(DATA_PATH, INPUT_SHAPE, BATCH_SIZE, seed=SEED)
train_portrait128(model_8bit, train_ds, val_ds, QAT_EPOCHS, QAT_LR, seed=SEED)

q8_path = os.path.join(MODELS_DIR, 'akida_unet_portrait128_i8_w8_a8_qat.h5')
model_8bit.save(q8_path, include_optimizer=False)

model_8bit.compile(metrics=get_metrics())
q8_results = model_8bit.evaluate(val_ds, verbose=0, return_dict=True)
iou_8bit, acc_8bit = q8_results['binary_iou'], q8_results['accuracy']
print(f'8-bit QAT binary IoU: {iou_8bit:.4f}, pixel accuracy: {acc_8bit:.4f}')

# %% [markdown]
# Note: the quantized model can be saved using the standard method. However,
# for later reloading, because of the custom quantized layers in the model
# we have to use the `load_model` function from `quantizeml.model_io` (a wrapper
# around the standard tf_keras loading function).

# %%
# Just to illustrate how to load a quantized model
del model_8bit

from quantizeml.model_io import load_model
model_8bit = load_model(q8_path)

# %% [markdown]
# ## Conversion to Akida Format
#
# `cnn2snn.convert` compiles the quantized Keras model into an Akida `.fbz`
# model that can be loaded and executed directly on an Akida 2 hardware device.
# The converter verifies hardware compatibility and maps each layer to its
# corresponding Akida primitive.
#
# In the summary below, note the following:
#
# * The input rescaling has no layer of its own. It was folded into the first
#   convolution.
# * Each decoder block becomes a `DepthwiseConv2DTranspose` followed by a pointwise
#   `Conv2D`. On Akida 2 the depthwise and pointwise parts of a separable convolution
#   are two separate layers. The `DepthwiseConv2DTranspose` layers are the ones that
#   need 8 bits to run on the hardware.
# * The last layer is a `Dequantizer`, so the model outputs float logits of shape
#   `(128, 128, 1)`, which can be thresholded at 0 just like the Keras model's.

# %%
from cnn2snn import convert

akida_8bit = convert(model_8bit)
akida_8bit.save(os.path.join(MODELS_DIR, 'akida_unet_portrait128_i8_w8_a8_qat.fbz'))

akida_8bit.summary()


# %% [markdown]
# ## Evaluation on Akida (software backend)
#
# We now run evaluation through the Akida model, to check that its results match
# those of the quantized tf_keras model. Here, we deliberately use the software
# backend (the default, since we do not check for and map to a connected hardware
# device): this delivers a bit-accurate simulation of the results that will be
# obtained when running the model on hardware.

# %% [markdown]
# ### Run Evaluation on Akida
#
# The Akida runtime cannot consume `tf.data.Dataset` objects directly; it expects a
# 4D numpy array `(n, h, w, c)` of uint8. So we iterate over validation batches
# manually and convert each one.
#
# The output already has the mask's shape, `(B, 128, 128, 1)`, so it goes straight
# into the same IoU and accuracy metrics as the labels, with no reshaping. (Classifiers
# return `(B, 1, 1, C)` and need a squeeze first.)

# %%
def evaluate_akida(akida_model, ds):
    iou_metric, acc_metric = get_metrics()
    for batch, label_batch in tqdm(ds, desc='Evaluating on Akida'):
        logits = akida_model.predict(batch.numpy().astype(np.uint8))
        iou_metric.update_state(label_batch, logits)
        acc_metric.update_state(label_batch, logits)
    return float(iou_metric.result()), float(acc_metric.result())

akida_iou_8bit, akida_acc_8bit = evaluate_akida(akida_8bit, val_ds)
print(f'Akida 8-bit QAT binary IoU: {akida_iou_8bit:.4f}, pixel accuracy: {akida_acc_8bit:.4f}')

# %% [markdown]
# ## Activation Sparsity
#
# Akida is event-based: its standard neural processors only do work for non-zero
# activations. The more zeros a layer receives (its input *sparsity*), the less time and
# energy it costs. Sparsity depends on the data, so it has to be measured by running real
# images through the model. Here, we reuse the calibration samples built earlier.
#
# The input layer is the exception. It runs on the dedicated input processor, a
# conventional convolution whose cost doesn't depend on how many of its inputs are zero.
#
# For this model, the per-layer benchmark in `portrait128_notebook_benchmark.ipynb` shows
# where the time goes. The slowest layers are the encoder's first full convolutions
# (`conv_1` to `conv_3`), which apply 3 × 3 kernels across every channel of the 64 × 64 and
# 32 × 32 feature maps. The decoder's upsampling layers, which run at up to 128 × 128, come
# next, and the deep encoder layers at 8 × 8 and 4 × 4 cost little. So sparsity in those
# early convolutions has the most room to reduce latency.

# %%
from akida_models.sparsity import compute_sparsity
from brainchip_utils.plot_utils import pretty_print_sparsity

# Reuse the calibration samples built earlier (sparsity needs real activations too).
pretty_print_sparsity(compute_sparsity(akida_8bit, samples=samples))

# %% [markdown]
# ## Summary

# %%
print(f'{"Variant":<16}{"Keras IoU":<12}{"Keras acc":<12}{"Akida IoU":<12}{"Akida acc":<12}')
print(f'{"float":<16}{float_iou:<12.4f}{float_accuracy:<12.4f}{"-":<12}{"-":<12}')
print(f'{"8-bit PTQ":<16}{iou_ptq:<12.4f}{acc_ptq:<12.4f}{"-":<12}{"-":<12}')
print(f'{"8-bit QAT":<16}{iou_8bit:<12.4f}{acc_8bit:<12.4f}{akida_iou_8bit:<12.4f}{akida_acc_8bit:<12.4f}')
