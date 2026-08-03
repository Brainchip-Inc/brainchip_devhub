# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: ak2191
#     language: python
#     name: python3
# ---

# %% [markdown] id="83fe7c98"
# # PlantVillage Disease Classification
#
# <p align="right">
# Run Time: ~20 minutes with training included / ~15 minutes with training skipped
# </p>
#
# This notebook walks through the complete pipeline to train, quantize, convert, and
# benchmark an AkidaNet model on the **PlantVillage** dataset for Akida 1 hardware.
#
# PlantVillage contains 54,306 images of healthy and diseased plant leaves across
# 38 categories (14 crop species × multiple disease types plus healthy variants).
# The task is a 38-class image classification problem: given a 224×224 RGB image of
# a leaf, identify the crop species and disease (or healthy state). The pipeline loads
# 54,305 of these images (a few files are not recognised as valid images and are skipped).
#
# The pipeline follows the standard Akida workflow:
# 1. Train a float model
# 2. Post-training quantization (PTQ)
# 3. Quantization-aware training (QAT) fine-tuning
# 4. Conversion to Akida `.fbz` format
# 5. Hardware evaluation and benchmarking

# %% colab={"base_uri": "https://localhost:8080/"} id="cBpfzYVbH0pa" outputId="7ec9c880-1035-4004-a30a-174a0bd34cfc"
# Colab-only setup. Local users: ignore this cell — it does nothing for you.
import sys, os

if 'google.colab' in sys.modules:
    if not os.path.exists('colab_setup.py'):
        # !wget -q https://raw.githubusercontent.com/Brainchip-Inc/brainchip_devhub/main/akida1/model_zoo/plant_village/colab_setup.py
    import colab_setup; colab_setup.setup()


# %% id="2990a3e3"
import os
import numpy as np
import tensorflow as tf

import pooch
from tf_keras.utils import set_random_seed

from cnn2snn import load_quantized_model

DATA_PATH = './data/plant_village'
MODELS_DIR = './models/'
os.makedirs(MODELS_DIR, exist_ok=True)

RUN_FLOAT_TRAINING = True
RUN_QAT_TRAINING = True

SEED = 42

# Must be called before any TF ops to make GPU ops (conv backward passes,
# bilinear resize, etc.) deterministic. Has a small throughput cost.
tf.config.experimental.enable_op_determinism()

# %% [markdown] id="0fc44136"
# ## Dataset
#
# The **PlantVillage** dataset is loaded from the authors' official GitHub repository
# ([spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset)).
# On the first run, the images are downloaded and extracted to `DATA_PATH`
# automatically; subsequent runs read from the local copy.
#
# The dataset is split 80/10/10 (train/val/test). Images are resized from variable
# original sizes to **224 × 224 RGB** and delivered as uint8 pixel values (0–255).
# Training applies random horizontal flip, brightness jitter, and contrast jitter
# for regularisation.
#
# The download and extraction are handled inside `get_data()` on first call, so
# there is no separate pre-download step required.
#

# %% colab={"base_uri": "https://localhost:8080/"} id="4f941071" outputId="ca62b756-74ea-4c23-ff29-375427dda394"
from plant_village_data import get_data

INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32

train_ds, val_ds, test_ds = get_data(DATA_PATH, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, seed=SEED)
print('Datasets ready.')

# %% [markdown] id="72e45ea6"
# ## Model
#
# The model is based on **AkidaNet** (`akida_models.akidanet_imagenet`) with:
# - Width multiplier **alpha = 0.5** — provides sufficient capacity for 38 classes
#   while remaining efficient on Akida 1 hardware
# - Input resolution **224 × 224 RGB**
# - **38-class** classification head (replacing the ImageNet top)
# - **Input scaling (255, 0)** built into the model — the pipeline delivers raw
#   uint8 pixel values and the model normalises them internally
#
# AkidaNet is specifically designed for Akida hardware: it uses only operations
# that map efficiently to Akida Neural Processors (NPs), including depthwise
# separable convolutions and ReLU activations.

# %% colab={"base_uri": "https://localhost:8080/"} id="eee611de" outputId="58b54bfa-dd94-4fcb-d9b0-a12f1c001359"
from plant_village_model import build_plant_village_model

model = build_plant_village_model(seed=SEED)
model.summary()

# %% [markdown] id="4dcc13f8"
# ## Float Training
#
# The model is trained in full float32 precision for 10 epochs using the Adam
# optimiser and sparse categorical cross-entropy loss (with `from_logits=True`,
# since the model head outputs raw logits rather than softmax probabilities).
#
# The learning rate follows an **exponential decay** schedule, starting at `1e-3`
# and decaying to approximately `1e-5` by the final epoch.
#
# Set `RUN_FLOAT_TRAINING = True` above to train from scratch. Otherwise, the
# cell below loads a pre-trained float model from the `pretrained_models/` folder.

# %% colab={"base_uri": "https://localhost:8080/", "height": 384} id="e10c708f" outputId="5c3a884e-e0e9-4b05-e2fc-5fcbd7aa2961"
from plant_village_train import train_plant_village

if RUN_FLOAT_TRAINING:
    train_plant_village(
        model, train_ds, val_ds,
        epochs=10,
        learning_rate=1e-3,
        seed=SEED)
    model.save(
        MODELS_DIR + 'akidanet_plant_village.h5',
        include_optimizer=False)
    print('Float model saved.')
else:
    float_model_path = 'pretrained_models/akidanet_plant_village.h5'
    model = load_quantized_model(float_model_path)
    model.compile(metrics=['accuracy'])

# %% colab={"base_uri": "https://localhost:8080/"} id="cb1849a2" outputId="a921a0d6-7cd1-4452-b7c1-9d9fa3b441d5"
_, float_acc = model.evaluate(test_ds, verbose=1)
print(f'Float accuracy: {float_acc:.4f}')

# %% [markdown] id="75b10e92"
# ## Quantization
#
# Post-training quantization (PTQ) via `cnn2snn.quantize` converts the model
# to fixed-point arithmetic:
# - **Input**: 8-bit (`-i 8`)
# - **Weights**: 4-bit (`-w 4`)
# - **Activations**: 4-bit (`-a 4`)
#
# 4-bit quantization must be used to be compatible with Akida 1 hardware. Note though that
# the first layer (both its inputs and weights) can be 8-bit.

# %% colab={"base_uri": "https://localhost:8080/"} id="e1c1042a" outputId="ba73619b-155d-46ae-aef3-283637c2a066"
import cnn2snn

quantized_model = cnn2snn.quantize(
    model,
    input_weight_quantization=8,
    weight_quantization=4,
    activ_quantization=4)
print('Model quantized to i8/w4/a4.')

# %% [markdown] id="7c2bda60"
# Quantizing a model after training like this is referred to as Post-Training
# Quantization (PTQ). It can slightly reduce accuracy (especially at 4-bits as
# here) because the model was trained with continuous weights but is now
# evaluated with discrete values.

# %% colab={"base_uri": "https://localhost:8080/"} id="bf5fe299" outputId="71e10ad4-7cbd-4b52-a65b-f465d553ef69"
quantized_model.compile(metrics=['accuracy'])
_, ptq_acc = quantized_model.evaluate(test_ds, verbose=1)
print(f'PTQ accuracy: {ptq_acc:.4f}')

# %% [markdown] id="93943fd3"
# ## Quantization-Aware Training (QAT)
#
# We can run Quantization Aware Training (QAT) to recover most of the drop in
# accuracy. QAT fine-tunes the quantized model for a few epochs (here, 2) at a
# reduced learning rate (`1e-4`). Note that, although it can sound intimidating,
# QAT with BrainChip's quantization tools is no more complex than simply sending
# the quantized model back through the same training pipeline that was used to
# prepare the float model in the first place.
#
# Set `RUN_QAT_TRAINING = True` above to run QAT locally. Otherwise, the cell
# below loads a pre-trained QAT model from the `pretrained_models/` folder.

# %% colab={"base_uri": "https://localhost:8080/"} id="f79ad823" outputId="68c83923-9cbc-4ede-9765-7f17112d2cc2"
if RUN_QAT_TRAINING:
    # We refetch the dataset, only to ensure reproducibility against the non-notebook pipeline.
    # This resets the shuffle seed on the training data
    train_ds, val_ds, test_ds = get_data(DATA_PATH, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, seed=SEED)
    train_plant_village(
        quantized_model, train_ds, val_ds,
        epochs=2,
        learning_rate=1e-4)
    quantized_model.save(
        MODELS_DIR + 'akidanet_plant_village_qat.h5',
        include_optimizer=False)
    print('QAT model saved.')
else:
    qat_model_path = 'pretrained_models/akidanet_plant_village_qat.h5'
    quantized_model = load_quantized_model(qat_model_path)
    quantized_model.compile(metrics=['accuracy'])

# %% colab={"base_uri": "https://localhost:8080/"} id="b36bc778" outputId="5845d9ab-cecc-448d-e541-0defdf5ca02e"
_, qat_acc = quantized_model.evaluate(test_ds, verbose=1)
print(f'QAT accuracy: {qat_acc:.4f}')

# %% [markdown] id="74f0f887"
# ## Conversion to Akida Format
#
# `cnn2snn.convert` compiles the quantized Keras model into an Akida `.fbz`
# model that can be loaded and executed directly on AKD1500 hardware.
# The converter verifies hardware compatibility and maps each layer to its
# corresponding Akida primitive.

# %% colab={"base_uri": "https://localhost:8080/"} id="ec70dc83" outputId="7c792ae5-3443-433e-f0a0-342862adc6e7"
akida_model = cnn2snn.convert(quantized_model)

akida_model_path = os.path.join(MODELS_DIR, 'akidanet_plant_village_qat.fbz')
akida_model.save(akida_model_path)
print(f'Akida model saved to {akida_model_path}')
akida_model.summary()


# %% [markdown] id="39ac59e5"
# ## Evaluation of Akida Model
#
# We now run evaluation through the Akida model, to check that accuracy is
# comparable to that obtained from the quantized tf_keras model. Here, we deliberately
# use the software backend (the default, since we do not check for and map to
# a connected hardware device): this delivers a
# bit-accurate simulation of the results that will be obtained when running
# the model on hardware.
#
# In the accompanying [plant_village_notebook_benchmark.ipynb](plant_village_notebook_benchmark.ipynb)
# the same evaluation is run using the hardware backend (if, of course, a hardware Akida
# device is connected), allowing you to confirm that the results are identical.

# %% [markdown] id="434fb68e"
# ### Run Evaluation on Akida
#
# The Akida runtime cannot consume `tf.data.Dataset` objects directly, rather
# it expects a 4D numpy array (n, h, w, c) in uint8 format. So we
# iterate over test batches manually.
#
# The model output tensor has shape `(B, 1, 1, C)` which is squeezed to
# `(B, C)` before taking the class argmax.

# %% colab={"base_uri": "https://localhost:8080/"} id="30845549" outputId="b285a918-1993-492f-8648-500d4332f3a2"
from tqdm import tqdm

labels_all = []
logits_all = []
for batch, label_batch in tqdm(test_ds, desc="Evaluating on Akida"):
    if not isinstance(batch, np.ndarray):
        batch = batch.numpy()

    logits_batch = akida_model.predict(batch, batch_size=BATCH_SIZE)

    logits_batch = logits_batch.squeeze(axis=(1, 2))
    labels_all.append(label_batch)
    logits_all.append(logits_batch)

labels_all = np.concatenate(labels_all)
logits_all = np.concatenate(logits_all)
preds = np.argmax(logits_all, axis=1)

akida_acc = float(np.mean(preds == np.array(labels_all)))
print(f'Akida accuracy: {akida_acc:.4f}')

# %% [markdown] id="7139d125"
# ### Activation Sparsity
#
# Akida hardware skips computation for zero-valued activations, so activation
# sparsity directly reduces both energy consumption and inference latency.
# Below we measure per-layer sparsity on a 100-sample calibration batch drawn
# from the training set.

# %% colab={"base_uri": "https://localhost:8080/"} id="34950d3d" outputId="ad4236e7-5e19-4224-f16a-2feaab0f6373"
from akida_models.sparsity import compute_sparsity
from brainchip_utils.plot_utils import pretty_print_sparsity
from plant_village_data import get_samples

NUM_SAMPLES = 100

samples = get_samples(DATA_PATH, input_shape=INPUT_SHAPE, num_samples=NUM_SAMPLES)
sparsity_dict = compute_sparsity(akida_model, samples=samples)
pretty_print_sparsity(sparsity_dict)

# %% [markdown] id="6255ca47"
# ## Summary
#
# The table below compares test accuracy across the three model variants.
# The goal is that QAT and Akida accuracy remain close to the float baseline.

# %% colab={"base_uri": "https://localhost:8080/"} id="88ab56dc" outputId="db49cb35-3174-4221-b4f3-2dc7a2c1fdd4"
print('PlantVillage results')
print('=' * 40)
print(f'  Float accuracy:     {float_acc * 100:.2f}%')
print(f'  QAT accuracy:       {qat_acc * 100:.2f}%')
print(f'  Akida accuracy:     {akida_acc * 100:.2f}%')
