# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: ak2191
#     language: python
#     name: python3
# ---

# %% [markdown]
# <img src="../../../docs/assets/0.-BC-dev-hub-LOGO-flicker.svg" alt="BrainChip Dev Hub" width="200"/>
#
# # Visual Wake Words on Akida 1
#
# <p align="right">
# Run Time: ~1 hour with training included / ~20 seconds with training skipped
# </p>
#
# This notebook steps through the full pipeline for a **Visual Wake Words (VWW)**
# binary classifier on **Akida 1 (AKD1500)**: training a tf_keras model,
# quantization and conversion to Akida format, and evaluation of the resulting
# Akida model. For background on the VWW task, dataset, and model performance
# benchmarks, see the [README](README.md).
#
# The focus is on the **Akida-specific** aspects of the pipeline. The full data
# preprocessing and training code is available in the accompanying Python files —
# it is standard tf_keras code and is not described further in this notebook.
#
# By default, model training is run to ensure reproducibility. However, you can
# cut the running time of the notebook to under a minute if desired by 
# skipping the training runs and downloading pretrained float and quantized models
# instead: simply set the relevant `RUN_FLOAT_TRAINING` and `RUN_QAT_TRAINING`
# variables in the first code cell to `False`.

# %% [markdown]
# ## Setup
#
# The default dataset path is `./data/vw_coco2014_96`. See the [README](README.md)
# for download instructions and symlink setup. Update `DATA_PATH` below if needed.

# %%
import os
import pooch

import numpy as np
from tf_keras.utils import set_random_seed

# %%
DATA_PATH = './data/vw_coco2014_96'
MODELS_DIR = './models'
os.makedirs(MODELS_DIR, exist_ok=True)

RUN_FLOAT_TRAINING = True
RUN_QAT_TRAINING = True

SEED = 42

set_random_seed(SEED)

# %% [markdown]
# ## Dataset
#
# `get_data` returns a training and a validation `tf.data.Dataset`. The full
# preprocessing and augmentation code is in [vww_data.py](vww_data.py) —
# standard tf_keras pipeline code, not described further here.

# %%
from vww_data import get_data

BATCH_SIZE = 32
INPUT_SHAPE = (96, 96, 3)

train_ds, val_ds = get_data(DATA_PATH, INPUT_SHAPE, BATCH_SIZE)

# %% [markdown]
# ## Model
#
# The backbone is **AkidaNet** — a MobileNet V1 variant whose layer structure
# maps efficiently onto Akida hardware — with width multiplier `alpha=0.25`.
# The narrow variant keeps parameter count low while retaining sufficient
# capacity for the binary VWW task.
#
# Key design choices:
#
# - **Input resolution 96×96**, down from the 224×224 used for the ImageNet
#   base model.
# - **2-class output head** (person / non-person), returning raw logits — no
#   softmax. The loss function handles the softmax implicitly via
#   `from_logits=True`.
# - **`input_scaling=(255, 0)`** embeds a linear mapping from uint8 [0, 255]
#   to float [0, 1] as the first layer. This keeps the data pipeline
#   hardware-friendly — inputs never need to be normalised outside the model.
# - **`AkidaVersion.v1` context** constrains the layer configuration to what
#   the AKD1500 hardware supports.

# %%
from akida_models import akidanet_imagenet
from cnn2snn import set_akida_version, AkidaVersion

with set_akida_version(AkidaVersion.v1):
    model = akidanet_imagenet(
        input_shape=INPUT_SHAPE,
        classes=2,
        alpha=0.25,
        include_top=True,
        input_scaling=(255, 0),
    )

model.summary()

# %% [markdown]
# ## Float Training
#
# The model is trained for 50 epochs using Adam with a step-decay learning rate
# schedule. The full training code is in [vww_train.py](vww_train.py) —
# standard tf_keras code, not described further here.
#

# %%
from vww_train import train_vww


if RUN_FLOAT_TRAINING:
    LEARNING_RATE = 1e-3
    EPOCHS = 50

    train_vww(model, train_ds, val_ds,
            EPOCHS,
            LEARNING_RATE,
            )

    float_model_path = os.path.join(MODELS_DIR, 'akidanet_vww.h5')
    model.save(float_model_path, include_optimizer=False)
    print(f'Float model saved to {float_model_path}')
else:
    from tf_keras.models import load_model
    print('Training skipped. Retreiving model from BrainChip server...')
    model_url = 'https://data.brainchip.com/models/AkidaV1/akidanet/akidanet_vww.h5'
    model_path = pooch.retrieve(
            url=model_url,
            known_hash='00e03f13226cd622ad92bdb3402c4b4399a69875f2dde6ccadfb235ad6994d78',
            path="./models",
            fname='pretrained_' + os.path.basename(model_url),
        )
    model = load_model(model_path)

# %% [markdown]
# ### Evaluate float model

# %%
model.compile(metrics=['accuracy'])
_, float_accuracy = model.evaluate(val_ds, verbose=0)
print(f'Float validation accuracy: {float_accuracy:.4f}')

# %% [markdown]
# ## Quantization
#
# Akida 1 operates with integer weights and activations. We use `cnn2snn` to
# quantize the float model to 4 bits for both weights and activations (8-bit
# weights are enabled for the first layer only, which is also unusual in
# receiving uint8 inputs):
#
# Post-training quantization maps the float parameters to their nearest
# representable integer values. Some accuracy is typically lost in this step,
# which the subsequent QAT pass recovers.
#
# Note: the quantized model can be saved using the standard method. However,
# for later reloading, because of the custom quantized layers in the model
# we have to use the `load_quantized_model` function from cnn2snn (a wrapper
# around the standard tf_keras loading function)

# %%
from cnn2snn import quantize, load_quantized_model

quantized_model = quantize(
    model,
    input_weight_quantization=8,
    weight_quantization=4,
    activ_quantization=4,
)
ptq_model_path = 'akidanet_vww_ptq.h5'
quantized_model.save(ptq_model_path, include_optimizer=False)

del quantized_model

quantized_model = load_quantized_model(ptq_model_path)

# %% [markdown]
# ### Quantization-Aware Training (QAT)
#
# Two epochs of fine-tuning at a constant learning rate of 1×10⁻⁴ are
# sufficient to recover most of the accuracy lost during quantization. It is
# typical to find that a lower learning rate (e.g. /10) is required during this 
# phase than during the initial training.
#
# Note that, although Quatization Aware Training can sound intimidating,
# the model quantized via `cnn2snn` can simply be reinserted into the 
# same training function that was used for the initial float training.

# %%

if RUN_QAT_TRAINING:
    QAT_LEARNING_RATE = 1e-4
    QAT_EPOCHS = 2

    train_vww(quantized_model, train_ds, val_ds,
            QAT_EPOCHS,
            QAT_LEARNING_RATE,
            )

    qat_model_path = os.path.join(MODELS_DIR, 'akidanet_vww_qat.h5')
    quantized_model.save(qat_model_path, include_optimizer=False)
    print(f'QAT model saved to {qat_model_path}')
else:
    print('QAT skipped. Retreiving model from BrainChip server...')
    q_model_url = 'https://data.brainchip.com/models/AkidaV1/akidanet/akidanet_vww_iq8_wq4_aq4.h5'
    q_model_path = pooch.retrieve(
            url=q_model_url,
            known_hash='cd130d90ed736447b6244dc1228e708b9dab20af0d2bf57b9a49df4362467ea8',
            path="./models",
            fname='pretrained_'+ os.path.basename(q_model_url),
        )
    quantized_model = load_quantized_model(q_model_path)

# %% [markdown]
# ### Evaluate quantized model

# %%
quantized_model.compile(metrics=['accuracy'])
_, qat_accuracy = quantized_model.evaluate(val_ds, verbose=0)
print(f'QAT validation accuracy: {qat_accuracy:.4f}')

# %% [markdown]
# ## Conversion to Akida Format
#
# `cnn2snn.convert` compiles the quantized Keras model into an Akida `.fbz`
# model that can be loaded and executed directly on AKD1500 hardware.
# The converter verifies hardware compatibility and maps each layer to its
# corresponding Akida primitive.

# %%
from cnn2snn import convert

akida_model = convert(quantized_model)

akida_model_path = os.path.join(MODELS_DIR, 'akidanet_vww_qat.fbz')
akida_model.save(akida_model_path)
print(f'Akida model saved to {akida_model_path}')
akida_model.summary()

# %% [markdown]
# ## Evaluation on Akida Hardware
#
# The Akida runtime cannot consume `tf.data.Dataset` objects directly, rather
# it expects a 4D numpy array (n, h, w, c) in uint8 format. So we
# iterate over validation batches manually.
#
# The model output tensor has shape `(B, 1, 1, C)` which is squeezed to 
# `(B, C)` before taking the class argmax.

# %%
import akida

akida_hw_model = akida.Model(akida_model_path)

labels_all = []
pots_all = []

for batch, label_batch in val_ds:
    if not isinstance(batch, np.ndarray):
        batch = batch.numpy()
    pots_batch = akida_hw_model.predict(batch)
    pots_batch = pots_batch.squeeze(axis=(1, 2))
    labels_all.append(label_batch)
    pots_all.append(pots_batch)

labels_all = np.concatenate(labels_all)
pots_all = np.concatenate(pots_all)
preds = np.argmax(pots_all, axis=1)

akida_accuracy = float(np.mean(preds == np.array(labels_all)))
print(f'Akida hardware accuracy: {akida_accuracy:.4f}')

# %% [markdown]
# ### Activation Sparsity
#
# Akida hardware skips computation for zero-valued activations, so activation
# sparsity directly reduces both energy consumption and inference latency.
# Below we measure per-layer sparsity on a 1024-sample calibration batch drawn
# from the training set.

# %%
from tf_keras.preprocessing.image import ImageDataGenerator
from akida_models.sparsity import compute_sparsity

def get_samples(data_path, input_shape, num_samples=1024):
    generator = ImageDataGenerator().flow_from_directory(
        os.path.join(data_path, 'train'),
        target_size=input_shape[:2],
        batch_size=num_samples,
        shuffle=False,
    )
    images, _ = next(generator)
    return images[:num_samples].astype(np.uint8)

samples = get_samples(DATA_PATH, INPUT_SHAPE)
sparsity_dict = compute_sparsity(akida_hw_model, samples=samples)

col_w = max(len(k) for k in sparsity_dict) + 2
print(f"\n{'Layer':<{col_w}} {'Sparsity':>10}")
print("-" * (col_w + 11))
for layer, sparsity in sparsity_dict.items():
    print(f"{layer:<{col_w}} {sparsity:>9.2%}")
print("-" * (col_w + 11))
print(f"{'Mean':<{col_w}} {np.mean(list(sparsity_dict.values())):>9.2%}")

# %% [markdown]
# ## Summary
#

# %%
print(f"Float: {float_accuracy:.4f}")
print(f"QAT:   {qat_accuracy:.4f}")
print(f"Akida: {akida_accuracy:.4f}")
