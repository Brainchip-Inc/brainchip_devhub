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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# <img src="https://raw.githubusercontent.com/Brainchip-Inc/brainchip_devhub/main/docs/assets/0.-BC-dev-hub-LOGO-flicker.svg" alt="BrainChip Dev Hub" width="200"/>
#
# # Visual Wake Words (VWW) — Akida 2 Training
#
# This notebook walks through the full Akida 2 pipeline for the Visual Wake Words (person / non-person) task using an AkidaNet-0.25 model at 96×96 resolution: float training, quantization with **quantizeml** (an 8-bit variant and a 4-bit QAT variant), conversion to Akida with **cnn2snn**, and evaluation on the Akida software backend.

# %%
# Colab-only setup. Local users: ignore this cell — it does nothing for you.
import sys, os

if 'google.colab' in sys.modules:
    if not os.path.exists('colab_setup.py'):
        !wget -q https://raw.githubusercontent.com/Brainchip-Inc/brainchip_devhub/main/akida2/model_zoo/vww/colab_setup.py
    import colab_setup; colab_setup.setup()

# %% [markdown]
# ## Setup

# %%
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

DATA_PATH = './data/vw_coco2014_96'
MODELS_DIR = './models'
os.makedirs(MODELS_DIR, exist_ok=True)

RUN_FLOAT_TRAINING = True

SEED = 42

# Must be called before any TF ops to make GPU ops deterministic.
tf.config.experimental.enable_op_determinism()

# %% [markdown]
# ## Dataset
#
# VWW is a directory dataset (`train/` + `val/`). `get_data` returns two batched datasets; images are uint8 in `[0, 255]` (the model rescales on-graph).

# %%
from vww_data import get_data

BATCH_SIZE = 32
INPUT_SHAPE = (96, 96, 3)

train_ds, val_ds = get_data(DATA_PATH, INPUT_SHAPE, BATCH_SIZE, seed=SEED)

# %% [markdown]
# ## Model
#
# The model definition is lifted from the source example (`vww_model.py`). For Akida 2 it is built under `set_akida_version(AkidaVersion.v2)`; the factory is version-aware, so no per-layer changes are needed.

# %%
from vww_model import build_vww_model
model = build_vww_model(seed=SEED)
model.summary()

# %% [markdown]
# ## Float Training
#
# The Akida 2 VWW model is trained from scratch (no ImageNet transfer learning). Epoch count here is a provisional default — adjust for a real run.

# %%
from vww_train import train_vww

if RUN_FLOAT_TRAINING:
    LEARNING_RATE = 1e-3
    EPOCHS = 50
    # Freshly set the dataset seed for reproducibility
    train_ds, val_ds = get_data(DATA_PATH, INPUT_SHAPE, BATCH_SIZE, seed=SEED)

    train_vww(model, train_ds, val_ds, EPOCHS, LEARNING_RATE, seed=SEED)

    float_model_path = os.path.join(MODELS_DIR, 'akidanet_vww.h5')
    model.save(float_model_path, include_optimizer=False)
    print(f'Float model saved to {float_model_path}')
else:
    from tf_keras.models import load_model
    print('Training skipped. Loading an existing float model...')
    model = load_model(os.path.join(MODELS_DIR, 'akidanet_vww.h5'))

# %% [markdown]
# ### Evaluate float model

# %%
model.compile(metrics=['accuracy'])
_, float_accuracy = model.evaluate(val_ds, verbose=0)
print(f'Float validation accuracy: {float_accuracy:.4f}')

# %% [markdown]
# ## Quantization (quantizeml)
#
# Akida 2 quantizes with **quantizeml** (not `cnn2snn.quantize`). We produce two variants:
#
# * **8-bit** (i8/w8/a8) — post-training quantization only; 8-bit PTQ is accurate enough that QAT is not needed.
# * **4-bit** (i8/w4/a4) — quantization-aware training (QAT); 4-bit PTQ accuracy is poor, so we fine-tune. The input layer weights stay 8-bit in both variants.

# %%
from quantizeml.models import quantize
from quantizeml.layers import QuantizationParams

# --- 8-bit variant (i8 / w8 / a8), PTQ only ---
qparams_8bit = QuantizationParams(input_weight_bits=8, weight_bits=8, activation_bits=8)
model_8bit = quantize(model, qparams=qparams_8bit)

q8_path = os.path.join(MODELS_DIR, 'akidanet_vww_i8_w8_a8.h5')
model_8bit.save(q8_path, include_optimizer=False)

model_8bit.compile(metrics=['accuracy'])
_, acc_8bit = model_8bit.evaluate(val_ds, verbose=0)
print(f'8-bit quantized validation accuracy: {acc_8bit:.4f}')

# %% [markdown]
# ### 4-bit variant with QAT
#
# Quantize to 4-bit weights and activations, then fine-tune (QAT) to recover the accuracy lost at 4 bits. The intermediate PTQ model is not kept.

# %%
# --- 4-bit variant (i8 / w4 / a4), QAT ---
qparams_4bit = QuantizationParams(input_weight_bits=8, weight_bits=4, activation_bits=4)
model_4bit = quantize(model, qparams=qparams_4bit)

# QAT fine-tune the quantized 4-bit model. quantizeml-quantized models are standard
# Keras models, so the same training loop applies.
QAT_EPOCHS = 5
QAT_LR = 1e-4
train_ds, val_ds = get_data(DATA_PATH, INPUT_SHAPE, BATCH_SIZE, seed=SEED)
train_vww(model_4bit, train_ds, val_ds, QAT_EPOCHS, QAT_LR, seed=SEED)

q4_path = os.path.join(MODELS_DIR, 'akidanet_vww_i8_w4_a4_qat.h5')
model_4bit.save(q4_path, include_optimizer=False)

model_4bit.compile(metrics=['accuracy'])
_, acc_4bit = model_4bit.evaluate(val_ds, verbose=0)
print(f'4-bit QAT validation accuracy: {acc_4bit:.4f}')

# %% [markdown]
# ## Conversion to Akida Format
#
# `cnn2snn.convert` accepts quantizeml-quantized models and produces the Akida `.fbz`. We convert both variants.

# %%
from cnn2snn import convert

akida_8bit = convert(model_8bit)
akida_8bit.save(os.path.join(MODELS_DIR, 'akidanet_vww_i8_w8_a8.fbz'))

akida_4bit = convert(model_4bit)
akida_4bit.save(os.path.join(MODELS_DIR, 'akidanet_vww_i8_w4_a4_qat.fbz'))

akida_8bit.summary()


# %% [markdown]
# ## Evaluation on Akida (software backend)
#
# Run both Akida models over the validation set. No hardware is required — Akida runs on the software backend when no device is present.

# %%
def evaluate_akida(akida_model, ds):
    ds.reset()
    labels_all, logits_all = [], []
    for _ in tqdm(range(len(ds)), desc='Evaluating on Akida'):
        batch, label_batch = next(ds)
        if not isinstance(batch, np.ndarray):
            batch = batch.numpy()
        logits = akida_model.predict(batch.astype(np.uint8)).squeeze(axis=(1, 2))
        labels_all.append(label_batch)
        logits_all.append(logits)
    labels_all = np.concatenate(labels_all)
    preds = np.argmax(np.concatenate(logits_all), axis=1)
    return float(np.mean(preds == labels_all))

akida_acc_8bit = evaluate_akida(akida_8bit, val_ds)
akida_acc_4bit = evaluate_akida(akida_4bit, val_ds)
print(f'Akida 8-bit accuracy: {akida_acc_8bit:.4f}')
print(f'Akida 4-bit QAT accuracy: {akida_acc_4bit:.4f}')

# %% [markdown]
# ## Activation Sparsity
#
# Activation sparsity drives efficiency on Akida (zero activations are skipped).

# %%
from akida_models.sparsity import compute_sparsity
from brainchip_utils.plot_utils import pretty_print_sparsity
from vww_data import get_samples

NUM_SAMPLES = 1024
samples = get_samples(DATA_PATH, INPUT_SHAPE, num_samples=NUM_SAMPLES)

print('8-bit sparsity:')
pretty_print_sparsity(compute_sparsity(akida_8bit, samples=samples))
print('\n4-bit QAT sparsity:')
pretty_print_sparsity(compute_sparsity(akida_4bit, samples=samples))

# %% [markdown]
# ## Summary

# %%
print(f'{"Variant":<16}{"Keras acc":<12}{"Akida acc":<12}')
print(f'{"float":<16}{float_accuracy:<12.4f}{"-":<12}')
print(f'{"8-bit (w8a8)":<16}{acc_8bit:<12.4f}{akida_acc_8bit:<12.4f}')
print(f'{"4-bit QAT":<16}{acc_4bit:<12.4f}{akida_acc_4bit:<12.4f}')
