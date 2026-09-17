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
#     display_name: brainchip_devhub_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# <img src="https://raw.githubusercontent.com/Brainchip-Inc/brainchip_devhub/main/docs/assets/0.-BC-dev-hub-LOGO-flicker.svg" alt="BrainChip Dev Hub" width="200"/>
#
# # PlantVillage Disease Classification — Akida 2 Training
#
# Run Time: ~20 minutes with training included / TBD with training skipped
#
# This notebook walks through the complete pipeline to train, quantize, convert, and evaluate an AkidaNet model on the **PlantVillage** dataset for Akida 2 hardware.
#
# The PlantVillage dataset has 54,303 leaf images across 38 categories (14 crop species plus disease and healthy variants).
#
# The pipeline follows the standard Akida workflow:
# 1. Train a float model
# 2. 8-bit quantization (PTQ)
# 3. 4-bit quantization with QAT fine-tuning
# 4. Conversion to Akida `.fbz` format
# 5. Evaluation on Akida

# %%
# Colab-only setup. Local users: ignore this cell — it does nothing for you.
import sys, os

if 'google.colab' in sys.modules:
    if not os.path.exists('colab_setup.py'):
        !wget -q https://raw.githubusercontent.com/Brainchip-Inc/brainchip_devhub/main/akida2/model_zoo/plant_village/colab_setup.py
    import colab_setup; colab_setup.setup()

# %% [markdown]
# ## Setup
#
# The default dataset path is `./data/plant_village`; the dataset is downloaded automatically via TensorFlow Datasets on first use (see the Dataset section below). The `RUN_FLOAT_TRAINING` and `RUN_QAT_TRAINING` flags control whether the training steps run locally or load an existing model. `enable_op_determinism()` is called before any TF ops so results are reproducible on the same hardware.

# %%
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

DATA_PATH = './data/plant_village'
MODELS_DIR = './models'
os.makedirs(MODELS_DIR, exist_ok=True)

RUN_FLOAT_TRAINING = True
RUN_QAT_TRAINING = True

SEED = 42

# Must be called before any TF ops to make GPU ops deterministic.
tf.config.experimental.enable_op_determinism()

# %% [markdown]
# ## Dataset
#
# The **PlantVillage** dataset is loaded via TensorFlow Datasets (`plant_village`). On the first run, TFDS automatically downloads and prepares the dataset to `DATA_PATH`; subsequent runs read from the local cache.
#
# The dataset is split 80/10/10 (train/val/test). Images are resized from their variable original sizes to **224 × 224 RGB** and delivered as uint8 pixel values (0–255). Training applies random horizontal flip, brightness jitter, and contrast jitter for regularisation. `get_data` returns all three splits; we use train and validation here.

# %%
from plant_village_data import get_data

BATCH_SIZE = 32
INPUT_SHAPE = (224, 224, 3)

train_ds, val_ds, test_ds = get_data(DATA_PATH, INPUT_SHAPE, BATCH_SIZE, seed=SEED)

# %% [markdown]
# ## Model
#
# The model is based on **AkidaNet** (`akida_models.akidanet_imagenet`) with:
# - Width multiplier **alpha = 0.5** — provides sufficient capacity for the 38 classes while remaining efficient on Akida hardware
# - Input resolution **224 × 224 RGB**
# - A **38-class** classification head replacing the ImageNet top layers, ending in a softmax activation
# - **Input scaling** built into the model (a Rescaling layer) — the pipeline delivers raw uint8 pixel values and the model normalises them internally
#
# It uses transfer learning: the AkidaNet backbone is initialised from ImageNet-pretrained weights, then fine-tuned on PlantVillage. AkidaNet is designed for Akida hardware, using only operations that map efficiently to Akida Neural Processors (NPs) — depthwise separable convolutions and ReLU activations. The model is built under `set_akida_version(AkidaVersion.v2)`.

# %%
from plant_village_model import build_plant_village_model

model = build_plant_village_model(seed=SEED)
model.summary()

# %% [markdown]
# ## Float Training
#
# The model is fine-tuned in full float32 precision using the Adam optimiser and sparse categorical cross-entropy loss. Because the model head ends in a **softmax** activation (it outputs probabilities, not raw logits), the loss is configured with `from_logits=False`.
#
# The learning rate follows an **exponential decay** schedule, decaying continuously from the initial rate to ~1% of it by the final epoch. Since the backbone is already pretrained, only a modest number of fine-tuning epochs is needed.
#
# Set `RUN_FLOAT_TRAINING = True` above to train locally; otherwise the cell below loads an existing float model from `pretrained_models/`.

# %%
from plant_village_train import train_plant_village

if RUN_FLOAT_TRAINING:
    LEARNING_RATE = 1e-3
    EPOCHS = 10
    train_ds, val_ds, test_ds = get_data(DATA_PATH, INPUT_SHAPE, BATCH_SIZE, seed=SEED)

    train_plant_village(model, train_ds, val_ds, EPOCHS, LEARNING_RATE, seed=SEED)

    float_model_path = os.path.join(MODELS_DIR, 'akidanet_plant_village.h5')
    model.save(float_model_path, include_optimizer=False)
    print(f'Float model saved to {float_model_path}')
else:
    from tf_keras.models import load_model
    print('Training skipped. Loading an existing float model...')
    model = load_model(os.path.join('pretrained_models', 'akidanet_plant_village.h5'))

# %% [markdown]
# ### Evaluate float model

# %%
model.compile(metrics=['accuracy'])
_, float_accuracy = model.evaluate(val_ds, verbose=0)
print(f'Float validation accuracy: {float_accuracy:.4f}')

# %% [markdown]
# ## Quantization (quantizeml)
#
# Akida 2 quantizes with **`quantizeml`** (not `cnn2snn.quantize`, which targets Akida 1). Quantization converts the model to fixed-point arithmetic; `quantizeml` calibrates the quantization ranges using representative samples, so a batch of real data is passed to `quantize`. Ideally these samples are drawn from the training split to avoid data leakage.
#
# We produce **two** variants:
# - **8-bit** (i8 / w8 / a8) — post-training quantization (PTQ) only. At 8 bits the accuracy drop from float is typically negligible, so no further training is required.
# - **4-bit** (i8 / w4 / a4) — quantized more aggressively, then recovered with QAT (below). The **input layer weights stay 8-bit** in both variants, which the hardware supports and which protects accuracy at the sensitive first layer.
#
# We start with the 8-bit variant.

# %%
from quantizeml.models import quantize, QuantizationParams

from plant_village_data import get_samples

NUM_SAMPLES = 1024
samples = get_samples(DATA_PATH, INPUT_SHAPE, num_samples=NUM_SAMPLES)

# --- 8-bit variant (i8 / w8 / a8), PTQ only ---
qparams_8bit = QuantizationParams(input_weight_bits=8, weight_bits=8, activation_bits=8)
model_8bit = quantize(model, qparams=qparams_8bit, samples=samples, batch_size=100, epochs=2)

q8_path = os.path.join(MODELS_DIR, 'akidanet_plant_village_i8_w8_a8.h5')
model_8bit.save(q8_path, include_optimizer=False)

model_8bit.compile(metrics=['accuracy'])
_, acc_8bit = model_8bit.evaluate(val_ds, verbose=0)
print(f'8-bit quantized validation accuracy: {acc_8bit:.4f}')

# %% [markdown]
# Note: a quantized model is saved with the standard Keras `save()`, but because it contains custom quantized layers it must be reloaded with `load_model` from `quantizeml.model_io` (a wrapper around the standard tf_keras loader that registers those layers). The cell below illustrates this.

# %%
# Just to illustrate how to load a quantized model
del model_8bit

from quantizeml.model_io import load_model
model_8bit = load_model(q8_path)

# %% [markdown]
# ### 4-bit variant with QAT
#
# At 4 bits, post-training quantization alone usually causes a noticeable accuracy drop, because the model was trained with continuous weights but is now constrained to a small set of discrete values. **Quantization-Aware Training (QAT)** recovers most of that loss by fine-tuning the quantized model for a few epochs at a reduced learning rate (`1e-4`), so the weights adapt to the quantization.
#
# Although QAT can sound intimidating, with `quantizeml` it is no more complex than sending the quantized model back through the same training function used for the float model. Because 4-bit PTQ accuracy is poor, we do not store or report a PTQ-only 4-bit model — we go straight to the QAT result.
#
# Set `RUN_QAT_TRAINING = True` above to run QAT locally; otherwise the cell below loads an existing QAT model from `pretrained_models/`.

# %%
if RUN_QAT_TRAINING:
    # --- 4-bit variant (i8 / w4 / a4), QAT ---
    qparams_4bit = QuantizationParams(input_weight_bits=8, weight_bits=4, activation_bits=4)
    model_4bit = quantize(model, qparams=qparams_4bit, samples=samples, batch_size=100, epochs=2)

    QAT_EPOCHS = 20
    QAT_LR = 1e-4
    train_ds, val_ds, test_ds = get_data(DATA_PATH, INPUT_SHAPE, BATCH_SIZE, seed=SEED)
    train_plant_village(model_4bit, train_ds, val_ds, QAT_EPOCHS, QAT_LR, seed=SEED)

    q4_path = os.path.join(MODELS_DIR, 'akidanet_plant_village_i8_w4_a4_qat.h5')
    model_4bit.save(q4_path, include_optimizer=False)
else:
    from quantizeml.model_io import load_model
    print('Training skipped. Loading an existing 4-bit model...')
    model_4bit = load_model(os.path.join('pretrained_models', 'akidanet_plant_village_i8_w4_a4_qat.h5'))

model_4bit.compile(metrics=['accuracy'])
_, acc_4bit = model_4bit.evaluate(val_ds, verbose=0)
print(f'4-bit QAT validation accuracy: {acc_4bit:.4f}')

# %% [markdown]
# ## Conversion to Akida Format
#
# `cnn2snn.convert` compiles a quantized Keras model into an Akida `.fbz` model that can be loaded and executed directly on Akida 2 hardware. The converter verifies hardware compatibility and maps each layer to its corresponding Akida primitive. `cnn2snn.convert` accepts `quantizeml`-quantized models directly. We convert both quantized variants.
#
# The softmax head is not converted (Akida stops at the preceding dequantizer) — this is expected for a classification head, since argmax over logits and over softmax probabilities gives the same prediction.

# %%
from cnn2snn import convert

akida_8bit = convert(model_8bit)
akida_8bit.save(os.path.join(MODELS_DIR, 'akidanet_plant_village_i8_w8_a8.fbz'))

akida_4bit = convert(model_4bit)
akida_4bit.save(os.path.join(MODELS_DIR, 'akidanet_plant_village_i8_w4_a4_qat.fbz'))

akida_8bit.summary()


# %% [markdown]
# ## Evaluation on Akida (software backend)
#
# We now run evaluation through the Akida models to check that accuracy is comparable to the quantized tf_keras models. Here we deliberately use the **software backend** (the default, since we do not check for or map to a connected hardware device): this delivers a bit-accurate simulation of the results that will be obtained when running on hardware.
#
# In the accompanying [plant_village_notebook_benchmark.ipynb](plant_village_notebook_benchmark.ipynb) the same evaluation is run using the hardware backend (if an Akida 2 device is connected), allowing you to confirm the results are identical.

# %% [markdown]
# ### Run Evaluation on Akida
#
# The Akida runtime cannot consume `tf.data.Dataset` objects directly — it expects 4D numpy arrays `(n, h, w, c)` in uint8 format. So we iterate over the validation batches manually and convert each to uint8. The model output tensor has shape `(B, 1, 1, C)`, which is squeezed to `(B, C)` before taking the class argmax.

# %%
def evaluate_akida(akida_model, ds):
    labels_all, logits_all = [], []
    for batch, label_batch in tqdm(ds, desc='Evaluating on Akida'):
        if not isinstance(batch, np.ndarray):
            batch = batch.numpy()
        if not isinstance(label_batch, np.ndarray):
            label_batch = label_batch.numpy()
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
# Akida hardware skips computation for zero-valued activations, so activation sparsity directly reduces both energy consumption and inference latency. Below we measure per-layer sparsity for each variant on a batch of real samples (reusing the calibration samples generated earlier), since sparsity is activity-dependent and must be measured on representative inputs.

# %%
from akida_models.sparsity import compute_sparsity
from brainchip_utils.plot_utils import pretty_print_sparsity

print('8-bit sparsity:')
pretty_print_sparsity(compute_sparsity(akida_8bit, samples=samples))
print('\n4-bit QAT sparsity:')
pretty_print_sparsity(compute_sparsity(akida_4bit, samples=samples))

# %% [markdown]
# ## Summary
#
# The table below compares validation accuracy across the model variants. The goal is that the quantized and Akida accuracies remain close to the float baseline — confirming that quantization and conversion preserve the model's behaviour.

# %%
print(f'{"Variant":<16}{"Keras acc":<12}{"Akida acc":<12}')
print(f'{"float":<16}{float_accuracy:<12.4f}{"-":<12}')
print(f'{"8-bit (w8a8)":<16}{acc_8bit:<12.4f}{akida_acc_8bit:<12.4f}')
print(f'{"4-bit QAT":<16}{acc_4bit:<12.4f}{akida_acc_4bit:<12.4f}')
