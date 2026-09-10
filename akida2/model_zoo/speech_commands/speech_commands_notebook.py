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
# # Speech Commands Keyword Spotting (KWS) — Akida 2 Training
#
# Run Time: ~20 minutes with training included / TBD with training skipped
#
# This notebook walks through the complete pipeline to train, quantize, convert, and evaluate a DS-CNN model on the **Speech Commands** dataset for Akida 2 hardware.
#
# The Speech Commands dataset has 10 classes (yes, no, up, down, left, right, on, off, stop, go) plus silence and unknown.
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
        !wget -q https://raw.githubusercontent.com/Brainchip-Inc/brainchip_devhub/main/akida2/model_zoo/speech_commands/colab_setup.py
    import colab_setup; colab_setup.setup()

# %% [markdown]
# ## Setup
#
# The default dataset path is `./data/speech_commands`. The dataset is downloaded automatically via TensorFlow Datasets on first use; see the [README](README.md) for details. Update `DATA_PATH` below if you keep the dataset elsewhere.

# %%
import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

DATA_PATH = './data/speech_commands'
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
# Speech Commands (SC12) is loaded via TensorFlow Datasets and converted to MFCC features of shape `(49, 10, 1)`. `get_data` returns three splits `(train, test, val)`. MFCC features are scaled to uint8 using percentile bounds computed from the training split (`compute_mfcc_range`), matching the model's uint8 input.

# %%
from speech_commands_data import get_data, compute_mfcc_range

BATCH_SIZE = 100
INPUT_SHAPE = (49, 10, 1)

# Percentile-based MFCC range from the train split, used to scale features to uint8.
data_transform = compute_mfcc_range(data_dir=DATA_PATH)

train_ds, test_ds, val_ds = get_data(DATA_PATH, batch_size=BATCH_SIZE,
                                     data_transform=data_transform, seed=SEED)

# %% [markdown]
# ## Model
#
# The model is a **DS-CNN** (depthwise-separable CNN) from `akida_models`, a lightweight architecture for keyword spotting drawn from the MLPerf Tiny benchmark. It is built for Akida 2 under `set_akida_version(AkidaVersion.v2)`. A Rescaling layer is included up front so the model accepts raw uint8 MFCC features directly: folding the rescaling into the model graph (rather than the preprocessing pipeline) means the same inputs feed the float and Akida versions, and the step is automatically folded into the Akida layer parameters at conversion.

# %%
from speech_commands_model import build_speech_commands_model

model = build_speech_commands_model(seed=SEED)
model.summary()

# %% [markdown]
# ## Float Training
#
# The model is trained with Adam and a cosine-decay learning-rate schedule (with warmup). The full training code is in [speech_commands_train.py](speech_commands_train.py) — standard tf_keras code, not described further here.

# %%
from speech_commands_train import train_speech_commands

if RUN_FLOAT_TRAINING:
    LEARNING_RATE = 1e-3
    EPOCHS = 16
    # Freshly build the datasets for reproducibility
    train_ds, test_ds, val_ds = get_data(DATA_PATH, batch_size=BATCH_SIZE,
                                         data_transform=data_transform, seed=SEED)

    train_speech_commands(model, train_ds, val_ds, EPOCHS, LEARNING_RATE, seed=SEED)

    float_model_path = os.path.join(MODELS_DIR, 'ds_cnn_speech_commands.h5')
    model.save(float_model_path, include_optimizer=False)
    print(f'Float model saved to {float_model_path}')
else:
    from tf_keras.models import load_model
    print('Training skipped. Loading an existing float model...')
    model = load_model(os.path.join('pretrained_models', 'ds_cnn_speech_commands.h5'))

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
#
# Note that quantization using `quantizeml` requires samples for calibration. Ideally those should be representative samples for the task, drawn from the training split to avoid data leakage.

# %%
from quantizeml.models import quantize, QuantizationParams

from speech_commands_data import get_samples

# quantizeml calibrates during quantization and needs real, representative samples.
# We build them from the dataset (validation split) here so the process is visible;
# the speech_commands_train.sh pipeline instead uses the published kws_batch1024.npz pack.
NUM_SAMPLES = 1024
samples = get_samples(DATA_PATH, data_transform=data_transform, num_samples=NUM_SAMPLES)

# --- 8-bit variant (i8 / w8 / a8), PTQ only ---
qparams_8bit = QuantizationParams(input_weight_bits=8, weight_bits=8, activation_bits=8)
model_8bit = quantize(model, qparams=qparams_8bit, samples=samples, batch_size=100, epochs=2)

q8_path = os.path.join(MODELS_DIR, 'ds_cnn_speech_commands_i8_w8_a8.h5')
model_8bit.save(q8_path, include_optimizer=False)

model_8bit.compile(metrics=['accuracy'])
_, acc_8bit = model_8bit.evaluate(val_ds, verbose=0)
print(f'8-bit quantized validation accuracy: {acc_8bit:.4f}')

# %% [markdown]
# Note: the quantized model can be saved using the standard method. However,
# for later reloading, because of the custom quantized layers in the model
# we have to use the `load_model` function from `quantizeml.model_io` (a wrapper
# around the standard tf_keras loading function)

# %%
# Just to illustrate how to load a quantized model
del model_8bit

from quantizeml.model_io import load_model
model_8bit = load_model(q8_path)

# %% [markdown]
# ### 4-bit variant with QAT
#
# While 8-bit quantization usually results in a negligible loss of accuracy, that is rarely the case when quantizing aggressively to 4-bits. However, a few epochs of fine-tuning are typically sufficient to recover most of the accuracy lost. It is typical to find that a lower learning rate (e.g. /10) is required during this phase than during the initial training.
#
# Note that, although Quantization Aware Training can sound intimidating,
# the model quantized via `quantizeml` can simply be reinserted into the 
# same training function that was used for the initial float training.

# %%
if RUN_QAT_TRAINING:
    # --- 4-bit variant (i8 / w4 / a4), QAT ---
    qparams_4bit = QuantizationParams(input_weight_bits=8, weight_bits=4, activation_bits=4)
    model_4bit = quantize(model, qparams=qparams_4bit, samples=samples, batch_size=100, epochs=2)

    # QAT fine-tune the quantized 4-bit model. quantizeml-quantized models are standard
    # Keras models, so the same training loop applies.
    QAT_EPOCHS = 16
    QAT_LR = 1e-4
    train_ds, test_ds, val_ds = get_data(DATA_PATH, batch_size=BATCH_SIZE,
                                         data_transform=data_transform, seed=SEED)
    train_speech_commands(model_4bit, train_ds, val_ds, QAT_EPOCHS, QAT_LR, seed=SEED)

    q4_path = os.path.join(MODELS_DIR, 'ds_cnn_speech_commands_i8_w4_a4_qat.h5')
    model_4bit.save(q4_path, include_optimizer=False)
else:
    from quantizeml.model_io import load_model
    print('Training skipped. Loading an existing 4-bit model...')
    model_4bit = load_model(os.path.join('pretrained_models', 'ds_cnn_speech_commands_i8_w4_a4_qat.h5'))

model_4bit.compile(metrics=['accuracy'])
_, acc_4bit = model_4bit.evaluate(val_ds, verbose=0)
print(f'4-bit QAT validation accuracy: {acc_4bit:.4f}')

# %% [markdown]
# ## Conversion to Akida Format
#
# `cnn2snn.convert` compiles the quantized Keras model into an Akida `.fbz`
# model that can be loaded and executed directly on an Akida 2 hardware device.
# The converter verifies hardware compatibility and maps each layer to its
# corresponding Akida primitive. We convert both quantized variants.

# %%
from cnn2snn import convert

akida_8bit = convert(model_8bit)
akida_8bit.save(os.path.join(MODELS_DIR, 'ds_cnn_speech_commands_i8_w8_a8.fbz'))

akida_4bit = convert(model_4bit)
akida_4bit.save(os.path.join(MODELS_DIR, 'ds_cnn_speech_commands_i8_w4_a4_qat.fbz'))

akida_8bit.summary()


# %% [markdown]
# ## Evaluation on Akida (software backend)
#
# We now run evaluation through the Akida model, to check that accuracy is 
# comparable to that obtained from the quantized tf_keras model. Here, we deliberately use the software backend (the default, since we do not check 
# for and map to a connected hardware device): this delivers a bit-accurate 
# simulation of the results that will be obtained when running the model on
# hardware.

# %% [markdown]
# ### Run Evaluation on Akida
#
# The Akida runtime cannot consume `tf.data.Dataset` objects directly, rather
# it expects a 4D numpy array (n, h, w, c). So we iterate over validation 
# batches manually.
#
# The model output tensor has shape `(B, 1, 1, C)` which is squeezed to 
# `(B, C)` before taking the class argmax.

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
# Activation sparsity drives efficiency on Akida (zero activations are skipped). To check activation sparsity within the model, we need to run some real samples through. Here, we re-use the samples that we generated earlier for calibration during quantization.

# %%
from akida_models.sparsity import compute_sparsity
from brainchip_utils.plot_utils import pretty_print_sparsity

# Reuse the calibration samples built earlier (sparsity needs real activations too).
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
