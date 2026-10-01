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
# # Camera-Less Eye Tracking — Training (Akida 2)
#
# Trains a causal, streaming Tiny Temporal CNN to estimate (x, y) gaze angle from a simulated 8-channel photodiode array, then quantizes it with `quantizeml` and converts it to Akida 2 format. See `README.md` for the full dataset/model background — this notebook focuses on *running* the pipeline and explaining the Akida-specific steps.
#

# %%
# Colab-only setup. Local users: ignore this cell — it does nothing for you.
import sys, os

if 'google.colab' in sys.modules:
    if not os.path.exists('colab_setup.py'):
        !wget -q https://raw.githubusercontent.com/Brainchip-Inc/brainchip_devhub/main/akida2/model_zoo/eyetracking/colab_setup.py
    import colab_setup; colab_setup.setup()


# %% [markdown]
# ## Setup
#
# The dataset must already be built (see README.md "Dataset setup") at `./data/processed` — unlike the image-classification examples in this zoo, there is no single tarball to fetch.
#

# %%
import os
import numpy as np
import tensorflow as tf

DATA_PATH = './data/processed'
MODELS_DIR = './models'
os.makedirs(MODELS_DIR, exist_ok=True)
SEED = 42
WINDOW, CHANNELS = 64, 8

RUN_FLOAT_TRAINING = False  # set True to train from scratch instead of downloading

tf.config.experimental.enable_op_determinism()


# %% [markdown]
# ## Dataset
#
# `get_data` returns `(X_train, y_train), (X_val, y_val)`: signed int8 windows of shape `(N, 64, 1, 1, 8)` and float32 `(x, y)` gaze targets of shape `(N, 64, 1, 1, 2)` — one gaze value **per timestep**, not one per window (see "Why streaming evaluation" in README.md). The signed-int8 input is required by Akida's `BufferTempConv` primitive, not the uint8 convention used by the image examples in this zoo.
#

# %%
from eyetracking_data import get_data, get_samples

train_data, val_data = get_data(DATA_PATH)
print('train:', train_data[0].shape, train_data[1].shape)
print('val:  ', val_data[0].shape, val_data[1].shape)


# %% [markdown]
# ## Model
#
# Causal Conv3D(16, k=5) → BN → ReLU → causal Conv3D(32, k=3) → BN → ReLU → Dense(32) → ReLU → Dense(2). Only **3,490 parameters**. The Conv3D layers use `(k, 1, 1)` kernels over a `(window, 1, 1, channels)` input — a 5D shape with trivial height/width dims, since there is no spatial structure here, just channels over time. `quantizeml`'s `replace_conv3d` turns them into `BufferTempConv` automatically at quantization time.
#

# %%
from eyetracking_model import build_eyetracking_model

model = build_eyetracking_model(seed=SEED)
model.summary()
print(f'total params: {model.count_params():,}')


# %% [markdown]
# ## Float Training
#
# Regression (MSE loss) on z-normalized targets, normalized using all-timestep statistics from the training split. `EarlyStopping` + `ReduceLROnPlateau` on validation loss, same as `eyetracking_train.py`.
#

# %%
from eyetracking_train import train_eyetracking

if RUN_FLOAT_TRAINING:
    history, y_mean, y_std = train_eyetracking(
        model, train_data, val_data, epochs=30, learning_rate=1e-3, seed=SEED)
    model.save(os.path.join(MODELS_DIR, 'tinytemporalcnn_eyetracking.h5'), include_optimizer=False)
    np.savez(os.path.join(MODELS_DIR, 'tinytemporalcnn_eyetracking.h5.yscaler.npz'), mean=y_mean, std=y_std)
else:
    import shutil
    # No published pooch URL yet for this example -- use the checked-in
    # pretrained_models/ copy instead (see README.md "Reference Models").
    shutil.copy('pretrained_models/tinytemporalcnn_eyetracking.h5',
                os.path.join(MODELS_DIR, 'tinytemporalcnn_eyetracking.h5'))
    shutil.copy('pretrained_models/tinytemporalcnn_eyetracking.h5.yscaler.npz',
                os.path.join(MODELS_DIR, 'tinytemporalcnn_eyetracking.h5.yscaler.npz'))
    scaler = np.load(os.path.join(MODELS_DIR, 'tinytemporalcnn_eyetracking.h5.yscaler.npz'))
    y_mean, y_std = scaler['mean'], scaler['std']


# %% [markdown]
# ### Evaluate float model
#
# Evaluation must feed **one timestep at a time** (the model is causal/streaming), not a normal batched `model.evaluate` call — see `eyetracking_eval.py:predict_streaming_keras`.
#

# %%
from eyetracking_eval import angular_error_deg, predict_streaming_keras

import tf_keras as keras
model = keras.models.load_model(os.path.join(MODELS_DIR, 'tinytemporalcnn_eyetracking.h5'))

X_val, y_val = val_data
y_pred = predict_streaming_keras(model, X_val[:2000]) * y_std + y_mean
y_true = y_val[:2000, :, 0, 0, :]
print(f"Float overall mean angular error: {angular_error_deg(y_true, y_pred):.3f} deg")
print(f"Float last-timestep mean angular error: {angular_error_deg(y_true[:, -1], y_pred[:, -1]):.3f} deg")


# %% [markdown]
# ## Quantization (quantizeml)
#
# Akida 2 quantizes with **quantizeml** (not `cnn2snn.quantize`). 8-bit weights and activations already cost very little accuracy for this model, so no QAT fine-tuning is needed (unlike the VWW example's 4-bit variant). Calibration samples must be **individual frames**, not whole windows — `get_samples()` flattens the window axis away for exactly this reason.
#

# %%
from quantizeml.models import quantize, QuantizationParams

calib_samples = get_samples(DATA_PATH, num_samples=1024)
qparams = QuantizationParams(input_weight_bits=8, weight_bits=8, activation_bits=8,
                              input_dtype='int8')
q_model = quantize(model, qparams=qparams, samples=calib_samples, num_samples=1024, batch_size=128)
q_model.summary()


# %% [markdown]
# ### Evaluate quantized model
#

# %%
from quantizeml.models import reset_buffers

y_pred_q = predict_streaming_keras(q_model, X_val[:2000]) * y_std + y_mean
print(f"Quantized overall mean angular error: {angular_error_deg(y_true, y_pred_q):.3f} deg")
print(f"Quantized last-timestep mean angular error: {angular_error_deg(y_true[:, -1], y_pred_q[:, -1]):.3f} deg")


# %% [markdown]
# ## Conversion to Akida Format
#
# `cnn2snn.convert` compiles the quantized Keras model into an Akida model. `quantizeml`'s `replace_conv3d` transform (applied automatically during `quantize()` above) already turned the Conv3D layers into `BufferTempConv` — the FIFO-buffered primitive that makes this genuinely a streaming model on hardware, not just in this notebook's evaluation loop.
#

# %%
from cnn2snn import convert

akida_model = convert(q_model)
akida_model.save(os.path.join(MODELS_DIR, 'tinytemporalcnn_eyetracking_i8_w8_a8.fbz'))
akida_model.summary()


# %% [markdown]
# ## Evaluation on Akida (software backend)
#
# Akida's buffered/FIFO layers only maintain correct state for **batch_size=1** — sharing one FIFO across a batch of independent windows silently produces garbage. Each window below gets a fresh model instance (resets FIFO state) and is fed one timestep at a time. This is expensive, so only a small subsample is evaluated here.
#

# %%
from eyetracking_eval import evaluate_akida_model

n_eval = 50  # keep small for notebook runtime; eyetracking_eval.py defaults to 300
X_eval = X_val[:n_eval]
y_eval = y_true[:n_eval]

y_pred_akida, akida_model = evaluate_akida_model(akida_model, X_eval)
y_pred_akida = y_pred_akida * y_std + y_mean
print(f"Akida overall mean angular error: {angular_error_deg(y_eval, y_pred_akida):.3f} deg")
print(f"Akida last-timestep mean angular error: {angular_error_deg(y_eval[:, -1], y_pred_akida[:, -1]):.3f} deg")


# %% [markdown]
# ## Activation Sparsity
#

# %%
from akida_models.sparsity import compute_sparsity
from brainchip_utils.plot_utils import pretty_print_sparsity

sparsity_dict = compute_sparsity(akida_model, samples=calib_samples)
pretty_print_sparsity(sparsity_dict)


# %% [markdown]
# ## Summary
#

# %%
print(f'{"Stage":<24}{"Overall (deg)":<16}{"Last-timestep (deg)":<20}')
print(f'{"Float":<24}{angular_error_deg(y_true, y_pred):<16.3f}{angular_error_deg(y_true[:, -1], y_pred[:, -1]):<20.3f}')
print(f'{"Quantized (w8a8)":<24}{angular_error_deg(y_true, y_pred_q):<16.3f}{angular_error_deg(y_true[:, -1], y_pred_q[:, -1]):<20.3f}')
print(f'{"Akida":<24}{angular_error_deg(y_eval, y_pred_akida):<16.3f}{angular_error_deg(y_eval[:, -1], y_pred_akida[:, -1]):<20.3f}')

