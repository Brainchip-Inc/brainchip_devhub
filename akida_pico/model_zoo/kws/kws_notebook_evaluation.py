# ---
# # Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: brainchip_devhub_env
#     language: python
#     name: python3
# ---

# %% [markdown]
# <img src="https://raw.githubusercontent.com/Brainchip-Inc/brainchip_devhub/main/docs/assets/0.-BC-dev-hub-LOGO-flicker.svg" alt="BrainChip Dev Hub" width="200"/>
#
# # Keyword Spotting (KWS) on Akida Pico — from weights to streaming
#
# This notebook takes the pretrained 12-class Speech Commands model through the Akida Pico
# workflow and then runs it the way Pico does: one 256-sample chunk of raw audio at a time.
#
# 1. Load the pretrained **kernelized** model (the form used for training)
# 2. Convert it to the **stateful** form Pico runs
# 3. **Quantize** it: 8-bit weights and activations, 16-bit input
# 4. Convert to Akida and **map** it on Pico
# 5. Check accuracy at each stage
# 6. **Stream** audio through the model and turn its outputs into a trigger
#
# Training is skipped: it takes 200 epochs, and `kws_train.py` reproduces it. The cells that
# need a Pico FPGA (mapping on hardware, latency) run only if one is attached; everything
# else runs in the software simulator.

# %%
# Colab-only setup. Local users: ignore this cell — it does nothing for you.
import sys, os

if 'google.colab' in sys.modules:
    if not os.path.exists('colab_setup.py'):
        !wget -q https://raw.githubusercontent.com/Brainchip-Inc/brainchip_devhub/main/akida_pico/model_zoo/kws/colab_setup.py
    import colab_setup; colab_setup.setup()

# %% [markdown]
# ## Setup
#
# The Speech Commands test set is downloaded by TensorFlow Datasets on first use (about 2 GB).
# Set `DATA_PATH` to keep it somewhere other than `~/tensorflow_datasets`.

# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
import akida

from kws_data import (LABEL_NAMES, SAMPLE_LENGTH, CHUNK_SIZE, SAMPLE_RATE,
                      load_split, get_calibration_samples)
from kws_model import load_kws_model, to_stateful, quantize_kws_model, to_akida
from kws_eval import evaluate_kernelized, evaluate_stateful, evaluate_akida

DATA_PATH = None
WEIGHTS = 'pretrained_models/tenn_recurrent_sc12.h5'
CALIBRATION = 'pretrained_models/sc12_calibration_batch100_1024samples.npz'

# Scoring all 4,890 clips takes a while for the quantized and Akida models. Use a subset for
# a quick look; the README reports the full test set.
N_EVAL = 500

# A Pico FPGA, if one is attached. Without one, mapping is shown on a software Pico and
# inference runs in the software simulator.
devices = akida.devices()
device = devices[0] if devices else None
print('Pico FPGA found' if device else 'No Pico FPGA found: using the software simulator')

# %% [markdown]
# ## Data
#
# Each test clip is **16,384 raw samples** at 16 kHz (about 1.02 s), in the int16 range.
# There is no feature extraction: Pico reads the waveform directly, so the only preparation
# is zero-padding to a fixed length. The labels are the ten keywords plus *silence* and
# *unknown*.

# %%
X_test, y_test = load_split('test', SAMPLE_LENGTH, DATA_PATH)
print(f'{len(X_test)} test clips, shape {X_test.shape}')
print('Classes:', ', '.join(f'{i}={n}' for i, n in enumerate(LABEL_NAMES)))

# A fixed, shuffled subset for the quick checks below
subset = np.random.RandomState(0).permutation(len(X_test))[:N_EVAL]
X_sub, y_sub = X_test[subset], y_test[subset]

# %% [markdown]
# ## 1. The kernelized model
#
# `tenn_recurrent_sc12` stacks six state-space (SSM) blocks on the raw waveform. In its
# **kernelized** form each block convolves the whole clip with a learned kernel, which is
# efficient to train but needs the full 16,384-sample clip at once. The model downsamples time
# by 256×, so a clip becomes 64 output steps.

# %%
model = load_kws_model(WEIGHTS)
model.summary()

# %% [markdown]
# ## 2. Stateful conversion
#
# A microphone delivers audio continuously, so Pico runs the equivalent **stateful** form: each
# `Kernelized` block becomes a recurrent block that takes a chunk, updates a small internal
# state and passes the state on to the next call. Memory and compute stay constant however long
# the stream runs.
#
# The chunk is **256 samples**: the model downsamples time by 256×, so 256 samples produce
# exactly one output step, one decision every 16 ms. The conversion is exact: same weights,
# new interface.

# %%
stateful = to_stateful(model, CHUNK_SIZE)
print('input :', stateful.input_shape, ' output:', stateful.output_shape)

# %% [markdown]
# ## 3. Quantization
#
# Pico runs integer arithmetic. `quantizeml` quantizes to **8-bit weights and activations**
# with post-training quantization, calibrated on real audio: 100 segments of 1,024 samples cut
# from the middle of training clips.
#
# The **input stays 16-bit**. The model reads raw audio samples, and 8 bits would throw away
# most of their dynamic range.

# %%
samples = get_calibration_samples(CALIBRATION)
print('calibration samples:', samples.shape, samples.dtype)

quantized = quantize_kws_model(stateful, samples)

# %% [markdown]
# ## 4. Convert to Akida and map on Pico
#
# `cnn2snn.convert` turns the quantized model into an Akida model. Mapping places it on the
# hardware: this model fits in **one sequence on one `TNP_R`**, the Pico recurrent processing
# element, so the summary should show 1 sequence and 1 `TNP_R`.

# %%
model_akida = to_akida(quantized, device)

# Without a Pico FPGA, map on a software Pico just to check the mapping. (A mapped software
# device does not run inference; the unmapped model above does, in the simulator.)
mapped = model_akida if device else to_akida(quantized, akida.PicoIP())
mapped.summary()

# %% [markdown]
# ## 5. Accuracy at each stage
#
# A stateful model is scored the way Pico runs it: one chunk at a time, state carried between
# chunks, giving 64 logit vectors per clip. The clip's prediction is the argmax of the **mean of
# those logits**. Do not average softmax probabilities.

# %%
acc_kernelized = evaluate_kernelized(model, X_sub, y_sub)
acc_stateful = evaluate_stateful(stateful, X_sub, y_sub)
acc_quantized = evaluate_stateful(quantized, X_sub, y_sub)
acc_akida = evaluate_akida(model_akida, X_sub, y_sub)

print(f'\nOn {N_EVAL} test clips:')
for name, acc in [('float, kernelized', acc_kernelized), ('float, stateful', acc_stateful),
                  ('quantized (int16 in, w8/a8)', acc_quantized), ('Akida', acc_akida)]:
    print(f'  {name:<30s} {acc * 100:6.2f}%')

# %% [markdown]
# The stateful row should match the kernelized one (the conversion is exact). A small drop at
# the quantized stage is expected. A **large** drop there usually means the calibration audio
# does not match the audio you evaluate on. The Akida row should stay within a fraction of a
# point of the quantized one.

# %% [markdown]
# ## 6. Streaming
#
# On a device nothing waits for a full second. The audio arrives in 256-sample chunks, and the
# model's state carries what it has heard so far. Here ten test clips are joined into one stream,
# and the Akida model is called once per chunk.

# %%
N_STREAM = 10
rng = np.random.RandomState(1)
order = rng.permutation(len(X_test))[:N_STREAM]
stream = np.concatenate([X_test[i] for i in order])[:, 0]          # (10 * 16384,)
stream_labels = y_test[order]

model_stream = akida.Model(model_akida.layers)                      # zero state
logits = []
for start in range(0, len(stream), CHUNK_SIZE):
    chunk = stream[start:start + CHUNK_SIZE].astype(np.int16)
    logits.append(model_stream.forward(chunk[np.newaxis, np.newaxis, :, np.newaxis]).reshape(-1))
logits = np.array(logits)                                           # (640, 12)
print('one logit vector per chunk:', logits.shape)

# %% [markdown]
# Each clip is 64 chunks. Averaging the logits over a clip's chunks gives one prediction per
# clip, as in the accuracy numbers above. (The state is not reset between clips here, as in a live stream.)

# %%
chunks_per_clip = SAMPLE_LENGTH // CHUNK_SIZE
clip_preds = [int(np.argmax(logits[i * chunks_per_clip:(i + 1) * chunks_per_clip].mean(axis=0)))
              for i in range(N_STREAM)]
print('predicted:', [LABEL_NAMES[p] for p in clip_preds])
print('truth    :', [LABEL_NAMES[t] for t in stream_labels])

# %%
t_ms = np.arange(len(logits)) * CHUNK_SIZE / SAMPLE_RATE * 1000
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax1.plot(np.arange(len(stream)) / SAMPLE_RATE * 1000, stream, lw=0.4)
ax1.set_ylabel('audio (int16)')
ax2.plot(t_ms, np.argmax(logits, axis=1), '.', ms=3)
ax2.set_yticks(range(len(LABEL_NAMES)), LABEL_NAMES)
ax2.set_xlabel('time (ms)')
ax2.set_ylabel('chunk prediction')
for k in range(1, N_STREAM):
    for ax in (ax1, ax2):
        ax.axvline(k * SAMPLE_LENGTH / SAMPLE_RATE * 1000, color='gray', ls='--', lw=0.5)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### From per-chunk outputs to a trigger
#
# A product does not use a whole second per decision. It smooths the per-chunk logits and fires
# when a keyword stays above a threshold for a few chunks. The window, threshold and run length
# below are a starting point that works reasonably on these joined test clips: it fires about once
# per keyword, but still misses some and fires on a few wrong words. Choose yours on your own
# recordings, by counting false accepts over hours of realistic background audio. A benchmark clip
# set does not tell you a product's false-accept rate.

# %%
def sliding_mean(x, window):
    kernel = np.ones(window) / window
    return np.stack([np.convolve(x[:, c], kernel)[:len(x)] for c in range(x.shape[1])], axis=1)


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


WINDOW, THRESHOLD, MIN_CHUNKS = 32, 0.9, 8
probs = softmax(sliding_mean(logits, WINDOW))

keywords = [i for i, n in enumerate(LABEL_NAMES) if n not in ('silence', 'unknown')]
run, last = 0, None
for t, p in enumerate(probs):
    best = keywords[int(np.argmax(p[keywords]))]
    above = p[best] > THRESHOLD
    run = run + 1 if (above and best == last) else int(above)
    last = best if above else None
    if run == MIN_CHUNKS:
        print(f'{t_ms[t]:8.0f} ms  trigger: {LABEL_NAMES[best]}')

# %% [markdown]
# ## Latency on the Pico FPGA
#
# If a Pico FPGA is attached, the model's hardware counters give its cycles per chunk. The cycle
# count is fixed for a given model and mapping, so the latency at any other clock follows
# from it. Real time needs one chunk every 16 ms.

# %%
FPGA_CLOCK_HZ = 50e6

if device is None:
    print('No Pico FPGA attached: skipping the latency measurement.')
else:
    frames = X_test.reshape(-1, CHUNK_SIZE, 1)[:1000].astype(np.int16)[:, np.newaxis]
    model_akida.forward(frames)
    cycles = model_akida.metrics['inference_clk'] / len(frames)
    print(f'{cycles:.0f} cycles per chunk -> {cycles / FPGA_CLOCK_HZ * 1000:.3f} ms at '
          f'{FPGA_CLOCK_HZ / 1e6:.0f} MHz (chunk period: {CHUNK_SIZE / SAMPLE_RATE * 1000:.0f} ms)')
