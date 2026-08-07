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
# # PlantVillage Disease Classification
#
# <p align="right">
# Run Time: ~20 minutes with training included / ~15 minutes with training skipped
# </p>
#
# This notebook walks through the complete pipeline to train, quantize, convert, and
# benchmark an AkidaNet model on the **PlantVillage** dataset for Akida 1 hardware.
#
# PlantVillage contains 54,303 images of healthy and diseased plant leaves across
# 38 categories (14 crop species × multiple disease types plus healthy variants).
# The task is a 38-class image classification problem: given a 224×224 RGB image of
# a leaf, identify the crop species and disease (or healthy state).
#
# The pipeline follows the standard Akida workflow:
# 1. Train a float model
# 2. Post-training quantization (PTQ)
# 3. Quantization-aware training (QAT) fine-tuning
# 4. Conversion to Akida `.fbz` format
# 5. Hardware evaluation and benchmarking

# %%
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

# %% [markdown]
# ## Dataset
#
# The **PlantVillage** dataset is loaded via TensorFlow Datasets (`plant_village`).
# On the first run, TFDS will automatically download and prepare the dataset to
# `DATA_PATH`. Subsequent runs read from the local cache.
#
# The dataset is split 80/10/10 (train/val/test). Images are resized from variable
# original sizes to **224 × 224 RGB** and delivered as uint8 pixel values (0–255).
# Training applies random horizontal flip, brightness jitter, and contrast jitter
# for regularisation.
#
# To pre-download without training, run:
# ```bash
# python -c "import tensorflow_datasets as tfds; tfds.load('plant_village', data_dir='./data/plant_village')"
# ```
# %%
import sys, os

if 'google.colab' in sys.modules:
    if not os.path.exists('colab_setup.py'):
        !wget -q https://raw.githubusercontent.com/Brainchip-Inc/brainchip_devhub/main/akida1/model_zoo/plant_village/colab_setup.py
    import colab_setup; colab_setup.setup()
# %%
from plant_village_data import get_data

INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32

train_ds, val_ds, test_ds = get_data(DATA_PATH, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, seed=SEED)
print('Datasets ready.')

# %% [markdown]
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

# %%
from plant_village_model import build_plant_village_model

model = build_plant_village_model(seed=SEED)
model.summary()

# %% [markdown]
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
# cell below downloads a pre-trained float model directly from BrainChip servers.

# %%
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

# %%
_, float_acc = model.evaluate(test_ds, verbose=1)
print(f'Float accuracy: {float_acc:.4f}')

# %% [markdown]
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

# %%
import cnn2snn

quantized_model = cnn2snn.quantize(
    model,
    input_weight_quantization=8,
    weight_quantization=4,
    activ_quantization=4)
print('Model quantized to i8/w4/a4.')

# %% [markdown]
# Quantizing a model after training like this is referred to as Post-Training
# Quantization (PTQ). It can slightly reduce accuracy (especially at 4-bits as
# here) because the model was trained with continuous weights but is now 
# evaluated with discrete values.

# %%
quantized_model.compile(metrics=['accuracy'])
_, ptq_acc = quantized_model.evaluate(test_ds, verbose=1)
print(f'PTQ accuracy: {ptq_acc:.4f}')

# %% [markdown]
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
# below downloads a pre-trained QAT model from BrainChip servers.

# %%
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

# %%
_, qat_acc = quantized_model.evaluate(test_ds, verbose=1)
print(f'QAT accuracy: {qat_acc:.4f}')

# %% [markdown]
# ## Conversion to Akida Format
#
# `cnn2snn.convert` compiles the quantized Keras model into an Akida `.fbz`
# model that can be loaded and executed directly on AKD1500 hardware.
# The converter verifies hardware compatibility and maps each layer to its
# corresponding Akida primitive.

# %%
akida_model = cnn2snn.convert(quantized_model)

akida_model_path = os.path.join(MODELS_DIR, 'akidanet_plant_village_qat.fbz')
akida_model.save(akida_model_path)
print(f'Akida model saved to {akida_model_path}')
akida_model.summary()


# %% [markdown]
# ## Evaluation of Akida Model
#
# We now run evaluation through the Akida model, to check that accuracy is 
# comparable to that obtained from the quantized tf_keras model. If an Akida 1
# hardware device is connected, it will be used for inference; if not, the
# code will fall back to using the software backend: this delivers a
# bit-accurate simulation of the results that will be obtained when running
# the model on hardware. Let's run that check before going any further
#
# ### Check for a connected Akida hardware device
#
# We can use the `akida.devices()` function to detect connected hardware devices.
# That returns a list - if it's empty, there were no hardware devices. Otherwise, 
# typically we'd only have a single Akida device connected on a given machine, 
# and we can just select the first (and only) device returned.

# %%
import akida
devices = akida.devices()
if len(devices)>0:
    # Hardware is available
    device = devices[0]
else:
    # Hardware is not available
    device = None

# %% [markdown]
# In the present case, we want to be a bit more careful and ensure that the
# device is the right version for the model we want to test (here, Akida IP
# version 1). We'll import a local function to do that - check out the details 
# if interested

# %%
from brainchip_utils.hardware_utils import get_akida_device

# Load the Akida model
akida_model = akida.Model(akida_model_path)
# Look for a matching hardware device
device = get_akida_device(target_version = akida_model.ip_version)
if device is not None:
    akida_model.map(device, mode=akida.MapMode.Minimal, hw_only=True)


# %% [markdown]
# ### Run Evaluation on Akida
#
# The Akida runtime cannot consume `tf.data.Dataset` objects directly, rather
# it expects a 4D numpy array (n, h, w, c) in uint8 format. So we
# iterate over validation batches manually.
#
# The model output tensor has shape `(B, 1, 1, C)` which is squeezed to 
# `(B, C)` before taking the class argmax.

# %%
from tqdm import tqdm

labels_all = []
logits_all = []
for batch, label_batch in tqdm(test_ds, desc="Evaluating on Akida"):
    if not isinstance(batch, np.ndarray):
        batch = batch.numpy()

    logits_batch = akida_model.predict(batch, batch_size=1)

    logits_batch = logits_batch.squeeze(axis=(1, 2))
    labels_all.append(label_batch)
    logits_all.append(logits_batch)

labels_all = np.concatenate(labels_all)
logits_all = np.concatenate(logits_all)
preds = np.argmax(logits_all, axis=1)

akida_acc = float(np.mean(preds == np.array(labels_all)))
print(f'Akida accuracy: {akida_acc:.4f}')

# %% [markdown]
# ### Activation Sparsity
#
# Akida hardware skips computation for zero-valued activations, so activation
# sparsity directly reduces both energy consumption and inference latency.
# Below we measure per-layer sparsity on a 1024-sample calibration batch drawn
# from the training set.

# %%
from akida_models.sparsity import compute_sparsity
from brainchip_utils.plot_utils import pretty_print_sparsity
from plant_village_data import get_samples

NUM_SAMPLES = 100

samples = get_samples(DATA_PATH, input_shape=INPUT_SHAPE, num_samples=NUM_SAMPLES)
sparsity_dict = compute_sparsity(akida_model, samples=samples)
pretty_print_sparsity(sparsity_dict)

# %% [markdown]
# ## Hardware Benchmark
#
# **These cells require a physical AKD1500 device to be connected.** If `device is
# None` (reported in the evaluation section above), skip ahead to the Summary.
#
# Akida is an event-driven architecture: computations scale with the number of
# non-zero activations, not with tensor size. That means benchmark results are
# *input-dependent* — random or synthetic data would give artificially fast or
# slow timings. The `samples` array loaded above (real images from the validation
# split) is therefore the correct input to use here.

# %% [markdown]
# ### Simple Benchmark
#
# The simplest way to time an Akida model is to call `forward` in a loop and
# read back two clocks after each inference:
#
# - **System clock** (`time.perf_counter_ns`) — wall time including Python and
#   USB/PCIe transfer overhead.
# - **On-chip clock** (`akida_model.metrics['inference_clk']`) — raw clock cycles
#   counted by the AKD1500 itself. Dividing by the 400 MHz core frequency gives
#   the pure compute time.
#
# The two numbers should agree closely; a large divergence would indicate a
# transfer or driver bottleneck.

# %%
import time

CLOCK_FREQUENCY = 400e6  # 400 MHz for AKD1500

if device is not None:
    akida_model.map(device, mode=akida.MapMode.Minimal, hw_only=True)

    inf_clks = []
    inf_times = []
    for i in range(len(samples)):
        start_t = time.perf_counter_ns()
        akida_model.forward(samples[i:i+1])
        inf_times.append(time.perf_counter_ns() - start_t)
        inf_clks.append(akida_model.metrics['inference_clk'])

    mean_inf_clk = np.mean(inf_clks) / CLOCK_FREQUENCY * 1e3  # cycles → ms
    mean_inf_time = np.mean(inf_times) * 1e-6                  # ns → ms
    print(f'Mean inference time (system clock):      {mean_inf_time:.3f} ms')
    print(f'Mean on-chip time (chip clock cycles):   {mean_inf_clk:.3f} ms')

# %% [markdown]
# ### Full Model Benchmark
#
# The loop above is clear, but it misses two things: power consumption and a
# comparison between mapping modes. `full_model_benchmark` from
# [brainchip_utils/hardware_utils.py](../../../brainchip_utils/hardware_utils.py)
# runs the same timed loop while also coordinating optional INA219 power
# measurement in a separate process. It sweeps both `MapMode.Minimal` (fewest
# NPs, lowest power) and `MapMode.AllNps` (all NPs, maximum parallelism) so the
# trade-off is visible. The multiprocessing and power-meter wiring are
# non-trivial and not of interest to most users — consult the source if needed.

# %%
from brainchip_utils.hardware_utils import full_model_benchmark, get_mapping_stats
from brainchip_utils.plot_utils import plot_full_model_results

if device is not None:
    map_modes = ['Minimal', 'AllNps']
    POWER_REPEATS = 10
    full_results = {}
    for mm in map_modes:
        map_mode = getattr(akida.MapMode, mm)
        print(f'Running full-model benchmark (MapMode={mm}, {POWER_REPEATS} repeat(s))...')
        full_results[mm] = full_model_benchmark(
            akida_model, device, samples, map_mode=map_mode, repeats=POWER_REPEATS)

        akida_model.map(device, mode=map_mode)
        num_nps, num_passes, num_sequences = get_mapping_stats(akida_model)
        full_results[mm]['num_nps'] = num_nps
        full_results[mm]['num_passes'] = num_passes
        print(f'  Mapping: {num_nps} NP(s), {num_passes} pass(es), {num_sequences} sequence(s)')
        if num_sequences > 1:
            print('WARNING: model not completely mapped to hardware')

# %% [markdown]
# The plot below shows one column per map mode: a power trace (if a power meter
# was connected) and the hardware mapping layout.

# %%
if device is not None:
    plot_full_model_results(full_results, akida_model, device,
                            model_name='akidanet_plant_village',
                            savepath='benchmark_results_full.png')

# %% [markdown]
# ### Per-Layer Benchmark
#
# Full-model timing tells us the total cost but not where time is spent.
# `per_layer_benchmark` from
# [brainchip_utils/hardware_utils.py](../../../brainchip_utils/hardware_utils.py)
# reconstructs latency layer by layer by running cumulative sub-models and
# differencing the results.
#
# Because Akida processes events (non-zero activations), a layer's cost is
# proportional to its *input* sparsity: a layer receiving 90% sparse inputs has
# far fewer events to process than one receiving 10% sparse inputs. The per-layer
# timing and the sparsity values computed above are therefore naturally correlated —
# low-sparsity layers are typically the latency bottlenecks.

# %%
from brainchip_utils.hardware_utils import per_layer_benchmark
from brainchip_utils.plot_utils import plot_per_layer_results

if device is not None:
    # Map without hw_only so akida_model.sequences is populated for the plot
    akida_model.map(device, mode=akida.MapMode.Minimal)

    print(f'Running per-layer benchmark ({len(samples)} samples)...')
    per_layer_results = per_layer_benchmark(akida_model, device, samples)

# %% [markdown]
# The plot stacks three panels: per-layer latency, input sparsity per layer, and
# the hardware mapping. The inverse relationship between sparsity and latency is
# the direct signature of the event-driven compute model: dense activations
# generate more events, and more events mean more work for the hardware.

# %%
if device is not None:
    plot_per_layer_results(per_layer_results, akida_model, sparsity_dict,
                           model_name='akidanet_plant_village',
                           savepath='benchmark_results_layers.png')

# %% [markdown]
# ## Summary
#
# The table below compares validation accuracy across the three model variants.
# The goal is that QAT and Akida accuracy remain close to the float baseline.

# %%
print('PlantVillage results')
print('=' * 40)
print(f'  Float accuracy:     {float_acc * 100:.2f}%')
print(f'  QAT accuracy:       {qat_acc * 100:.2f}%')
print(f'  Akida accuracy:     {akida_acc * 100:.2f}%')
