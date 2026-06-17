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
# Run Time: ~1 hour with training included / ~2 minutes with training skipped
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
# ## Evaluation of Akida Model
#
# We now run evaluation through the akida model, to check that accuracy is 
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
# typically we'd only have a single akida device connected on a given machine, 
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

# Load the akida model
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

labels_all = []
logits_all = []
for batch, label_batch in val_ds:
    if not isinstance(batch, np.ndarray):
        batch = batch.numpy()

    logits_batch = akida_model.predict(batch, batch_size=1)

    logits_batch = logits_batch.squeeze(axis=(1, 2))
    labels_all.append(label_batch)
    logits_all.append(logits_batch)

labels_all = np.concatenate(labels_all)
logits_all = np.concatenate(logits_all)
preds = np.argmax(logits_all, axis=1)

akida_accuracy = float(np.mean(preds == np.array(labels_all)))
print(f'Akida accuracy: {akida_accuracy:.4f}')

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
sparsity_dict = compute_sparsity(akida_model, samples=samples)

col_w = max(len(k) for k in sparsity_dict) + 2
print(f"\n{'Layer':<{col_w}} {'Sparsity':>10}")
print("-" * (col_w + 11))
for layer, sparsity in sparsity_dict.items():
    print(f"{layer:<{col_w}} {sparsity:>9.2%}")
print("-" * (col_w + 11))
print(f"{'Mean':<{col_w}} {np.mean(list(sparsity_dict.values())):>9.2%}")

# %% [markdown]
# ## Hardware Benchmarking
#
# **These cells require a physical AKD1500 device to be connected.** If `device is
# None` (reported in the evaluation section above), skip ahead to the Summary.
#
# Akida is an event-driven architecture: computations scale with the number of
# non-zero activations, not with tensor size. That means benchmark results are
# *input-dependent* — random or synthetic data would give artificially fast or
# slow timings. The `samples` array loaded above (real images from the training
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
                            model_name=akida_model_path,
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
                           model_name=akida_model_path,
                           savepath='benchmark_results_layers.png')

# %% [markdown]
# ## Summary
#

# %%
print(f"Float: {float_accuracy:.4f}")
print(f"QAT:   {qat_accuracy:.4f}")
print(f"Akida: {akida_accuracy:.4f}")
