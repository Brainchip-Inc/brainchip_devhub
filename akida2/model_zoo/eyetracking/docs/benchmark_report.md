# Benchmark Report: Camera-Less Eye Tracking on Akida

Full pipeline: GazeBase (real gaze) -> simulated PSOG photodiode signal ->
Tiny Temporal CNN -> QuantizeML -> Akida conversion, evaluated end to end.

**Environment caveat:** no physical Akida device (AKD1000/AKD1500) was
attached in this environment (`akida.devices()` returns empty). All Akida
numbers below are from the **software backend** (a functional/accuracy
simulator), not measured on real silicon. Power consumption specifically
cannot be measured at all without hardware and is not reported - reporting
an estimate here would overstate confidence.

## Accuracy through the pipeline

### Tiny Temporal CNN

Two model designs were tried; only the second is deployable on Akida
(details in `README.md` and commit history):

| Stage | Design | Mean angular error |
|---|---|---|
| Original spec (Conv1D, GlobalAveragePooling1D) | float | 0.214 deg |
| Conv1D reshaped as Conv2D (dead end - see below) | float | 0.211 deg |
| Conv1D reshaped as Conv2D | quantized int8 | 0.266 deg |
| Conv1D reshaped as Conv2D | **Akida conversion: rejected** | n/a |
| Causal streaming (Conv3D -> BufferTempConv) | float, last-timestep | 0.224 deg |
| Causal streaming | quantized int8, last-timestep | 0.247 deg |
| Causal streaming | **Akida-converted, last-timestep** (n=300 windows) | **0.253 deg** |

The accuracy degradation from float (0.224 deg) through quantization
(0.247 deg) to actual Akida conversion (0.253 deg) is small and monotonic -
a credible, real result, not a rounding artifact.

### Tiny MLP

No architecture rework needed - a plain Flatten+Dense feedforward network
hits none of the CNN's Conv2D/BufferTempConv complications:

| Stage | Mean angular error |
|---|---|
| Float (tf_keras rebuild) | 0.238 deg |
| Quantized int8 | 0.352 deg |
| **Akida-converted** (full 301,808-window test set) | **0.352 deg** |

Two things stand out relative to the CNN. First, quantization hurts the MLP
more (0.238 -> 0.352 deg, +48%) than the CNN (0.224 -> 0.247 deg, +10%) -
plausibly because the MLP's first Dense layer takes a 512-wide flattened
input directly (no conv/pooling to average out per-channel quantization
noise first). Second, Akida conversion added **zero** further error
(0.352 -> 0.352 deg) versus the CNN's small extra drop (0.247 -> 0.253 deg) -
consistent with the MLP having no buffered/FIFO state for conversion to
perturb, just a static weight-precision mapping. The full test set could
also be evaluated in one batched call (unlike the CNN's forced
300-window sequential sample - see toolchain point 3 below), since a
feedforward network has no per-stream state for batching to corrupt.

Recall from `docs/dataset_report.md` and the earlier shift-robustness work:
these are all under the **idealized zero-shift** simulation. Under the
sensor-shift stress test (see README's Results table), error rises to
~1.9-2.0 deg, which is the number actually comparable to published
real-hardware papers - and a physical sensor would add further noise this
simulation doesn't model. Read every number here as "does the deployment
pipeline work," not "this is field-ready accuracy."

## Why two model designs

The original spec (`models/tiny_temporal_cnn.py`: Conv1D + GAP1D) cannot be
quantized/converted at all - quantizeml only supports Conv2D-family,
Dense, and BufferTempConv (1D temporal, FIFO-buffered) layers. Two attempts:

1. **Conv1D reshaped as Conv2D** (`models/tiny_temporal_cnn_akida.py`): treat
   the (64, 8) window as a (64, 1, 8) "tall image", replacing Conv1D(k) with
   Conv2D(k, 1). Trains and quantizes fine, but `cnn2snn.convert()` rejects
   it: **"Conv2D handle only square kernels"** - Akida's 2D conv hardware
   can't do a (5,1)-shaped kernel. Real hardware constraint, not a bug.
2. **Causal streaming** (`models/tiny_temporal_cnn_streaming.py`): the
   actual supported primitive, matching BrainChip's own eye-tracking
   reference (`akida_models/tenn_spatiotemporal/eye_train.py`). Requires a
   5D (batch, T, 1, 1, C) input, causal-padded Conv3D layers that
   `quantizeml`'s `sanitize()` auto-bufferizes into `BufferTempConv` during
   `quantize()`, and forbids the original design's whole-window
   GlobalAveragePooling. The model instead predicts gaze **continuously at
   every timestep from past-only context** - a genuine streaming design,
   arguably a better match for "ultra-low-power continuous inference" than
   the original one-prediction-per-window spec. Early timesteps in a window
   have little receptive field ("warm-up"): error starts at 0.48 deg and
   drops to a flat 0.224 deg by timestep 6-7 (see
   `docs/figures/streaming_warmup.png`), exactly matching the two causal
   convs' combined receptive field (5+3-1=7 taps).

## Real toolchain bugs and constraints found along the way

1. `quantizeml.models.transforms.replace_conv3d` is decorated with
   `@safe_fail`, which silently swallows ALL exceptions and returns the
   unmodified model on any failure - surfacing only as a vague downstream
   warning ("layer not supported to quantize"), not the real error. Calling
   the undecorated function directly (`replace_conv3d.__wrapped__`) revealed
   the actual cause: `_verify_reshape` assumes every `Reshape` layer's saved
   config has a `batch_input_shape` key, true only for a model's very first
   layer. A trailing `Reshape` (squeezing trivial (1,1) spatial dims) broke
   this. Fixed by dropping the Reshape and squeezing in numpy instead.
2. **`cnn2snn.convert()` requires signed (int8) input, not unsigned (uint8),
   for ANY model** - first found on the CNN (assumed specific to
   `BufferTempConv`), then hit again on the plain-Dense MLP
   ("Only signed inputs are supported" from `QuantizedDense`/`QuantizedReLU`
   too). Fixed by recentering the [0, 255] ADC-style reading to int8's
   [-128, 127] range in data prep, with the model's `Rescaling` layer's
   `offset` undoing the shift for the float math.
3. Akida's buffered/FIFO layers (`BufferTempConv`) only correctly maintain
   independent state for **batch_size=1**. Feeding multiple independent
   windows as one batch (sharing a single FIFO indexed by batch position)
   silently produces wrong results - no error, just elevated error that
   never settles, because each batch slot's "history" isn't what it should
   be for its own eyewear session. One continuous stream at a time is also
   how a real device is actually used, so this isn't just a workaround: it
   changed the evaluation harness to loop windows sequentially (300 windows
   x 64 sequential predict() calls each, with a full model reset between
   windows), not a bug to route around for the real deployment case. This
   does NOT apply to the plain-Dense MLP (no buffered state), which
   evaluated the full 301,808-window test set in one batched call.
4. `quantizeml`'s `QuantizedFlatten` only accepts `FixedPoint`-typed input,
   but a `Rescaling` layer's quantized output is `QFloat`, and a raw
   (unquantized) `Input` tensor is neither - so a `Flatten` layer placed
   either right after `Rescaling` or right after `Input` both fail
   (`"QuantizedFlatten only accepts [FixedPoint] inputs"`). Since `Flatten`
   has no learnable weights, worked around by reshaping in numpy data prep
   instead of using a Keras `Flatten` layer at all (see
   `models/tiny_mlp_akida.py`).

## Model size and estimated performance (software backend)

| Metric | Tiny Temporal CNN (streaming) | Tiny MLP |
|---|---|---|
| Parameters | 3,490 (target < 50K) | 35,474 (target ~10-20K) |
| Quantization | 8-bit weights/activations, int8 input | 8-bit weights/activations, int8 input |
| Akida model file | `deploy/model_streaming.fbz`, 8.7 KB | `deploy/model_mlp.fbz`, 46 KB |
| Akida layer count | 6 (2x BufferTempConv, 2x Dense1D, input/dequantizer) | 6 (4x Dense1D, input/dequantizer) |
| MACs per inference step | 3,264 | 35,360 |
| Minimal Akida device (`compute_min_device`) | **4 NPs**: 2x TNP_B + 1x FNP3 + 1x FNP2, 2 skip-DMA channels, 192B ext. memory | **4 NPs**: 3x FNP3 + 1x FNP2, no skip DMAs, 48B ext. memory |
| Software-backend throughput | ~52,000 fps (batch=1 sequential - see caveat) | ~94,000 fps (batched - valid, no buffered state) |
| Estimated per-frame latency | ~19 microseconds | ~11 microseconds |
| Power | **not measurable** - no physical device attached | **not measurable** |

Both models fit in just 4 Neural Processors - a small fraction of a full
AKD1000 mesh - computed by mapping each `.fbz` to
`akida.compute_min_device(model)` (the minimal device that fits the model)
and inspecting the resulting per-layer NP allocation. The CNN's 2 extra
skip-DMA channels support its `BufferTempConv` FIFO connections, which the
MLP doesn't need.

**Throughput caveat:** these figures come from the software backend's raw
per-call speed on this machine's CPU, not a hardware cycle count or a
measurement on an actual AKD1000/AKD1500. The MLP's throughput number is
directly comparable across runs (ordinary batched inference); the CNN's is
from sequential batch=1 calls (required by its buffered state) and so
reflects a different, slower calling pattern rather than a real
CNN-vs-MLP hardware speed comparison. Both indicate the models are small
enough to run far faster than needed for real-time gaze tracking, not what
a physical chip's power/latency would actually be.

**CNN vs. MLP trade-off:** the CNN is 10x smaller (3,490 vs 35,474 params,
8.7 KB vs 46 KB) and edges out the MLP on zero-shift accuracy (see
README's Results table: 0.214 vs 0.243 deg), but required a real
architecture redesign (causal streaming) to become Akida-deployable at
all, plus the batch-size-1 evaluation constraint. The MLP deployed with no
architecture changes and evaluates efficiently in batches, at the cost of
10x the parameters/model size and worse post-quantization accuracy
(+48% vs +10% relative error increase from quantization).

## What's still missing for a genuine hardware benchmark

- Access to a physical AKD1000 or AKD1500 device (`akida.devices()` is
  empty here) - needed for real latency and power numbers.
- Evaluation on the full test set rather than a 300-window sample (limited
  here purely by the sequential-per-window evaluation being slow in
  software; not a hardware limitation).
- Real photodiode sensor data - every number in this report, including this
  one, is on the simulated PSOG signal from `data/psog_simulate.py`, not a
  physical sensor (see `docs/dataset_report.md` for why no such public
  dataset exists).
