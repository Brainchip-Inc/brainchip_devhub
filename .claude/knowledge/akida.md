# Akida hardware knowledge base

How Akida hardware and its toolchain actually behave: facts that matter when building,
mapping and explaining models, and that are easy to get wrong. It is loaded into every
Claude session through `CLAUDE.md`, and written for developers too, so entries can move
into tutorials as they are.

**Rules for entries**
- Tag the platform: **[Akida 1]**, **[Akida 2]**, **[Akida Pico]** or **[both]** (Akida 1 and 2).
- Give a source: a link to the official docs, *BrainChip engineering* (with date) for
  knowledge passed on directly, *measured* (with the example) for results from this
  repo, or a file in this repo.
- Mark anything not yet confirmed as **(unconfirmed)**, and say what would confirm it.
- When Claude is corrected about Akida, the correction is added here in the same PR as
  the fix. General Akida behaviour only; example-specific findings stay in the example.

---

## Input layer (dedicated input processor)

**[both] The first layer of a model usually runs on a dedicated input processor, not on
a standard NP.** It appears as
[`InputConvolutional`](https://doc.brainchipinc.com/api_reference/akida_apis.html#akida.InputConvolutional)
in Akida 1 and
[`InputConv2D`](https://doc.brainchipinc.com/api_reference/akida_apis.html#akida.InputConv2D)
in Akida 2. Internally it is called the HRC ("high-resolution convolution"; the exact
expansion is unconfirmed). In mapping statistics it counts as a single NP, but it is a
separate unit, and no mapping mode can spread it over more hardware.
*Source: BrainChip engineering, 2026-10-09.*

**[both] It was designed for dense image input.**
- It takes 1-channel (greyscale) or 3-channel (RGB) input.
- It accepts uint8 inputs. It is the only layer that does on Akida 1: see
  [Input ranges](#input-ranges).
- It is a conventional convolution and **does not exploit sparsity**. Its cost depends
  on kernel size, input size and number of filters, not on how many inputs are zero.
- In most image models it does a small share of the work: its input has few channels
  and it usually has few filters, even though its spatial size is the largest in the
  network. A large kernel over a large input can change that. In
  `akida1/model_zoo/uored_vafcls` a 7 × 7 stem over a 300 × 140 input is the slowest
  layer (measured, AKD1500, 2026-10-08).

*Source: BrainChip engineering, 2026-10-09; measured as noted.*

**[both] It sets a hard limit on the second input dimension: 256 on Akida 1, 384 on
Akida 2.** Standard NPs have a smaller per-NP limit on the second dimension (32 on
Akida 1, 64 on Akida 2), but a standard layer can be split spatially across several NPs,
up to the number available, so in practice the input layer's limit is the one that
constrains input shapes. That is why `uored_vafcls` frames its input as 300 × 140 rather
than 140 × 300, and why the Akida 2 segmentation spec (#49) feeds a 384 × 768 image to
Akida transposed, as 768 × 384.
*Source: BrainChip engineering, 2026-10-09.*

## Input ranges

**[Akida 1] Standard NPs take 4-bit unsigned inputs (0–15).** At the Python level the
values are passed as uint8 (the type isn't enforced there), but values above 15 make the
hardware raise errors. **(unconfirmed:** whether every path raises an error, or some
saturate silently. A deliberate out-of-range test on AKD1500 would settle it.)
*Source: BrainChip engineering, 2026-10-09.*

**[Akida 2] Standard NPs take int8 inputs**, signed so that layers can receive the output
of non-ReLU activations.
*Source: BrainChip engineering, 2026-10-09.*

## Sparsity

**[both] Standard NPs process events, i.e. non-zero activations.** A layer's cost scales
with how many of its inputs are non-zero, so activation sparsity saves both time and
energy. Benchmarks must therefore use real inputs: random or synthetic data drives
unrepresentative activity and gives misleading timings. The input layer is the
exception (see above).
*Source: akida1/README.md, "The technology"; benchmark notebooks in this repo.*

## Mapping modes

**[both] `akida.MapMode` has three strategies.**
- `Minimal`: the fewest NPs that hold the model.
- `AllNps`: as many NPs as possible while keeping the minimum number of passes.
- `HwPr` (hardware partial reconfiguration): splits the model into several passes so
  each layer can use more NPs. The weights are reloaded between passes on every
  inference.

*Source: `.claude/commands/model-zoo-eval-example.md`; akida1/README.md.*

**[Akida 1] HwPr needs AKD1500.** AKD1000 does not support it.
*Source: akida1/README.md, chips table.*

**[Akida 2] Both Akida 2 platforms support HwPr:** AKD2500 and the Akida 2 FPGA.
*Source: BrainChip engineering, 2026-10-09.*

**[Akida 1] Which mode is fastest depends on the model.**
- `AllNps` traded power for time in the examples measured here: more power, less time,
  about the same dynamic energy per inference. *Source: measured: `uored_vafcls`
  (2026-10-08) and `arrhythmia_classification` benchmark notebooks.*
- `HwPr` gave a strong advantage for most model zoo models on AKD1500. Only a few of
  the smallest showed no gain, or rarely a loss. *Source: BrainChip internal
  benchmarking of the Akida 1 model zoo, 2026.*
- For a small model that already fits in a single pass, reloading weights between
  passes can cost more than the per-layer gain: in `uored_vafcls`, `HwPr` was slower
  than `AllNps` and used more energy. *Source: measured, AKD1500, 2026-10-08.*
- Across this repo's Akida 1 examples, `HwPr` was fastest and used the least energy for
  the AkidaNet image models (`vww`, `plant_village`, and every `imagenet_akidanet`
  variant). For the small models it either lost outright (`speech_commands`: about 4.5×
  slower than `AllNps`) or was slightly faster but used more energy
  (`arrhythmia_classification`). So benchmark all three modes; don't assume.
  *Source: measured, AKD1500, 2026-10-09 (#58); numbers in each example's `metrics.json`.*

## Platforms and clocks

| | NPs | HwPr | Clock | Notes |
| --- | --- | --- | --- | --- |
| AKD1000 [Akida 1] | 80 | no | 300 MHz | |
| AKD1500 [Akida 1] | 32 | yes | 400 MHz | reference chip for Akida 1 numbers in this repo |
| AKD2500 [Akida 2] | | yes | 1 GHz target | pre-production; the clock may change |
| Akida 2 FPGA [Akida 2] | 24 (6 nodes) | yes | 25 MHz | current Akida 2 benchmark platform; no power measurement |

**[both] A node is 4 NPs.** Akida 2 hardware is often described in nodes, so the
6-node FPGA has 24 NPs.

*Source: akida1/README.md; `brainchip_utils/hardware_utils.py` (`AKIDA_CLOCKS_HZ`);
BrainChip engineering, 2026-10-09 (Akida 2 HwPr, nodes and NP counts).*

## Graph constraints

**[both] An Akida graph cannot contain a reshape.** Reshape the input in the data
pipeline instead, so the deployed model receives an already-shaped tensor.
*Source: `akida1/model_zoo/uored_vafcls` data and model docstrings.*

**[Akida 1] Global average pooling needs at least 3 columns at its input.** That limits
how much pooling can come before it.
*Source: `akida1/model_zoo/uored_vafcls/uored_vafcls_model.py` docstring.*

## Backbones and blocks

**[both] Separable convolutions: fused on Akida 1, two layers on Akida 2.**
- On Akida 1 a depthwise-separable convolution is a single fused layer
  (`SeparableConv2D`). It can't have a ReLU between its depthwise and pointwise parts.
- On Akida 2 the depthwise convolution is a distinct layer (`DepthwiseConv2D` followed by
  a pointwise `Conv2D`), so a block can have a ReLU after the depthwise layer as well, as
  in MobileNet.
- Without that ReLU the depthwise outputs are dense, and the pointwise layer that takes
  them processes more events (see [Sparsity](#sparsity)). Adding it should make Akida 2
  models faster, and it's one reason MobileNet runs faster than AkidaNet there. The risk
  is accuracy, possibly because quantization gets harder.
- `akida_models` factories don't add it. Under `set_akida_version(AkidaVersion.v2)`,
  `separable_conv_block` (with `fused=False`) and models built from it, such as
  `ds_cnn_kws` and `akidanet_imagenet`, still go depthwise → pointwise → BN → ReLU, with
  nothing in between. Adding the ReLU means defining the block locally.

- **Measured** on Akida 2 Speech Commands (DS-CNN, #101 / PR #107):
  - With the ReLU, the pointwise layers' input sparsity rose from 11–30 % to 52–60 %
    (8-bit).
  - Projected AKD2500 latency fell 8–11 % (8-bit) and 19–23 % (4-bit) in every mapping
    mode, with the same NP counts.
  - 8-bit accuracy was unchanged. 4-bit QAT lost 0.23 points, after raising the QAT
    learning rate from 1e-4 to 1e-3 (it lost 0.65 at 1e-4). That's within run-to-run
    noise.

*Source: BrainChip engineering (repo owner), 2026-10-10; layer structure checked with
akida_models 1.14.0; measured on the Akida 2 FPGA, 2026-10-10
(`akida2/model_zoo/speech_commands/docs/metrics.json`).*

**[Akida 2] Prefer an ImageNet-pretrained MobileNet (V1) backbone to AkidaNet.**
AkidaNet (`akida_models.akidanet_imagenet`) was designed around Akida 1. On Akida 2, an
off-the-shelf `tf_keras.applications.mobilenet.MobileNet` with ImageNet weights is
expected to do better, and `akida2/model_zoo/vww` uses one (#26). Build it as
`vww_model.py` does: `include_top=False, pooling='avg'`, then
`quantizeml.models.transforms.insert_rescaling(scale=1/127.5, offset=-1)`, so the model
still takes uint8 inputs. So an Akida 2 example
ported with an AkidaNet backbone is a candidate for a follow-up improvement that swaps in
MobileNet at the same alpha and resolution. Port first as the source has it, then raise
the swap as its own issue under the Model improvement epic (#39), as #99 does for
PlantVillage.

**Measured** on PlantVillage, alpha 0.5, 224 × 224 (#99):

| | AkidaNet | MobileNet |
| --- | --- | --- |
| HwPr projected latency (AKD2500) | 6.870 ms (8-bit), 8.324 ms (4-bit) | 5.784 ms (8-bit), 4.802 ms (4-bit) |
| Minimal projected latency (AKD2500) | 11.483 ms (8-bit), 12.254 ms (4-bit) | 7.872 ms (8-bit), 8.621 ms (4-bit) |
| Accuracy: float / 8-bit / 4-bit QAT | 99.67 % / 99.63 % / 99.65 % | 99.61 % / 99.43 % / 98.97 % |

- MobileNet is 16–42 % faster depending on mapping mode and bit width.
- It's also smaller: no 512-unit dense layer, which made no accuracy difference here.
- It lost accuracy in quantization: 0.2 points at 8 bits and 0.7 at 4 bits. A higher QAT
  learning rate made 4-bit worse here, unlike on Speech Commands.
- So expect a speed gain, but check quantized accuracy.

*Source: BrainChip engineering (repo owner), 2026-10-10; `akida2/model_zoo/vww` (#26);
measured on the Akida 2 FPGA, 2026-10-10 (`akida2/model_zoo/plant_village/docs/metrics.json`
after #99, AkidaNet values from before it).*

## Toolchain

**[both] The default `cnn2snn` context is Akida 2.** Akida 1 model construction and
conversion must be wrapped in `set_akida_version(AkidaVersion.v1)`, or the wrong
architecture is built silently.
*Source: `.claude/commands/model-zoo-eval-example.md` (verified against akida_models 1.14.0).*

**[both] The quantization route differs by platform.**
- Akida 1 uses `cnn2snn.quantize`: 4-bit weights and activations, with an 8-bit first
  layer, usually followed by quantization-aware training.
- Akida 2 uses `quantizeml`, defaulting to 8 bits, where post-training quantization
  alone should come close to float accuracy, without QAT. *Source: BrainChip
  engineering, 2026.*

The model-building defaults also differ: fused versus split separable convolutions,
global average pooling before versus after ReLU, ReLU6 versus ReLU3.75. The verified
table is in `.claude/commands/model-zoo-eval-example.md`.
*Source: that table; akida1/README.md.*

**[Akida 2] PyTorch models reach Akida through ONNX and `onnx2akida`.**
`onnx2akida.convert` returns a hybrid model (Akida parts plus CPU parts) and a
compatibility report. *Source: measured with onnx2akida 0.7.0 (2026-10-09), PR #82.*

**[Akida 1] The ONNX route is, in practice, Akida 2 only.** In that pipeline the
quantized model is held in ONNX form, and there is currently no way to run
quantization-aware training on it. Without QAT, post-training quantization is the only
option, which in practice means 8 bits, and that is Akida 2. Akida 1's 4-bit weights and
activations need QAT to recover accuracy. So a PyTorch model targeting Akida 1 would
have to be rebuilt in Keras and go through `cnn2snn` with QAT.
*Source: BrainChip engineering, 2026-10-09.*

Version constraints between torch, TensorFlow and `onnx2akida` are in the README, under
Requirements → "PyTorch and TensorFlow in one environment".

## Akida Pico

**[Akida Pico] Akida Pico models are recurrent TENNs.** They are trained with
`Kernelized` layers (temporal convolutions whose kernel is generated from state-space
parameters), then `convert_to_stateful` turns each one into a `StatefulRecurrent` layer
with the same weights, which processes the input in chunks of `timesteps` samples and
carries its state between chunks. The stateful model is the one quantized with
`quantizeml` and converted with `cnn2snn`. Stateful tf_keras models have a fixed batch
size (set with `quantizeml.layers.update_batch_size`), and their quantized form must be
called directly rather than through `predict`.
*Source: [official model zoo](https://doc.brainchipinc.com/model_zoo_performance.html)
("Pico models are recurrent TENNs targeting the Akida Pico Neuromorphic Processor
IP"); `akida_models.tenn_recurrent` docstrings (akida_models 1.14.0); measured,
`akida_pico/model_zoo/speech_commands` (#103).*

**[Akida Pico] Hardware limits for `StatefulRecurrent` models** (checked at mapping): at
most 8 `StatefulRecurrent` layers; only a `Dequantizer` or `PicoPostProcessing` may follow
them, as the last layer; ReLU must be unbounded; first layer input 8-bit with up to 256
channels or 16-bit with up to 128 (the packed input row must be a power-of-2 number of
32-bit words, at most 512 bytes; MetaTF doesn't pad); stateful channels at most 256, a
power of 2, the same in every layer; subsampling at most 4 per layer.
*Source: [Akida Pico hardware constraints](https://doc.brainchipinc.com/user_guide/hardware/pico.html), MetaTF 2.19.3.*

**[Akida Pico] Mapping can be checked without hardware.** `akida.PicoIP()` returns a
virtual Pico device; mapping a converted model onto it checks the limits above. Map a
copy (`akida.Model(model.layers)`), since a model mapped on a virtual device can't run
inference. `tenn_recurrent_sc12` maps entirely onto one `TNP_R` component, in a single
hardware sequence.
*Source: `akida.PicoIP` docstring (akida 2.19.3); measured on the virtual device,
`akida_pico/model_zoo/speech_commands` (#103).*

**[Akida Pico] Inputs can be raw 16-bit signals.** `tenn_recurrent_sc12` takes raw
16 kHz audio as int16 (quantized with `-id int16`), with no MFCC step.
**(unconfirmed:** Pico clock, power and latency: not yet measured in this repo; #104.)
*Source: `akida_models.tenn_recurrent_sc12` (akida_models 1.14.0).*
