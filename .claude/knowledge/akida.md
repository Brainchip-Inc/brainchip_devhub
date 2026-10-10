# Akida hardware knowledge base

How Akida hardware and its toolchain actually behave: facts that matter when building,
mapping and explaining models, and that are easy to get wrong. It is loaded into every
Claude session through `CLAUDE.md`, and written for developers too, so entries can move
into tutorials as they are.

**Rules for entries**
- Tag the platform: **[Akida 1]**, **[Akida 2]** or **[both]**.
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

**[Akida 1] A layer's time is a straight line in the events it receives.**
- With no events a layer costs almost nothing. With any events it pays a small fixed cost
  (the fit gives 25,577 clocks for a 3 × 3, 32-filter, 32 × 32 layer on one NP), then a
  constant cost per event.
- To a first approximation, clocks ≈ events × filters / NPs × *c*. Dividing by the NPs
  assumes a split layer shares out its events (a spatial split) or its filters evenly.
- *c*, the clocks per event per filter, depends on the kernel. On AKD1500 with 4-bit
  weights: 0.386 (1 × 1), 0.648 (3 × 3), 2.240 (5 × 5), 9.269 (7 × 7), and 0.646 for a
  3 × 3 separable layer. So 3 × 3 gets the most dense-equivalent MACs per clock.
- Events are counted at the layer's *input*, so a strided layer costs per input event like
  any other.
- AKD1000 gave the same values in clocks (0.375, 0.635, 2.185, 9.25; separable 0.625), so
  the per-event cost is the same core at a different clock. On AKD1000, 2-bit weights made
  5 × 5 and 7 × 7 convolutions much cheaper (0.98, 1.6), but not separable layers
  **(unconfirmed** on AKD1500: rerun the tutorial's kernel sweep with `weights_bits=2`).
- On VWW (AkidaNet 0.25, 96 × 96), `separable_4` measured 0.648, the same as the hand-built
  layer. Fitted per layer, most layers come out at 0.65–0.70 and a few up to 0.87, so the
  model runs a little low; `conv_2`, split over two NPs, is furthest off (1.08).

*Source: measured, AKD1500, 2026-10-10, `akida1/tutorials/sparsity_in_hardware`;
AKD1000 values: BrainChip engineering, 2026-10-10.*

**[Akida 1] A fused separable layer costs the same per event as a full convolution with
the same number of filters.** Measured on AKD1500 at 3 × 3: 0.648 clocks per event per
filter for a convolution, 0.646 for a separable layer. The depthwise-separable saving you'd
expect on a GPU doesn't happen on Akida 1 (see [Backbones and blocks](#backbones-and-blocks)).
*Source: measured, AKD1500, 2026-10-10, `akida1/tutorials/sparsity_in_hardware`.*

**[Akida 1] Each NP has 32 MAC units, in 8 blocks of 4, and processes each event for 8
filters at once.** Against a dense 32-MAC array computing the same 3 × 3 layer, an NP is
slower on fully dense input (42 % of the array's ideal speed) and faster below a break-even
input density of 0.40. AKD1000 and AKD1500 give the same figures.
*Source: BrainChip engineering, 2026-10-10 (NP structure); measured, AKD1500, 2026-10-10,
`akida1/tutorials/sparsity_in_hardware`.*

**[Akida 1] Energy follows the events too.** A busy NP draws roughly constant dynamic
power, so a layer's dynamic energy is proportional to its clock count, and static energy
scales with latency anyway. Forcing every activation of the VWW model non-zero (same
weights, same mapping) doubled both: 8.03 → 16.74 ms and 0.199 → 0.407 mJ dynamic energy
per inference, in MapMode.Minimal. Shifting thresholds in between traced a smooth curve.
*Source: measured, AKD1500, 2026-10-10, `akida1/tutorials/sparsity_in_hardware`.*

**[Akida 1] Zero weights save energy, not time.** With fully dense input, a 3 × 3 layer took
the same clocks with random weights (81 % non-zero), half of them zeroed (41 %) or an
identity kernel (0.3 %), while the dynamic energy of the whole test model fell from 74.6 to
62.8 to 34.9 µJ per inference. Only activation sparsity saves time. Measured on one
hand-built layer only.
*Source: measured, AKD1500, 2026-10-10, `akida1/tutorials/sparsity_in_hardware`.*

**[Akida 1] With batch size > 1, layers work on successive frames in parallel
(pipelining),** so time per frame tends towards that of the slowest stage rather than the
sum of the layers. On AKD1000 with AkidaNet 0.25 at batch 100 the bottleneck didn't always
fall where single-layer timings predicted. **(unconfirmed:** per-layer subtraction at
batch 1 vs batch 100 on AKD1500 would show it.)
*Source: BrainChip engineering (measured on AKD1000), 2026-10-10.*

## Measuring on hardware

**[both] `model.metrics['inference_clk']` is the chip's clock count for the last
inference.** It leaves out the host (Python, driver, PCIe), which varies by host: on a
Raspberry Pi with AKD1000 the host side could take several times the on-chip time.
It is very repeatable: rerunning the same image through VWW sub-models of 1–3 M clocks
gave differences of a few hundred clocks (median), rarely up to about 1 %. So
image-to-image variation in latency is real, and comes from the activity.
*Source: measured, AKD1500, 2026-10-10; BrainChip engineering (host side), 2026-10-10.*

**[both] There is no per-layer counter: time a layer by subtraction.** Run the model up to
and including the layer, then up to the layer before, and subtract
(`per_layer_benchmark` in `brainchip_utils`). Silence the last layer of each sub-model
(`silence_output_layer`) so it sends no events off the chip: copying outputs back to the
host is slow, especially on AKD1000, and would be charged to whichever layer is last.
*Source: BrainChip engineering, 2026-10-10; `brainchip_utils/hardware_utils.py`.*

**[Akida 1] The software backend reproduces the hardware's outputs exactly,** so event
counts per layer can be computed without a device: on VWW every layer's output matched
on 20 images. **(unconfirmed** exception: a hand-built model with identity kernels,
mapped with `AllNps` over 32 NPs, gave 72 differing values out of 131k; with `Minimal` it
matched. Worth reporting if it reproduces.)
*Source: measured, AKD1500, akida 2.19.3, 2026-10-10.*

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
