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

## Platforms and clocks

| | NPs | HwPr | Clock | Notes |
| --- | --- | --- | --- | --- |
| AKD1000 [Akida 1] | 80 | no | 300 MHz | |
| AKD1500 [Akida 1] | 32 | yes | 400 MHz | reference chip for Akida 1 numbers in this repo |
| AKD2500 [Akida 2] | | | 1 GHz target | pre-production; the clock may change |
| Akida 2 FPGA [Akida 2] | 6 nodes | | 25 MHz | current Akida 2 benchmark platform; no power measurement |

*Source: akida1/README.md; `brainchip_utils/hardware_utils.py` (`AKIDA_CLOCKS_HZ`).*

## Graph constraints

**[both] An Akida graph cannot contain a reshape.** Reshape the input in the data
pipeline instead, so the deployed model receives an already-shaped tensor.
*Source: `akida1/model_zoo/uored_vafcls` data and model docstrings.*

**[Akida 1] Global average pooling needs at least 3 columns at its input.** That limits
how much pooling can come before it.
*Source: `akida1/model_zoo/uored_vafcls/uored_vafcls_model.py` docstring.*

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
