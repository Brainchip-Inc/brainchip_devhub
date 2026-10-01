<img src="../../../docs/assets/0.-BC-dev-hub-LOGO-flicker.svg" alt="BrainChip Dev Hub" width="200"/>

# Cityscapes Semantic Segmentation — Akida 2

## Model Card

Float mIoU: **60.53%** (mosaic tiling) / **62.69%** (Hann-window
tiling) &nbsp;|&nbsp; Parameters: **6,294,164**

This is a **semantic segmentation** task (per-pixel 19-class prediction), so
the table below reports mean Intersection-over-Union (mIoU), not accuracy.
Unlike every other example in this zoo, the model is trained in **PyTorch**,
not tf_keras: quantization uses quantizeml's direct-from-ONNX path
(`quantizeml.models.quantize(onnx_model, ...)`), so no Keras model exists
anywhere in this pipeline — see "Why PyTorch?" below. 8-bit weights and
activations were enough to match the float model's accuracy, so no
lower-precision/QAT variant was pursued.

<table>
  <thead>
    <tr>
      <th>Stage</th>
      <th>Tiling</th>
      <th>mIoU</th>
      <th>Sparsity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2">Float (PyTorch)</td>
      <td>Mosaic</td>
      <td align="center">60.53%</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td>Hann window</td>
      <td align="center">62.69%</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td colspan="2">ONNX (pre-quantization)</td>
      <td align="center">TBD</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td colspan="2">8-bit Akida-converted (mosaic only — see note below)</td>
      <td align="center">60.63%</td>
      <td align="center">TBD</td>
    </tr>
  </tbody>
</table>

The Akida row is evaluated on **500 val images** (the full
Cityscapes val split), mosaic tiling only — the Akida software backend used
in this environment is CPU-bound, and Hann-window tiling's ~4x tile count
per image makes a full-split pass impractical here (mosaic-only already
takes over an hour; see `docs/original_readme.txt` and
`project_notebook_training.ipynb` for the as-run timings).

**The Akida-converted model matches the float model's accuracy**
(60.63% vs. 60.53% mosaic) — this is a fix of
an earlier (pre-"Dec 15" checkpoint) result where Akida conversion caused a
large accuracy drop; see "Project history" below for what changed.

**Akida 2 hardware benchmark (FPGA @ 25 MHz)**

> **Note:** no physical Akida 2 FPGA platform is attached in the environment
> this example was built in, so the fields below are unmeasured (`TBD`).
> Run `segmentation_benchmark.py --save-metrics` against real hardware to
> fill them in — see "Contributing and Maintenance" below.

<table>
  <thead>
    <tr>
      <th>Mapping</th>
      <th>NPs</th>
      <th>Passes</th>
      <th>Cycles</th>
      <th>Latency @ 25 MHz (ms)</th>
      <th>Projected @ 1000 MHz (ms) <i>(target)</i></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Minimal</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
    </tr>
    <tr>
      <td>AllNPs</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
    </tr>
    <tr>
      <td>HwPr</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
      <td align="center">TBD</td>
    </tr>
  </tbody>
</table>

> **Note:** the projected clock is BrainChip's current **target** for AKD2500
> production silicon — that silicon does not exist yet, so treat this as a
> target rather than a measured value. Power measurement on the FPGA platform
> is still under development, so only latency is reported.

### Why PyTorch?

The model is a `timm` MobileNetV4-conv-small encoder (ImageNet-pretrained)
with a lightweight depthwise-separable UNet-style decoder
(`segmentation_model.py`), two off-the-shelf components with no native
tf_keras equivalent used elsewhere in this repo. Rather than re-deriving an
equivalent Keras architecture (which could not be verified to match the
trained weights numerically), this example keeps the model and training
loop in PyTorch and relies on quantizeml's **ONNX** quantization path
instead of its Keras path:

```
PyTorch model --[torch.onnx.export]--> ONNX --[quantizeml.models.quantize]-->
quantized ONNX --[cnn2snn.convert]--> Akida
```

`segmentation_model.py`/`segmentation_train.py` therefore contain PyTorch
code where every other example's `<name>_model.py`/`<name>_train.py`
contain tf_keras code — the file layout, docs, and `pretrained_models/`
convention otherwise match the rest of this zoo.

Two of MobileNetV4's layers also needed patching for Akida compatibility
(both in `segmentation_model.py`):
- **Asymmetric padding**: several stride-2 convs use TensorFlow-style
  "SAME" padding, asymmetric for even input sizes — not supported by
  Akida's conv hardware (symmetric padding only). `AsymmetricPaddedConv`
  does the asymmetric pad as an explicit op instead.
- **Unsupported 5x5 stride-2 depthwise conv**: not supported by Akida's
  depthwise conv hardware at all. `SplittedDWConv` replaces it with a
  supported stride-1 5x5 depthwise conv followed by a fixed
  identity-weighted stride-2 3x3 depthwise conv — same receptive field and
  output shape.

## Requirements

For environment requirements and setup, see the [Requirements](../../../README.md#requirements)
section of the top-level README. This example additionally needs `torch`,
`timm`, `albumentations`, `torchmetrics`, `opencv-python`, `onnx`, and
`onnxruntime` (PyTorch training/export are outside this repo's usual
tf_keras dependency set — see "Why PyTorch?" above).

## Dataset

[Cityscapes](https://www.cityscapes-dataset.com/) is a semantic urban scene
understanding benchmark: 5,000 finely-annotated 2048x1024 street scenes
across 50 German cities, split into train (2,975) / val (500) / test (1,525,
labels withheld). Labels use the standard 19-class "trainId" scheme (the 34
raw label IDs collapse to these 19 classes + an ignored/void class — see
`ID_TO_TRAINID` in `segmentation_data.py`).

Training uses random 384x384 crops of the half-resolution (`DOWNSAMPLE=0.5`)
image; evaluation instead tiles the full-resolution image with a sliding
window (see "Model Card" above for the mosaic vs. Hann-window distinction).

## Pipeline

| Stage | Description |
|---|---|
| Full-precision | PyTorch training from scratch (ImageNet-pretrained encoder, random decoder), weighted cross-entropy loss |
| ONNX export | `torch.onnx.export`, opset 12 |
| 8-bit quantization | `quantizeml.models.quantize` on the **ONNX graph directly** (not a Keras model — see "Why PyTorch?"), calibrated on real tiles; no QAT required |
| Conversion to Akida | `cnn2snn.convert` on the quantized ONNX model |

## Dataset setup

Cityscapes requires a free account (no API download token) — there is no
single public wget URL:

```bash
# 1. Register and download leftImg8bit_trainvaltest.zip and
#    gtFine_trainvaltest.zip from https://www.cityscapes-dataset.com/downloads/
mkdir -p data/cityscapes
unzip leftImg8bit_trainvaltest.zip -d data/cityscapes
unzip gtFine_trainvaltest.zip -d data/cityscapes
```

This produces `data/cityscapes/leftImg8bit/{train,val,test}/<city>/*.png`
and the matching `data/cityscapes/gtFine/...` label tree, which
`segmentation_data.py`'s `get_data()`/`get_samples()` expect. Both are
git-ignored (large, external).

## Reference Models

Pretrained models are made available here, within the `pretrained_models/`
folder. However, those are handled using the `git-lfs` package (git large
file storage). For those to be downloaded with the repo, you will need to
set up `git-lfs`. For further instructions, see the
[Trained models](../../../README.md#trained-models) section of the top-level README.

## Usage

### Notebook

[segmentation_notebook_training.ipynb](segmentation_notebook_training.ipynb)
walks through the complete pipeline end-to-end: PyTorch training, ONNX
export, quantizeml quantization, Akida conversion, and evaluation (mIoU,
mosaic and Hann-window tiling).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Brainchip-Inc/brainchip_devhub/blob/main/akida2/model_zoo/segmentation/segmentation_notebook_training.ipynb)

[segmentation_notebook_benchmark.ipynb](segmentation_notebook_benchmark.ipynb)
walks through evaluation of the converted Akida model and, if a hardware
device is available, latency benchmarking.

> **Note:** the hardware benchmark section requires a physical Akida 2 FPGA
> platform with a connected board.

### Script

Once the dataset is built (see "Dataset setup" above), run the full
pipeline in one shot:

```bash
bash segmentation_train.sh [DATADIR]
```

The optional `DATADIR` argument overrides the default dataset location
(`./data/cityscapes`).

## Project history

This example distills a broader exploration of camera-based semantic
segmentation on Akida that also tried a full-image (downsampled 384x384,
single-pass, no tiling) approach and a dual-branch (full-res local tile +
global context) architecture — both left out of this standard zoo example
per scope, and available in the original project directory if needed. The
tiling approach here was chosen as the headline result; see
`docs/original_readme.txt` for the author's own account, including the
Akida-conversion accuracy regression mentioned there — **resolved** in the
checkpoint this example uses (see "Model Card" above for the before/after
numbers). `project_notebook_training.ipynb` and
`project_notebook_conversion.ipynb` are the original (lightly cleaned)
notebooks this example was distilled from, kept for reference alongside the
architecture diagrams in `docs/`.

## Contributing and Maintenance

This README is autogenerated from `docs/README.md.template`
so that the mIoU and hardware benchmark values are written directly
by the code (via the `metrics.json` file, also in the docs folder).

When the associated model or training pipeline is modified to improve
performance, you should rerun the evaluations of the float, ONNX, and Akida
model versions, plus the hardware benchmark, including the
`--save-metrics` argument, and then regenerate the README from the template
using `update_readme.py`:
```bash
# Float model
python segmentation_eval.py -l pretrained_models/unetv4akida_segmentation.pth --save-metrics

# ONNX export
python segmentation_eval.py -l pretrained_models/unetv4akida_segmentation.onnx --save-metrics

# Akida-converted
python segmentation_eval.py -l pretrained_models/unetv4akida_segmentation_i8_w8_a8.fbz --save-metrics
python segmentation_benchmark.py -l pretrained_models/unetv4akida_segmentation_i8_w8_a8.fbz --save-metrics

python update_readme.py
```
Then commit the changed files (template, metrics and updated README).

Likewise, if you want to edit the contents of this README, you should
not edit it directly, but instead edit `docs/README.md.template` and
then regenerate the README using
```bash
python update_readme.py
```
