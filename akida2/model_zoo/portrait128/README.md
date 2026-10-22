<img src="../../../docs/assets/0.-BC-dev-hub-LOGO-flicker.svg" alt="BrainChip Dev Hub" width="200"/>

# Portrait Segmentation (Portrait128) — Akida 2

## Model Card

Float binary IoU: **0.8883** &nbsp;|&nbsp; Float pixel accuracy: **94.31%** &nbsp;|&nbsp; Parameters: **1,058,865**

The model labels every pixel of a 128 × 128 portrait as person or background. Two
metrics are reported, both on the validation split and with each pixel's logit
thresholded at 0 (a probability of 0.5):

- **Binary IoU**: the intersection over union of the predicted and true masks,
  averaged over the two classes (person and background). This is the headline
  metric: it penalizes a mask that is the wrong shape even when most pixels are right.
- **Pixel accuracy**: the fraction of pixels labeled correctly. Background covers
  much of each image, so this is always high and says less about mask quality.

The quantized variants below all share the same float model. On Akida 2 the
model is quantized with **`quantizeml`**: 8-bit weights and activations need no
quantization-aware training (QAT), while a lower-precision 4-bit variant (4-bit
weights and activations, 8-bit input layer) uses QAT to recover accuracy — 4-bit
PTQ accuracy is poor, so only the QAT result is reported.

<table>
  <thead>
    <tr>
      <th>Variant</th>
      <th>Weights / Acts</th>
      <th>QAT</th>
      <th>Quantized IoU</th>
      <th>Quantized pixel acc.</th>
      <th>Akida IoU</th>
      <th>Akida pixel acc.</th>
      <th>Sparsity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>8-bit</td>
      <td align="center">w8 / a8</td>
      <td align="center">-</td>
      <td align="center">0.8861</td>
      <td align="center">94.19%</td>
      <td align="center">0.8861</td>
      <td align="center">94.19%</td>
      <td align="center">TBD</td>
    </tr>
    <tr>
      <td>4-bit</td>
      <td align="center">w4 / a4</td>
      <td align="center">yes</td>
      <td align="center">0.9091</td>
      <td align="center">95.48%</td>
      <td align="center">0.9091</td>
      <td align="center">95.48%</td>
      <td align="center">TBD</td>
    </tr>
  </tbody>
</table>

**Akida 2 hardware benchmark (FPGA @ 25 MHz)**

Latency is measured on the Akida 2 FPGA reference platform, which runs at
**25 MHz**. A projected latency at a higher target clock is also shown to
indicate expected performance on faster silicon. The cycle count is fixed for a
given model and mapping regardless of clock rate, so the projection is an exact
rescale of the measured cycles.

> **Note:** the projected clock is BrainChip's current **target** for AKD2500 production
> silicon — that silicon does not exist yet, so treat this as a target rather than a
> measured value; it may still change before production. Power measurement on the FPGA
> platform is still under development, so only latency is reported at this time.

<table>
  <thead>
    <tr>
      <th>Variant</th>
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
      <td rowspan="3">8-bit</td>
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
    <tr>
      <td rowspan="3">4-bit (QAT)</td>
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

### Benchmark plots

📝 Planned: the full-model and per-layer benchmark plots for each variant are added
when the FPGA benchmark is run with `--save-metrics` (see
[Contributing and Maintenance](#contributing-and-maintenance)).

The model is an **Akida U-Net** from `akida_models`: an **AkidaNet** encoder with
width multiplier **alpha = 0.5**, initialized from ImageNet-trained weights, takes the
**128 × 128** input down to a 4 × 4 feature map. A decoder of five separable
transposed-convolution blocks upsamples it back to 128 × 128, and a 1 × 1
convolution produces one logit per pixel. There are no skip connections between
encoder and decoder. Transposed convolutions are an Akida 2 feature, so this model
has no Akida 1 version.

## Requirements

For environment requirements and setup, see the [Requirements](../../../README.md#requirements)
section of the top-level README.

## Dataset

Portrait128 is a binary person-segmentation dataset of **18,698** RGB portraits at
**128 × 128**, each with a person/background mask. It was assembled by Anil Sathyan
for the [Portrait-Segmentation](https://github.com/anilsathyan7/Portrait-Segmentation)
project from the PFCN portrait dataset, hand-picked portraits from the Supervisely
person dataset, and selfies collected from the web whose masks were generated with a
DeepLab model. It also includes augmented and synthetic-background variants.

The dataset ships as a single pool. The scripts split it 80 % / 20 % into training and
validation, with a fixed shuffle seed, so every run sees the same validation images.

<!-- TODO (needs-human): confirm the dataset licence. The Portrait-Segmentation code
repository is MIT-licensed, but no separate licence is stated for the dataset, which
combines images from several sources. -->

Reference: Shen et al., *Automatic Portrait Segmentation for Image Stylization*,
Computer Graphics Forum (Eurographics), 2016 (the PFCN dataset).

## Dataset setup

The dataset is distributed by the Portrait-Segmentation project as
[Portseg_128](https://drive.google.com/file/d/1UBLzvcqvt_fin9Y-48I_-lWQYfYpt_6J/view)
(Google Drive). The scripts expect its two arrays, `img_uint8.npy` (images) and
`msk_uint8.npy` (masks, 0 or 255), in one folder.

The scripts default to looking for the data at `./data/portrait128`. If you
want to store the dataset on a dedicated data drive, you can pass the path
explicitly to each script (see `--data` / `-d` in the individual scripts).
Alternatively, it may be more convenient to keep the dataset in its preferred
location and create a symbolic link from the default path (one-off step):

```bash
ln -s /path/to/your/data/portrait128 ./data/portrait128
```

This way the scripts work out of the box without any extra arguments.

### Calibration samples

Quantization with `quantizeml` is calibrated on real data, so a batch of 1024
Portrait128 samples is needed in addition to the dataset itself. A prepared batch is
published on the BrainChip mirror, and `portrait128_train.sh` fetches it into `data/`
automatically:

```bash
wget -N https://data.brainchip.com/dataset-mirror/samples/portrait128/portrait128_batch1024.npz -P data/
```

The training notebook does not use this file. It builds its samples directly
from the training split with `get_samples()` from
[portrait128_data.py](portrait128_data.py), so that the process is visible and easy
to adapt to your own data. The same module can also write a pack from your local
copy of the dataset:

```bash
python portrait128_data.py --extract data/portrait128_batch1024.npz
```

## Pipeline

Training produces a float model, then quantizes it with `quantizeml` into
several variants, each converted to Akida format:

| Stage | Description |
|---|---|
| Full-precision | Float32 training of the full model, starting from an ImageNet-trained AkidaNet encoder |
| 8-bit quantization | `quantizeml quantize` to 8-bit weights and activations (8-bit input), calibrated on real samples; no QAT required |
| 4-bit quantization | `quantizeml quantize` to 4-bit weights and activations (8-bit input), calibrated on real samples, with QAT fine-tuning — 4-bit PTQ accuracy is poor so only the QAT model is kept |
| Conversion to Akida | Automated conversion of each quantized model to Akida 2 format with `cnn2snn convert` |

## Reference Models

📝 Planned: pretrained models will be made available in a `pretrained_models/`
folder once the training pipeline has been run. They are handled using the `git-lfs`
package (git large file storage); for set-up instructions, see the
[Trained models](../../../README.md#trained-models) section of the top-level README.

## Usage

### Notebook

Two notebooks are provided that walk through a) preparation of a trained Akida-compatible model and
b) evaluation and benchmarking of that model on Akida.

[portrait128_notebook_training.ipynb](portrait128_notebook_training.ipynb) walks through the
complete training pipeline end-to-end. It is written to expose and explain the Akida-specific
aspects of the workflow: how a segmentation model is built for Akida 2, what the
quantization constraints mean in practice, and what the conversion step does. Start
here if you want to understand *why* the pipeline is structured the way it is.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Brainchip-Inc/brainchip_devhub/blob/main/akida2/model_zoo/portrait128/portrait128_notebook_training.ipynb)

[portrait128_notebook_benchmark.ipynb](portrait128_notebook_benchmark.ipynb) walks through
evaluation of model accuracy on Akida and, if a hardware device is available, covers benchmarking
of model latency.

> **Note:** the hardware benchmark section requires a physical Akida 2 FPGA
> platform with a connected board.

### Script

For straightforward reproduction of the training and evaluation results, run
the full pipeline in one shot:

```bash
bash portrait128_train.sh [DATADIR]
```

The optional `DATADIR` argument overrides the default dataset location
(`./data/portrait128`).

## Contributing and Maintenance

This README is autogenerated from `docs/README.md.template`
so that the accuracy and hardware benchmark values are written directly
by the code (via the `metrics.json` file, also in the docs folder).

When the associated model or training pipeline is modified to improve
performance, you should rerun the evaluations of the float and quantized
model versions, plus the hardware benchmark, including the
`--save-metrics` argument, and then regenerate the README from the template
using `update_readme.py`:
```bash
# Float model
python portrait128_eval.py -l pretrained_models/akida_unet_portrait128.h5 --save-metrics

# 8-bit variant
python portrait128_eval.py -l pretrained_models/akida_unet_portrait128_i8_w8_a8.h5 --save-metrics
python portrait128_eval.py -l pretrained_models/akida_unet_portrait128_i8_w8_a8.fbz --save-metrics
python portrait128_benchmark.py -l pretrained_models/akida_unet_portrait128_i8_w8_a8.fbz --save-metrics

# 4-bit variant (QAT)
python portrait128_eval.py -l pretrained_models/akida_unet_portrait128_i8_w4_a4_qat.h5 --save-metrics
python portrait128_eval.py -l pretrained_models/akida_unet_portrait128_i8_w4_a4_qat.fbz --save-metrics
python portrait128_benchmark.py -l pretrained_models/akida_unet_portrait128_i8_w4_a4_qat.fbz --save-metrics

python update_readme.py
```
Then commit the changed files (template, metrics and updated README).

Likewise, if you want to edit the contents of this README, you should
not edit it directly, but instead edit `docs/README.md.template` and
then regenerate the README using
``` bash
python update_readme.py
```
