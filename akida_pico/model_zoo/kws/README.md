<img src="../../../docs/assets/0.-BC-dev-hub-LOGO-flicker.svg" alt="BrainChip Dev Hub" width="200"/>

# Keyword Spotting (KWS) — Akida Pico

## Model Card

Float accuracy: **95.89%** &nbsp;|&nbsp; Parameters: **47,584**

A 12-class keyword spotter for Akida Pico: the ten Speech Commands keywords
(*down, go, left, no, off, on, right, stop, up, yes*) plus *silence* and *unknown*. It reads
**raw 16 kHz audio** (16-bit samples) with no FFT, MFCC or spectrogram front end, one
256-sample chunk at a time, so it produces one decision every **16 ms**. It is quantized
after training (no QAT) to 8-bit weights and activations with a 16-bit input, and maps to a
single `TNP_R` on Pico.

**Accuracy at each stage of the pipeline**

Every row scores the same test clips, one prediction per clip.

<table>
  <thead>
    <tr>
      <th>Stage</th>
      <th>Form</th>
      <th>Precision</th>
      <th>Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Float, kernelized</td>
      <td>whole 1 s clip</td>
      <td>float32</td>
      <td align="center">95.89%</td>
    </tr>
    <tr>
      <td>Float, stateful</td>
      <td>256-sample chunks</td>
      <td>float32</td>
      <td align="center">95.89%</td>
    </tr>
    <tr>
      <td>Quantized, stateful</td>
      <td>256-sample chunks</td>
      <td>int16 input, w8 / a8</td>
      <td align="center">95.03%</td>
    </tr>
    <tr>
      <td>Akida model, software simulator</td>
      <td>256-sample chunks</td>
      <td>int16 input, w8 / a8</td>
      <td align="center">95.05%</td>
    </tr>
    <tr>
      <td>Akida model, Pico FPGA</td>
      <td>256-sample chunks</td>
      <td>int16 input, w8 / a8</td>
      <td align="center">95.07%</td>
    </tr>
  </tbody>
</table>

**Akida Pico hardware benchmark (FPGA @ 50 MHz)**

The whole network maps to **one hardware sequence on one `TNP_R`**, so there is a single mapping
to report. One forward pass consumes one 256-sample chunk, so real time needs one pass per
16 ms.

<table>
  <thead>
    <tr>
      <th>Mapping</th>
      <th>Latency per chunk @ 50 MHz (ms)</th>
      <th>Throughput (chunks/s)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1 sequence, 1 × TNP_R</td>
      <td align="center">1.448</td>
      <td align="center">690.6</td>
    </tr>
  </tbody>
</table>

> **Note:** latency is measured on the Pico FPGA reference platform. Power is not measured
> on the FPGA, so no energy figures are reported. The hardware values above (the FPGA
> accuracy and the latency) are reference values recorded from the `examples/kws` notebook in [bc_cloud_pico](https://github.com/Brainchip-Inc/bc_cloud_pico) (commit `818dfc9`), run on the Pico FPGA. Re-run
> `kws_benchmark.py` and `kws_eval.py --hardware` with `--save-metrics` on your own Pico
> platform to regenerate them.

## Requirements

For environment requirements and setup, see the [Requirements](../../../README.md#requirements)
section of the top-level README.

The dataset preparation also needs `ffmpeg` (TensorFlow Datasets uses it to decode the
audio): `conda install -c conda-forge ffmpeg`.

## How the model differs from the other examples

Pico targets always-on **1-D** signals, and the model is built around that:

- **Raw audio in.** There is no feature extraction, so the model sees int16 samples directly.
  Keep the input on one global scale: do not normalise each clip to its own peak, because a
  live microphone has no per-clip peak.
- **Two forms of one network.** Training uses the *kernelized* form of each state-space (SSM)
  block, which processes a whole 16,384-sample clip at once and trains efficiently. Pico runs
  the equivalent *stateful* form, which takes 256 samples at a time and carries a small state
  between calls, so memory and compute stay constant however long the stream runs. The
  conversion (`convert_to_stateful`) is exact.
- **Why 256 samples.** The model downsamples time by 256×, so 256 samples produce exactly one
  output step. That is the smallest chunk that completes an output, and the source of the
  16 ms decision period.
- **16-bit input.** Weights and activations are 8-bit, but the input stays 16-bit: 8 bits
  would discard most of the dynamic range of the audio.
- **Per-clip scoring.** A clip's prediction is the argmax of the **mean of its 64 per-chunk
  logit vectors**, not of the mean softmax probabilities.

## Dataset

[Speech Commands v0.02](https://www.tensorflow.org/datasets/catalog/speech_commands) is a
set of one-second recordings of 35 spoken words by thousands of speakers. This example uses
the standard 12-class task: ten keywords, plus *silence* (background noise) and *unknown* (all
the other words). It is released under the
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licence. Clips are zero-padded to
**16,384 samples** (2<sup>14</sup>, about 1.02 s), which the kernelized layers need to be a
power of two.

Reference: Warden, *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition*
(2018), [arXiv:1804.03209](https://arxiv.org/abs/1804.03209).

## Dataset setup

Nothing to do by hand: TensorFlow Datasets downloads and prepares the dataset the first time
a script needs it (about 2 GB, then cached in `~/tensorflow_datasets`). To keep it elsewhere,
pass the directory with `--data` / `-d` to any script.

### Calibration samples

Quantization with `quantizeml` is calibrated on real audio: 100 segments of 1,024 samples,
each cut from the middle of a training clip. The set used for the pretrained model is
committed as `pretrained_models/sc12_calibration_batch100_1024samples.npz`.
To build your own from your training data, for example for your own keywords:

```bash
tenn_sc12_train extract -m models/tenn_recurrent_sc12.h5 -s data/calibration.npz -ns 1024 -b 100
```

Calibrate on audio that covers every class and the noise conditions you expect. A large
accuracy drop at the quantized stage usually means the calibration data does not match your
deployment audio.

## Pipeline

| Stage | Description |
|---|---|
| Full-precision | Float32 training of the kernelized model, from scratch (200 epochs) |
| Stateful conversion | `convert_to_stateful(model, 256)`: same weights, 256-sample chunk interface; exact |
| Quantization | `quantizeml` post-training quantization to 8-bit weights and activations with 16-bit input, calibrated on real samples; no QAT |
| Conversion to Akida | `cnn2snn convert` of the quantized model |
| Mapping on Pico | `map(device, hw_only=True)`: expect 1 sequence and 1 `TNP_R` |

## Reference Models

Pretrained models are made available here, within the `pretrained_models/`
folder. However, those are handled using the `git-lfs` package (git large
file storage). For those to be downloaded with the repo, you will need to
set up `git-lfs`. For further instructions, see the
[Trained models](../../../README.md#trained-models) section of the top-level README.

| File | What it is |
|---|---|
| `tenn_recurrent_sc12.h5` | Float, kernelized weights of `akida_models.tenn_recurrent_sc12()` (12 classes) |
| `sc12_calibration_batch100_1024samples.npz` | Quantization calibration samples |

The stateful, quantized and Akida versions are produced from these by the pipeline, so only
the float weights are stored.

## Usage

### Notebook

[kws_notebook_evaluation.ipynb](kws_notebook_evaluation.ipynb) takes the pretrained weights
through stateful conversion, quantization, Akida conversion and Pico mapping, checks accuracy
at each stage, and then streams audio through the model one chunk at a time, as it runs on
Pico. Training is skipped (it takes 200 epochs); `kws_train.py` reproduces it.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Brainchip-Inc/brainchip_devhub/blob/main/akida_pico/model_zoo/kws/kws_notebook_evaluation.ipynb)

> **Note:** the hardware sections require a Pico FPGA platform, for example through Akida Cloud.
> Without one, the notebook evaluates with the software simulator.

### Script

For straightforward reproduction of the results, run the full pipeline in one shot:

```bash
bash kws_train.sh [DATADIR]
```

The optional `DATADIR` is the TensorFlow Datasets directory. Set `SKIP_TRAINING=1` to start
from the pretrained weights instead of training for 200 epochs.

## Adapting it to your keywords

Keeping the architecture and changing only the output head keeps the hardware structure of
this model. Measure the cycles per chunk after mapping, rather than assuming they match.
BrainChip's [Customizing the Akida Pico KWS Model](https://github.com/Brainchip-Inc/pico_guide/blob/main/customer/Customizing_the_Akida_Pico_KWS_Model.md)
guide covers the path: keyword set, data, fine-tuning, quantization, mapping and trigger logic.
Your data decides the accuracy: the figures above do not transfer to other keywords or noise.

### Train on your own recordings

`kws_custom.sh` runs the whole pipeline on a folder of your own `.wav` files, from training with
augmentation to the Akida model, and scores a held-out test split at every stage:

```
data/keywords/
    hello/*.wav              one folder per keyword
    stop/*.wav
    silence/*.wav            optional; otherwise cut from _background_noise_/
    unknown/*.wav            optional; every other folder also counts as "unknown"
    _background_noise_/*.wav long noise recordings: silence class and augmentation
```

1. List your keywords in [`kws_config.yaml`](kws_config.yaml). The classes are the keywords in that
   order, then `silence` and `unknown`: the same head structure as the 12-class model above. A
   model trained this way maps on the software Pico device as a single sequence, but measure it on
   the Pico FPGA, as in "Adapting it to your keywords". Quote `yes`, `no`, `on` and `off` so YAML
   keeps them as strings.
2. Run `bash kws_custom.sh kws_config.yaml [DATADIR]`. Set `SKIP_TRAINING=1` to reuse
   `models/kws_custom.h5`.

What differs from `kws_train.py`:

- **Splits.** Files go to train, validation and test by a hash of the file name, so the split is
  the same on every run. Name files `<speaker>_nohash_<n>.wav` and no speaker is in two splits.
  The best epoch is picked on validation; the reported accuracy is on test, which training never sees.
- **Augmentation.** The `augmentations` list in the config applies speed, time shift, room echo,
  background noise and gain to training clips only. Add your own as `name: "module:factory"`
  (see `kws_augment.py`).
- **Your own model.** Set `model: my_module:build_model` to a function that takes `num_classes`
  and returns an untrained Keras model with a `(16384, 1)` input. Re-check the hardware mapping
  if you change the architecture.
- **Quantization.** Same post-training quantization as `kws_train.sh` (8-bit weights and
  activations, 16-bit input), calibrated on clips from your training split.

Recordings can be any sample rate, mono or stereo; they are resampled to 16 kHz and fitted to the
model's 16,384-sample window. Convert other formats first (`ffmpeg -i in.mp3 out.wav`). Check the licence of any
recordings you add before redistributing them.

To try the pipeline before recording anything, unpack the raw
[Speech Commands v0.02 archive](https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)
(the dataset above, same [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licence) into
`data/keywords/`. It already has this layout, including `_background_noise_/` and
`<speaker>_nohash_<n>.wav` names. Then set `keywords` in the config to words from it, for example
`['yes', 'no', 'up', 'down']`.

## Contributing and Maintenance

This README is autogenerated from `docs/README.md.template`
so that the accuracy and hardware benchmark values are written directly
by the code (via the `metrics.json` file, also in the docs folder).

When the associated model or training pipeline is modified to improve
performance, you should rerun the evaluations of each model stage, plus the
hardware benchmark, including the `--save-metrics` argument, and then regenerate the
README from the template using `update_readme.py`:
```bash
# Float model, kernelized
python kws_eval.py -l pretrained_models/tenn_recurrent_sc12.h5 --save-metrics

# Stateful float, quantized and Akida models (see kws_train.sh for how they are made)
python kws_eval.py -l models/tenn_recurrent_sc12_stateful.h5 --save-metrics
python kws_eval.py -l models/tenn_recurrent_sc12_stateful_i8_w8_a8.h5 --save-metrics
python kws_eval.py -l models/tenn_recurrent_sc12_stateful_i8_w8_a8.fbz --save-metrics

# On a machine with a Pico FPGA
python kws_eval.py -l models/tenn_recurrent_sc12_stateful_i8_w8_a8.fbz --hardware --save-metrics
python kws_benchmark.py -l models/tenn_recurrent_sc12_stateful_i8_w8_a8.fbz --save-metrics

python update_readme.py
```
Then commit the changed files (template, metrics and updated README).

Likewise, if you want to edit the contents of this README, you should
not edit it directly, but instead edit `docs/README.md.template` and
then regenerate the README using
``` bash
python update_readme.py
```
