<img src="../../../docs/assets/0.-BC-dev-hub-LOGO-flicker.svg" alt="BrainChip Dev Hub" width="200"/>

# Speech Commands Keyword Spotting (KWS) — Akida Pico

## Model Card

Akida accuracy: **94.19%** &nbsp;|&nbsp; Parameters: **46,552** &nbsp;|&nbsp;
Input: **raw 16 kHz audio**, int16

A recurrent TENN that listens to raw audio and recognises 10 keywords, plus silence
and unknown speech. Akida Pico runs recurrent TENNs: the model reads the audio in
chunks of 256 samples (16 ms) and carries an internal state from one chunk to the next,
so it can run on a continuous audio stream.

All numbers are on the Speech Commands test split (4,890 clips, balanced across the
12 classes), per 1-second clip. Macro-F1 averages the F1 score of the 12 classes.

<table>
  <thead>
    <tr>
      <th>Stage</th>
      <th>Model</th>
      <th>Accuracy</th>
      <th>Macro-F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Float, training form</td>
      <td>Kernelized, whole clip</td>
      <td align="center">95.15%</td>
      <td align="center">0.9513</td>
    </tr>
    <tr>
      <td>Float, stateful</td>
      <td>StatefulRecurrent, 256-sample chunks</td>
      <td align="center">95.17%</td>
      <td align="center">0.9515</td>
    </tr>
    <tr>
      <td>8-bit quantized</td>
      <td>w8 / a8, int16 input</td>
      <td align="center">94.25%</td>
      <td align="center">0.9424</td>
    </tr>
    <tr>
      <td><b>Akida</b></td>
      <td>converted with <code>cnn2snn</code></td>
      <td align="center"><b>94.19%</b></td>
      <td align="center"><b>0.9417</b></td>
    </tr>
  </tbody>
</table>

The Akida figures come from the Akida runtime's software simulation of the Pico IP.
They are not hardware measurements.

**Comparison with the `akida_models` reference.** The same evaluation run on
`akida_models.tenn_recurrent_sc12_pretrained()`, BrainChip's published model for this
task, gives:

| | Float, stateful | 8-bit quantized | Akida |
| --- | :---: | :---: | :---: |
| Accuracy | 95.24% | 93.80% | 93.80% |
| Macro-F1 | 0.9521 | 0.9380 | 0.9380 |

The [official model zoo](https://doc.brainchipinc.com/model_zoo_performance.html) lists
93.80% for that model.

**Mapping on Akida Pico.** Mapped onto the Pico virtual device (`akida.PicoIP()`), the
whole model runs in 1 hardware sequence(s) (0 in
software) on 1 TNP_R, with a 52,504-byte program.

**Hardware benchmark.** 📝 Planned: latency and power on the Akida Pico FPGA are
tracked in [#104](https://github.com/Brainchip-Inc/brainchip_devhub/issues/104).

## The model, and why it fits Akida Pico

The model is [`tenn_recurrent_sc12`](https://doc.brainchipinc.com/api_reference/akida_models_apis.html#recurrent-tenns)
from `akida_models`: six recurrent TENN blocks of 8, 16, 32, 64, 96 and 128 channels,
each subsampling time by 4, 4, 2, 2, 2 and 2, and a 12-class recurrent head. Its input
is the raw waveform, so no MFCC or other feature extraction runs before the model. The
[Akida 1](../../../akida1/model_zoo/speech_commands) and
[Akida 2](../../../akida2/model_zoo/speech_commands) Speech Commands examples, by
contrast, feed the model 49 × 10 MFCC features.

The same weights exist in two forms:

- **Training form.** Each recurrent layer is a `Kernelized` layer: a temporal
  convolution whose kernel is generated from a small set of state-space parameters.
  Over a whole 16,384-sample clip this computes efficiently on a GPU, so this is the
  form that is trained.
- **Stateful form.** `convert_to_stateful` turns each layer into a `StatefulRecurrent`
  layer that processes the clip 256 samples at a time and keeps its internal state
  between chunks. This is the form that is quantized, converted and run on Akida Pico.
  It produces one output per chunk; the clip's prediction is the class with the
  highest logit averaged over its 64 chunks.

The architecture stays inside the Akida Pico limits for `StatefulRecurrent` models
listed in the [Akida Pico hardware constraints](https://doc.brainchipinc.com/user_guide/hardware/pico.html):
7 recurrent layers (at most 8), a 1-channel 16-bit input (up to 128 channels at 16
bits), subsampling of at most 4 per layer, unbounded ReLU activations, and only a
`Dequantizer` after the last recurrent layer. Mapping checks these limits, which is
why the mapping above is a useful test even without a device attached.

## Requirements

For environment requirements and setup, see the [Requirements](../../../README.md#requirements)
section of the top-level README.

## Dataset

The dataset is Google Speech Commands, through TensorFlow Datasets
([`speech_commands`](https://www.tensorflow.org/datasets/catalog/speech_commands)): 1-second
clips at 16 kHz of 10 keywords (yes, no, up, down, left, right, on, off, stop, go), plus
`_silence_` and `_unknown_` (12 classes). Each clip is zero-padded or cropped to 16,384
samples (2<sup>14</sup>) and fed to the model as int16 values.

- **Train** (85,511 clips) is dominated by `_unknown_`, so each epoch draws a fresh
  class-balanced set of 3,100 clips per class.
- **Validation** (10,102 clips, also dominated by `_unknown_`) picks the best epoch, by
  macro-F1 rather than accuracy for that reason.
- **Test** (4,890 clips, balanced) gives every number in this README.

Original dataset licensed under the Creative Commons BY 4.0 license:
> P. WARDEN (2018), *"Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition"*,
> Google Brain, V1, doi: 10.48550/arXiv.1804.03209

## Dataset setup

TensorFlow Datasets downloads and prepares `speech_commands` (several GB) on first use,
into the path you pass with `--data` (default: `./data/speech_commands`). To fetch it
ahead of time:

```bash
python -c "import tensorflow_datasets as tfds; tfds.load('speech_commands', data_dir='./data/speech_commands')"
```

If the dataset already lives elsewhere, link it to the default path:

```bash
ln -s /path/to/your/data/speech_commands ./data/speech_commands
```

### Calibration samples

Quantization with `quantizeml` is calibrated on real audio. BrainChip publishes a
prepared set of int16 chunks from the training split, which `speech_commands_train.sh`
downloads into `data/`:

```bash
wget -N https://data.brainchip.com/dataset-mirror/samples/sc12/sc12_batch100_1024samples.npz -P data/
```

## Pipeline

| Stage | Description |
|---|---|
| Full-precision training | 200 epochs, AdamW (weight decay 0.05), learning rate 0.01 with a 2.5% linear warm-up then cosine decay, batch size 128 |
| Stateful conversion | `convert_to_stateful`, 256-sample chunks; same weights |
| 8-bit quantization | `quantizeml quantize`: 8-bit weights and activations, int16 input, calibrated on real samples; no QAT |
| Conversion to Akida | `cnn2snn convert`, then evaluation with the Akida runtime and mapping onto the Pico virtual device |

The training recipe is the one `akida_models` uses for this model, with one change:
`akida_models` picks the best epoch on the test split, while this example picks it on
the validation split and keeps the test split for reporting.

## Reference Models

Trained models are in `pretrained_models/`, stored with `git-lfs`. To download them with
the repo, set up `git-lfs` as described in the
[Trained models](../../../README.md#trained-models) section of the top-level README.

| File | Stage |
| --- | --- |
| `tenn_recurrent_sc12.h5` | Float, training form |
| `tenn_recurrent_sc12_stateful.h5` | Float, stateful |
| `tenn_recurrent_sc12_stateful_i8_w8_a8.h5` | 8-bit quantized |
| `tenn_recurrent_sc12_stateful_i8_w8_a8.fbz` | Akida |

## Usage

### Notebook

[speech_commands_notebook_training.ipynb](speech_commands_notebook_training.ipynb) walks
through the pipeline: the data, the model in its training form, the conversion to
stateful form, quantization, conversion to Akida, and mapping onto Akida Pico. Full
training takes a few hours on a GPU, so by default the notebook loads the trained
model from `pretrained_models/` and shows how to launch training.

### Script

To reproduce the whole pipeline in one go:

```bash
bash speech_commands_train.sh [DATADIR]
```

The optional `DATADIR` argument overrides the default dataset location
(`./data/speech_commands`).

## Contributing and Maintenance

This README is generated from `docs/README.md.template`, so that every number in it
is written by the code, through `docs/metrics.json`.

When the model or the training pipeline changes, rerun the evaluations with
`--save-metrics`, then regenerate the README:

```bash
python speech_commands_eval.py -l pretrained_models/tenn_recurrent_sc12.h5 --save-metrics
python speech_commands_eval.py -l pretrained_models/tenn_recurrent_sc12_stateful.h5 --save-metrics
python speech_commands_eval.py -l pretrained_models/tenn_recurrent_sc12_stateful_i8_w8_a8.h5 --save-metrics
python speech_commands_eval.py -l pretrained_models/tenn_recurrent_sc12_stateful_i8_w8_a8.fbz --save-metrics
python speech_commands_eval.py --reference --save-metrics   # akida_models' published model

python update_readme.py
```

Then commit the template, the metrics and the regenerated README. To change this
README's text, edit `docs/README.md.template` and run `python update_readme.py`;
don't edit README.md directly.
