<!-- GENERATED FILE: edit akida_pico/docs/README.md.template, then run akida_pico/update_readme.py -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../docs/assets/0_-BC-dev-hub-LOGO-flicker-dark.svg">
    <img src="../docs/assets/0_-BC-dev-hub-LOGO-flicker-light.svg" alt="BrainChip Dev Hub" width="220"/>
  </picture>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License: Apache 2.0"/>
  <img src="https://img.shields.io/badge/python-3.10%20%E2%80%93%203.12-blue?logo=python&logoColor=white" alt="Python 3.10 – 3.12"/>
  <img src="https://img.shields.io/badge/akida__models-1.14.0-orange.svg" alt="akida_models 1.14.0"/>
  <img src="https://img.shields.io/badge/cnn2snn-2.19.x-orange.svg" alt="cnn2snn 2.19.x"/>
  <a href="https://discord.com/invite/9bmd9g52vn"><img src="https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white" alt="Join the BrainChip Discord community"/></a>
  <a href="https://developer.brainchip.com/signup/"><img src="https://img.shields.io/badge/BrainChip%20Developer%20Hub-Sign%20up-blue.svg" alt="BrainChip Developer Hub - Sign up"/></a>
</p>

# Akida Pico

<p align="center">
  <a href="#the-technology"><u>Technology</u></a> ·
  <a href="#hardware"><u>Hardware</u></a> ·
  <a href="#model-zoo"><u>Model Zoo</u></a>
</p>

Everything here targets the Akida Pico IP:

- **[Model zoo](#model-zoo).** Ready-to-run models, each covering data, training,
  quantization, conversion and evaluation, with the same layout as the
  [Akida 1](../akida1) and [Akida 2](../akida2) examples.
- **Hardware benchmarks.** 📝 Planned, on the Akida Pico FPGA
  ([#104](https://github.com/Brainchip-Inc/brainchip_devhub/issues/104)).

> 🚧 **Work in progress.** This is the first Akida Pico example; more will follow.
> The official model zoo lists the current
> [Akida Pico models](https://doc.brainchipinc.com/model_zoo_performance.html).

---

## The technology

Akida Pico models are **recurrent TENNs**. A recurrent TENN processes a signal as a
stream: it reads a short chunk of input at a time and carries an internal state from
one chunk to the next, instead of looking at a whole window at once. What that means
for you as a developer:

- **Train in one form, deploy in another.** Recurrent layers are trained as
  `Kernelized` layers, temporal convolutions that compute efficiently over a whole
  sequence on a GPU. `convert_to_stateful` then turns them into `StatefulRecurrent`
  layers, with the same weights, which process the stream chunk by chunk. The stateful
  model is the one you quantize and convert.
- **The usual MetaTF flow after that.** The stateful model is quantized with
  [`quantizeml`](https://doc.brainchipinc.com) (8-bit, no quantization-aware training
  in this example) and converted with `cnn2snn`, as for Akida 2.
- **Raw signals in.** The first layer accepts 16-bit inputs (up to 128 channels), so
  audio can go in as raw int16 samples, with no feature extraction in front.

## Hardware

| Platform | | |
| --- | --- | --- |
| **Akida Pico FPGA** | Benchmarks 📝 Planned ([#104](https://github.com/Brainchip-Inc/brainchip_devhub/issues/104)) | Mapping can be checked without hardware, on the virtual device `akida.PicoIP()` |

The limits a model must respect to map onto Akida Pico (number of recurrent layers,
input channels and bit widths, state sizes, subsampling) are listed in the
[Akida Pico hardware constraints](https://doc.brainchipinc.com/user_guide/hardware/pico.html).
Mapping onto the virtual device checks them, so each example maps its model there and
reports the result.

---

## Model zoo

Each task links to its example folder, where you'll find the full model card and the
steps to reproduce it.

### Audio

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Category</th>
      <th>Dataset</th>
      <th>Variant</th>
      <th>Performance</th>
      <th>Parameters</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="model_zoo/speech_commands">Keyword spotting</a></td>
      <td>Classification</td>
      <td>Google Speech Commands</td>
      <td>8-bit</td>
      <td>94.19% acc.</td>
      <td align="right">46,552</td>
      <td>Recurrent TENN on raw audio; 10 keywords + silence + unknown</td>
    </tr>
  </tbody>
</table>

<sub>Performance is for the converted Akida model, evaluated with the Akida runtime's
software simulation of the Pico IP, not on hardware. This table is generated from each
example's `docs/metrics.json`.</sub>
