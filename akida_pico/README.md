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
  <a href="https://discord.com/invite/9bmd9g52vn"><img src="https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white" alt="Join the BrainChip Discord community"/></a>
  <a href="https://developer.brainchip.com/signup/"><img src="https://img.shields.io/badge/BrainChip%20Developer%20Hub-Sign%20up-blue.svg" alt="BrainChip Developer Hub - Sign up"/></a>
</p>

# Akida Pico

<p align="center">
  <a href="#the-technology"><u>Technology</u></a> ·
  <a href="#hardware"><u>Hardware</u></a> ·
  <a href="#model-zoo"><u>Model Zoo</u></a>
</p>

Akida Pico is BrainChip's ultra-low-power, event-based neural processing IP for always-on
sensing of 1-D signals such as audio, vibration and biosignals. Everything here targets Pico:

- **[Model zoo](#model-zoo).** Ready-to-run models, each covering data, training,
  quantization, conversion, evaluation and latency on the Pico FPGA.

> 🚧 **Work in progress.** More Pico examples will be added.

---

## The technology

What changes for you as a developer, compared with the Akida 1 and Akida 2 examples:

- **1-D signals, raw in.** Pico models read the waveform directly, one small chunk at a time, so
  there is no FFT, MFCC or spectrogram front end to build or run.
- **Recurrent state-space layers.** Models are trained in a *kernelized* form, which processes a
  whole window at once, then converted to an exactly equivalent *stateful* form that takes one
  chunk at a time and carries a small state between calls. Memory and compute stay constant
  however long the stream runs.
- **Post-training quantization.** The `quantizeml` flow gives 8-bit weights and activations, with
  a 16-bit input for raw audio.
- **One mapping.** The model maps to a single `TNP_R`, the Pico recurrent processing element, so
  there are no mapping modes to compare.

## Hardware

Pico is delivered as IP. The examples here are benchmarked on the **Pico FPGA** reference
platform, clocked at 50 MHz. The number of clock cycles per inference does not depend on the
clock, so the latency at another clock follows from the cycle count. Power is not measured on
the FPGA, so no energy figures are reported.

---

## Model zoo

Each task links to its example folder, where you'll find the full model card, the accuracy at
each stage of the pipeline, and the steps to reproduce them.

### Audio

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Category</th>
      <th>Dataset</th>
      <th>Variant</th>
      <th>Performance</th>
      <th>Latency per chunk @ 50 MHz FPGA (ms)</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="model_zoo/kws">Keyword spotting</a></td>
      <td>Classification</td>
      <td>Google Speech Commands</td>
      <td>8-bit</td>
      <td>95.07% acc.</td>
      <td align="right">1.448</td>
      <td>10 keywords + silence + unknown, raw audio</td>
    </tr>
  </tbody>
</table>

<sub>Performance is for the converted Akida model on the Pico FPGA. Latency is per chunk of
input, measured on the Pico FPGA at 50 MHz. This table is generated from each example's
`docs/metrics.json`.</sub>
