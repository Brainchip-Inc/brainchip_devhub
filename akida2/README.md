<!-- GENERATED FILE: edit akida2/docs/README.md.template, then run akida2/update_readme.py -->
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
  <img src="https://img.shields.io/badge/MetaTF-akida%202.19.2-orange.svg" alt="MetaTF (akida 2.19.2)"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/hardware-Akida%202%20FPGA%20(Akida%20Cloud)-orange.svg" alt="Hardware: Akida 2 FPGA, through Akida Cloud"/>
  <a href="https://discord.com/invite/9bmd9g52vn"><img src="https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white" alt="Join the BrainChip Discord community"/></a>
  <a href="https://brainchip.com/aclp/"><img src="https://img.shields.io/badge/Akida%20Cloud-Request%20access-blue.svg" alt="Akida Cloud - Request access"/></a>
  <a href="https://developer.brainchip.com/signup/"><img src="https://img.shields.io/badge/BrainChip%20Developer%20Hub-Sign%20up-blue.svg" alt="BrainChip Developer Hub - Sign up"/></a>
</p>

# Akida 2

<p align="center">
  <a href="#the-technology"><u>Technology</u></a> ·
  <a href="#hardware"><u>Hardware</u></a> ·
  <a href="#model-zoo"><u>Model Zoo</u></a> ·
  <a href="#tutorials"><u>Tutorials</u></a>
</p>

Everything here targets the Akida 2 platform:

- **[Model zoo](#model-zoo).** Ready-to-run models:
  - each one covers data, training, quantization, conversion and evaluation
  - each is benchmarked on the **Akida 2 FPGA** (latency, with a projection to the AKD2500 target clock)
  - each comes in an 8-bit and a 4-bit variant, so you can compare the two
- **[Tutorials](#tutorials)** *(coming)*.

> 🚧 **Work in progress.** Content is being added and refined regularly. More Akida 2
> models are listed in the [official model zoo](https://doc.brainchipinc.com/model_zoo_performance.html#akida-2-0-models).

---

## The technology

Akida 2 keeps what makes Akida efficient: it is event-based, computing only on non-zero
activations, so **activation sparsity directly saves time and energy**, and the whole
network runs on-chip across the Neural Processors (NPs). What changes for you as a
developer:

- **8-bit by default.** Akida 2 models are quantized with
  [`quantizeml`](https://doc.brainchipinc.com) rather than `cnn2snn.quantize`. With 8-bit
  weights and activations, post-training quantization alone usually comes close to float
  accuracy, so no quantization-aware training (QAT) is needed. Every example here also
  has a 4-bit variant, which does use QAT.
- **Signed activations between layers.** Standard NPs take int8 inputs, so layers can
  receive the output of non-ReLU activations.
- **A wider input layer.** The dedicated input layer accepts a second input dimension of
  up to 384 (256 on Akida 1).

These examples cover the same tasks as the Akida 1 model zoo, built with the Akida 2
pipeline. Examples of
what only Akida 2 can run (skip connections, non-ReLU activations, upsampling for
detection and segmentation) are on the [roadmap](../ROADMAP.md).

## Hardware

No Akida 2 silicon is available yet.

| Platform | NPs | HWPR enabled<sup>†</sup> | Clock | |
| --- | :---: | :---: | :---: | --- |
| **Akida 2 FPGA** | 24 (6 nodes) | Yes | 25 MHz | Reference implementation of Akida 2. Every Akida 2 benchmark in this repo is measured on it. Available for testing through [Akida Cloud](https://brainchip.com/aclp/) |
| **AKD2500** | | Yes | 1 GHz *(target)* | Production silicon, in development. The clock is BrainChip's current target and may change |

<sub>† **HWPR** (Hardware Partial Reconfiguration): a model is split into a series of
sub-models ("passes"), loaded one after another for each inference, so each layer can use
more NPs. A node is 4 NPs.</sub>

The FPGA runs at 25 MHz, far below production silicon, so its absolute latencies are
long. The number of clock cycles per inference does not depend on the clock, so each
example also gives a **projected** latency at the AKD2500 target clock: an exact rescale
of the measured cycles, not an estimate. Power is not measured on the FPGA, so no energy
figures are reported for Akida 2 yet.

---

## Model zoo

Each task links to its example folder, where you'll find the full model card, the
benchmarks for every mapping, and the steps to reproduce them.

### Image

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Category</th>
      <th>Dataset</th>
      <th>Variant</th>
      <th>Performance</th>
      <th>Mapping</th>
      <th>Latency @ 25 MHz FPGA (ms)</th>
      <th>Projected @ 1 GHz AKD2500 (ms)</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><a href="model_zoo/vww">Person detection</a></td>
      <td rowspan="2">Classification</td>
      <td rowspan="2">Visual Wake Words</td>
      <td>8-bit</td>
      <td>87.77% acc.</td>
      <td>AllNPs</td>
      <td align="right">34.982</td>
      <td align="right">0.875</td>
      <td></td>
    </tr>
    <tr>
      <td>4-bit QAT</td>
      <td>84.02% acc.</td>
      <td>AllNPs</td>
      <td align="right">29.584</td>
      <td align="right">0.740</td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="2"><a href="model_zoo/plant_village">Plant disease recognition</a></td>
      <td rowspan="2">Classification</td>
      <td rowspan="2">PlantVillage</td>
      <td>8-bit</td>
      <td>99.63% acc.</td>
      <td>HwPr</td>
      <td align="right">274.796</td>
      <td align="right">6.870</td>
      <td></td>
    </tr>
    <tr>
      <td>4-bit QAT</td>
      <td>99.65% acc.</td>
      <td>HwPr</td>
      <td align="right">332.960</td>
      <td align="right">8.324</td>
      <td></td>
    </tr>
  </tbody>
</table>

### Audio

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Category</th>
      <th>Dataset</th>
      <th>Variant</th>
      <th>Performance</th>
      <th>Mapping</th>
      <th>Latency @ 25 MHz FPGA (ms)</th>
      <th>Projected @ 1 GHz AKD2500 (ms)</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="2"><a href="model_zoo/speech_commands">Keyword spotting</a></td>
      <td rowspan="2">Classification</td>
      <td rowspan="2">Google Speech Commands</td>
      <td>8-bit</td>
      <td>95.77% acc.</td>
      <td>HwPr</td>
      <td align="right">4.511</td>
      <td align="right">0.113</td>
      <td>10 keywords + silence + unknown</td>
    </tr>
    <tr>
      <td>4-bit QAT</td>
      <td>94.67% acc.</td>
      <td>HwPr</td>
      <td align="right">3.455</td>
      <td align="right">0.086</td>
      <td>10 keywords + silence + unknown</td>
    </tr>
  </tbody>
</table>

<sub>Performance is for the converted Akida model. Latency is measured on the Akida 2 FPGA
at 25 MHz and projected to the AKD2500 target clock (1 GHz), for the fastest mapping mode
(the simplest one, if modes are within 1% of each other). This table is generated from each example's
`docs/metrics.json`.</sub>

---

## Tutorials

📝 Planned. Akida 2 tutorials will follow the Akida 1 series; see the
[roadmap](../ROADMAP.md).
