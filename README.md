<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/0_-BC-dev-hub-LOGO-flicker-dark.svg">
    <img src="docs/assets/0_-BC-dev-hub-LOGO-flicker-light.svg" alt="BrainChip Dev Hub" width="260"/>
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
  <img src="https://img.shields.io/badge/hardware-AKD2500%20%7C%20AKD1500%20%7C%20AKD1000-orange.svg" alt="Supported hardware: AKD2500, AKD1500, AKD1000"/>
  <a href="https://discord.com/invite/9bmd9g52vn"><img src="https://img.shields.io/badge/Discord-Join%20the%20community-5865F2?logo=discord&logoColor=white" alt="Join the BrainChip Discord community"/></a>
  <a href="https://shop.brainchipinc.com/"><img src="https://img.shields.io/badge/Shop-BrainChip-blue.svg" alt="BrainChip Shop"/></a>
  <a href="https://developer.brainchip.com/signup/"><img src="https://img.shields.io/badge/BrainChip%20Developer%20Hub-Sign%20up-blue.svg" alt="BrainChip Developer Hub - Sign up"/></a>
</p>

# BrainChip Developer Hub

<p align="center">
  <a href="https://doc.brainchipinc.com"><u>Docs</u></a> ·
  <a href="#get-akida-hardware"><u>Hardware</u></a> ·
  <a href="#getting-started"><u>Getting Started</u></a> ·
  <a href="#platform-overview"><u>Examples</u></a> ·
  <a href="#repository-structure"><u>Structure</u></a> ·
  <a href="#community-and-support"><u>Community</u></a> ·
  <a href="#license"><u>License</u></a>
</p>

Resources for developing and deploying AI models on BrainChip Akida neuromorphic processors — training, conversion, evaluation, deployment, and benchmarking.

> This repository complements the [official BrainChip documentation](https://doc.brainchipinc.com). It focuses on practical, runnable examples and insider knowledge for getting the best out of Akida hardware.

---

## What do you want to do?

| Goal | Where to go |
| --- | --- |
| Train, convert, and evaluate a model | [Akida 1](akida1) · [Akida 2](akida2) · Akida Pico (**COMING SOON**) |
| Deploy to hardware and benchmark | Deployment (**COMING SOON**) |
| Understand how Akida works | Concepts (**COMING SOON**) |
| New to Akida — not sure where to start | [Getting Started](#getting-started) |

---

## Platform overview

Akida 1 and Akida 2 examples are available today; more Akida 2 models can be found [here](https://doc.brainchipinc.com/model_zoo_performance.html#akida-2-0-models) in the official docs, and Akida Pico content for this repo is on the way.

| | Akida 1 | Akida 2 | Akida Pico |
| --- | --- | --- | --- |
| **Chip** | AKD1500 | AKD2500 | — |
| **Typical use cases** | Image classification, keyword spotting, object detection | Larger models, higher accuracy targets | Always-on sensing, edge inference |
| **Examples in this repo** | [Image Classification (PlantVillage)](akida1/model_zoo/plant_village) · [ImageNet / AkidaNet](akida1/model_zoo/imagenet_akidanet) · [Visual Wake Words](akida1/model_zoo/vww) · [Keyword Spotting](akida1/model_zoo/speech_commands) · [ECG Arrhythmia](akida1/model_zoo/arrhythmia_classification) | [Visual Wake Words](akida2/model_zoo/vww) | 🔜 Coming soon |

---

## Why Akida?

Akida is BrainChip's neuromorphic processor — it processes data event by event instead of running dense computation on every input, so it only spends power and cycles on activity that's actually there. In practice, that means real models running at a fraction of the latency and power of conventional hardware.

The examples in this repo make that concrete on real AKD1500 silicon: [Image Classification (PlantVillage)](akida1/model_zoo/plant_village) reaches 99.43% accuracy.

Full benchmark breakdowns, mapping comparisons, and reproduction steps live in each example's own README.

---

## Get Akida Hardware

Every result in this repo runs on real Akida silicon, not a simulation.

**Chips**
- [AKD1500](https://brainchip.com/chips/) — 22nm neuromorphic co-processor, up to 800 effective GOPS, pairs with any host CPU/MCU over PCIe or SPI
- **AKD1000** — the processor behind the PCIe and Raspberry Pi dev boards below; ARM Cortex-M4 host, Linux (x86-64/ARM) support

**Dev kits & boards**
- AKD1000 PCIe Development Board
- AKD1000 Raspberry Pi 4 Dev Kit
- AKD1000 Raspberry Pi 5 Dev Kit
- AKD1500 Edge AI Co-Processor

All available through the [BrainChip Shop](https://shop.brainchipinc.com/).

> **No hardware yet?** [Akida Cloud](https://shop.brainchipinc.com/) lets you test, benchmark, and validate models on real Akida hardware remotely — no board required.

---

## Getting Started

Go from a fresh clone to your first result in four steps.

1. **Clone the repo (with Git LFS).** Pretrained weights are stored with
   [Git LFS](https://git-lfs.com/), so set it up first — otherwise the weight
   files arrive as small text pointers. More detail in [Trained models](#trained-models).

   ```bash
   git lfs install        # one-time per machine
   git clone https://github.com/Brainchip-Inc/brainchip_devhub.git
   cd brainchip_devhub
   git lfs pull           # fetch the real model files
   ```

2. **Create an environment and install.** Python 3.10–3.12 in a fresh venv or
   conda env (details in [Requirements](#requirements)). `pip install -e .` pulls
   the full Python toolkit — TensorFlow and `akida_models` (which brings in the
   Akida / MetaTF packages).

   ```bash
   conda create -n brainchip_devhub_env python=3.12 -y
   conda activate brainchip_devhub_env
   pip install -v -e .
   ```

3. **(For on-device runs) set up hardware.** You can train, quantize, convert,
   and evaluate in simulation with no board. To reproduce the latency and power
   numbers you'll need a physical AKD1500 / AKD1000 device and its runtime/driver —
   see the [official installation guide](https://doc.brainchipinc.com). No hardware?
   [Akida Cloud](https://shop.brainchipinc.com/) runs models on real silicon remotely.

4. **Pick an example and follow its README.** Browse the
   [available examples](#platform-overview) across Akida 1 and Akida 2 and open the
   one you want under its `model_zoo/` directory
   ([`akida1/model_zoo/`](akida1/model_zoo) or [`akida2/model_zoo/`](akida2/model_zoo)).
   Each README walks you through dataset setup, evaluation, and hardware
   benchmarking, and lists the accuracy and power numbers you should expect.
   **New to Akida?** [`plant_village`](akida1/model_zoo/plant_village) is a good
   first run — the full pipeline goes end-to-end in about 20 minutes.

---

### Requirements

This section covers the *why* and the gotchas.

- **Python 3.10–3.12.** The range is pinned by the TensorFlow 2.19 and `akida_models` 1.14 dependencies; other Python versions won't have matching wheels. Use whatever environment manager you prefer (`venv`, `conda`, or Docker) — the quickstart uses conda.
- **What `pip install -e .` actually installs.** Beyond TensorFlow, it pulls `akida_models`, which brings in the Akida / MetaTF stack (`akida`, `cnn2snn`, `quantizeml`), plus the helpers the examples need: `pyftdi` (reads power measurements from the board over I²C), `pywavelets` and `wfdb` (used by the ECG example), and `ipykernel` for the notebooks. The full pinned list is in [`pyproject.toml`](pyproject.toml).
- **No separate toolkit install needed.** The Python toolkit comes from that one command; the [official installation guide](https://doc.brainchipinc.com) is only for the on-device runtime and drivers, which you need to run on real silicon — not for simulation.

### Trained models

Pretrained weights (`.h5`, `.fbz`) live in the repo but are tracked with [Git LFS](https://git-lfs.com/) rather than regular git: the binaries are large, so git stores a small text *pointer* in history and fetches the real file on demand, keeping clones fast.

- **Did LFS actually run?** If a weight file is only a few hundred bytes and opens as text starting with `version https://git-lfs.github.com/spec/v1`, you have a pointer, not a model — LFS didn't fetch it. `git lfs ls-files` shows what LFS is tracking.
- **Fixing a pointer-only checkout.** Install Git LFS, then pull the real files:
  ```bash
  sudo apt install git-lfs   # linux; see git-lfs.com for other platforms
  git lfs install            # one-time per machine
  git lfs pull               # fetch the real files for this clone
  ```

---

## Anatomy of a model_zoo example

Every example under `model_zoo/` follows the same layout and naming convention, so
once you've run one you can find your way around any of them.

<details>
<summary><b>What you'll find inside an example folder</b></summary>

<br>

Each example is a self-contained folder named after its task (e.g. `plant_village/`),
with files following an `<example>_<role>` convention:

| File / folder | What it is |
|---|---|
| `<example>_model.py` | Model architecture definition |
| `<example>_data.py` / `_data_loader.py` | Dataset download + preprocessing |
| `<example>_train.py` + `_train.sh` | Training pipeline; the `.sh` runs it in one shot |
| `<example>_eval.py` + `_eval.sh` | Accuracy for float, quantized (QAT) and Akida models |
| `<example>_benchmark.py` | Latency and power measurement on real hardware |
| `*_notebook_training.ipynb` | Notebook walkthrough of training |
| `*_notebook_evaluation.ipynb` | Notebook walkthrough of evaluation |
| `*_notebook_benchmark.ipynb` | Notebook for accuracy + on-device benchmarking |
| `colab_setup.py` | One-shot environment setup for running the notebooks in Colab |
| `pretrained_models/` | Committed weights (via Git LFS) |
| `data/`, `models/` | Populated at runtime; not committed (git-ignored) |
| `docs/` | Benchmark plots, dataset mosaics, `metrics.json`, and the README template |
| `README.md` | Generated from `docs/README.md.template` + `docs/metrics.json` |

**Two ways to run every example**
- **Scripts** (`.py` / `.sh`) — reproducible command-line runs; the `_train.sh` /
  `_eval.sh` wrappers run the whole pipeline in one command.
- **Notebooks** (`*_notebook_*.ipynb`) — the same steps, interactive, each with an
  *Open in Colab* badge. On-device power benchmarks read from a physical board and
  won't run in Colab.

**Model weight formats**
- `.h5` — full-precision (float) Keras model
- `_qat.h5` — quantization-aware trained model
- `_qat.fbz` — converted, Akida-ready model

> Not every example has every file. Eval-only examples (e.g. `imagenet_akidanet`)
> ship evaluation and benchmark files but no training stage. A few carry extras like
> `configs/` or task-specific preprocessing.

</details>

---

## Repository Structure

```
brainchip_devhub/
├── akida1/
│   ├── model_zoo/        # Self-contained training, conversion & evaluation scripts
│   └── notebooks/       # Pedagogic notebooks on key concepts
├── akida2/
│   ├── model_zoo/
│   └── notebooks/
├── akida_pico/
│   ├── model_zoo/
│   └── notebooks/
├── deployment/          # Hardware deployment and benchmarking
│   ├── akida1/
│   ├── akida2/
│   └── akida_pico/
└── concepts/            # Cross-platform guides: how Akida works, optimisation strategies
```

Each platform's [`model_zoo/`](https://github.com/Brainchip-Inc/brainchip_devhub/tree/main/akida1/model_zoo) directory is intentionally self-contained — model definition, training, conversion, and evaluation for each example live together in a single script or small group of related files. This is a deliberate contrast to `akida_models`, which is structured as a reusable library; here, readability and reproducibility take priority.

---

## Community and Support

Hit a problem reproducing an example, or anything else in this repository? [Open an issue](../../issues) and say what you ran, what happened, and what hardware you're on.

- [Sign up for the BrainChip Developer Hub](https://developer.brainchip.com/signup/) for tools, the model zoo and Akida Cloud
- [Join the BrainChip Discord](https://discord.com/invite/9bmd9g52vn) for discussion and community help
- [Read the documentation](https://doc.brainchipinc.com) for MetaTF and the Akida platform
- [Subscribe to the newsletter](https://brainchip.com/newsletter/) for releases and announcements
- [Get in touch with sales](https://brainchip.com/contact/) to talk about a deployment
- Follow BrainChip on [LinkedIn](https://www.linkedin.com/company/7792006) and [X](https://x.com/BrainChip_inc)

## License

Apache 2.0 — see [LICENSE](LICENSE).
