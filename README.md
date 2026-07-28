<p align="center">
  <img src="docs/assets/0.-BC-dev-hub-LOGO-flicker.svg" alt="BrainChip Dev Hub" width="260"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License: Apache 2.0"/>
  <img src="https://img.shields.io/badge/python-3.10%20%E2%80%93%203.12-blue.svg" alt="Python 3.10 to 3.12"/>
  <img src="https://img.shields.io/badge/akida__models-1.14.0-orange.svg" alt="akida_models 1.14.0"/>
</p>

# BrainChip Developer Hub

<p align="center">
  <a href="https://doc.brainchipinc.com">Docs</a> ·
  <a href="#get-akida-hardware">Hardware</a> ·
  <a href="#getting-started">Getting Started</a> ·
  <a href="#platform-overview">Examples</a> ·
  <a href="#community-and-support">Community</a>
</p>

Resources for developing and deploying AI models on BrainChip Akida neuromorphic processors — training, conversion, evaluation, deployment, and benchmarking.

> This repository complements the [official BrainChip documentation](https://doc.brainchipinc.com). It focuses on practical, runnable examples and insider knowledge for getting the best out of Akida hardware.

---

## What do you want to do?

| Goal | Where to go |
| --- | --- |
| Train, convert, and evaluate a model | [Akida 1](akida1) · Akida 2 (**COMING SOON**) · Akida Pico (**COMING SOON**) |
| Deploy to hardware and benchmark | Deployment (**COMING SOON**) |
| Understand how Akida works | Concepts (**COMING SOON**) |
| New to Akida — not sure where to start | [Getting Started](#getting-started) |

---

## Platform overview

Akida 1 examples are available today; Akida 2 models/examples can be found [here](https://doc.brainchipinc.com/model_zoo_performance.html#akida-2-0-models) in the official docs, and Akida Pico content for this repo is on the way.

| | Akida 1 | Akida 2 | Akida Pico |
| --- | --- | --- | --- |
| **Chip** | AKD1500 | AKD2500 | — |
| **Typical use cases** | Image classification, keyword spotting, object detection | Larger models, higher accuracy targets | Always-on sensing, edge inference |
| **Examples in this repo** | [Image Classification](akida1/model_zoo/plant_village) · [Object Detection](akida1/model_zoo/vww) | 🔜 Coming soon | 🔜 Coming soon |

---

## Why Akida?

Akida is BrainChip's neuromorphic processor — it processes data event by event instead of running dense computation on every input, so it only spends power and cycles on activity that's actually there. In practice, that means real models running at a fraction of the latency and power of conventional hardware.

The two examples in this repo make that concrete on real AKD1500 silicon: [Image Classification (PlantVillage)](akida1/model_zoo/plant_village) reaches 99.43% accuracy, and [Object Detection ()](akida1/model_zoo/vww) runs inference in as little as 3.3 ms at 0.5 mJ per inference.

<p align="center">
  <img src="akida1/model_zoo/vww/docs/ref_benchmark_results_full.png" alt="AKD1500 power measurements during inference" width="600"/>
</p>

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

1. **Install the Akida toolkit** — see the [official installation guide](https://doc.brainchipinc.com).
2. **Pick your platform** — use the table above to find the best fit for your use case. Akida 1 is recommended for now.
3. **Run an example** — each platform directory has self-contained examples you can run immediately.

---

### Requirements

- Python versions: 3.10 to 3.12

We recommend using your preference of docker or a virtual environment such as `venv` or `conda`.
For example, to create and activate an appropriate virtual environment with `conda`:

```
conda create -n brainchip_devhub_env python=3.12 -y
conda activate brainchip_devhub_env
```

With your container or virtual environment active, all further requirements along with utilities
local to this repository should be installed by running the following at the top level of the
repository (you can check what packages will be installed in the `pyproject.toml` file):

```
pip install -v -e .
```

### Trained models

Pretrained model weights (`.h5`, `.fbz`, etc.) are stored directly in this repository, tracked with [Git LFS](https://git-lfs.com/) rather than regular git. If you cloned the repo without LFS support, these files will show up as small text pointers instead of real weights.

If `git-lfs` isn't installed on your machine yet, see the official installation instructions for your platform. On a linux machine, one option is

```
sudo apt install git-lfs
```

With git-lfs available, to pull the actual model files:

```
git lfs install   # one-time setup per machine
git lfs pull       # fetch the real model files for this clone
```

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

- **Bugs and feature requests** — open a [GitHub Issue](../../issues).
- **Reference material and official guides** — see the [BrainChip documentation](https://doc.brainchipinc.com).
- **Discussion and community help** — join the [BrainChip Discord](https://discord.com/invite/9bmd9g52vn).

## License

Apache 2.0 — see [LICENSE](LICENSE).
