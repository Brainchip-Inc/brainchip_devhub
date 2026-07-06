# BrainChip Developer Hub

Resources for developing and deploying AI models on BrainChip Akida neuromorphic processors — training, conversion, evaluation, deployment, and benchmarking.

> This repository complements the [official BrainChip documentation](https://docs.brainchipinc.com). It focuses on practical, runnable examples and insider knowledge for getting the best out of Akida hardware.

---

## What do you want to do?

| Goal | Where to go |
|------|-------------|
| Train, convert, and evaluate a model | [Akida 1](akida1/) · [Akida 2](akida2/) · [Akida Pico](akida_pico/) |
| Deploy to hardware and benchmark | [deployment/](deployment/) |
| Understand how Akida works | [concepts/](concepts/) |
| New to Akida — not sure where to start | [Getting Started](#getting-started) |

---

## Platform Overview

| | Akida 1 | Akida 2 | Akida Pico |
|---|---|---|---|
| **Chip** | AKD1500 | AKD2500 | — |
| **Key strengths** | Broad ecosystem support, proven in deployment | Higher capacity, expanded model support | Ultra-low power, embedded / IoT |
| **Typical use cases** | Image classification, keyword spotting, object detection | Larger models, higher accuracy targets | Always-on sensing, edge inference |
| **Form factor** | PCIe / USB | PCIe / USB | Compact module |

---

## Getting Started

1. **Install the Akida toolkit** — see the [official installation guide](https://docs.brainchipinc.com).
2. **Pick your platform** — use the table above, or start with Akida 2 if you have access to it.
3. **Run an example** — each platform directory has self-contained examples you can run immediately.


---

### Trained models

Pretrained model weights (`.h5`, `.fbz`, etc.) are stored directly in this repository, tracked with [Git LFS](https://git-lfs.com/) rather than regular git. If you cloned the repo without LFS support, these files will show up as small text pointers instead of real weights.

To pull the actual model files:

```bash
git lfs install   # one-time setup per machine
git lfs pull       # fetch the real model files for this clone
```

If `git-lfs` isn't installed on your machine yet, see the [official installation instructions](https://git-lfs.com/) for your platform.

---

## Repository Structure

```
brainchip_devhub/
├── akida1/
│   ├── examples/        # Self-contained training, conversion & evaluation scripts
│   └── notebooks/       # Pedagogic notebooks on key concepts
├── akida2/
│   ├── examples/
│   └── notebooks/
├── akida_pico/
│   ├── examples/
│   └── notebooks/
├── deployment/          # Hardware deployment and benchmarking
│   ├── akida1/
│   ├── akida2/
│   └── akida_pico/
└── concepts/            # Cross-platform guides: how Akida works, optimisation strategies
```

The `examples/` in each platform directory are intentionally self-contained — model definition, training, conversion, and evaluation live together in a single script or small group of related files. This is a deliberate contrast to `akida_models`, which is structured as a reusable library; here, readability and reproducibility take priority.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
