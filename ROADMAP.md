# Roadmap

<!--
OWNER: Douglas McLelland. Claude may propose changes by PR; only the owner merges them.
This repo is PUBLIC: write nothing here you wouldn't say to a customer (no dates promised
to named accounts, no unreleased silicon specs, no internal-only names).
Task-level status does NOT live here. It lives on the GitHub Projects board. Each theme
links to its epic issue(s); the board tracks everything below that.
-->

_Last reviewed: 2026-10-08_

## Why this repo exists

To help developers prepare models for Akida hardware and deploy them. Full-length examples
take a model all the way from data to results measured on silicon, to demystify each step
of that process. Each one is also built to be a starting point for a developer's own
project.

## Principles

- **Runnable over exhaustive.** Every example reproduces end to end; depth lives in the
  official docs.
- **Measured on silicon.** Published numbers come from real Akida hardware and are
  regenerated from code, never typed in.
- **Self-contained examples.** Readability and reproducibility over code reuse;
  `akida_models` is the library.
- **Built to be adapted.** An example can be copied out as a single folder and pointed at
  the developer's own data. The Akida-specific choices (architecture constraints,
  quantization settings, input scaling) are explained where they're made, so a developer
  knows what's safe to change.
- **Notebooks show the Akida part; scripts carry the rest.** A notebook takes the reader
  from data to a deployed model, except where training is prohibitively long. Its visible
  code is mostly the Akida-specific steps, to show how few and simple they are. Standard ML
  pieces (data loading, model definition, training loops) are imported from the example's
  scripts to keep the notebook uncluttered. The same scripts, chained by the `.sh` drivers,
  give a one-command path to reproducing the results.

## Themes

Each theme has an outcome (what's true for the reader when it's done) and one or more epics.

| Theme | Outcome | Epic(s) | Priority |
| --- | --- | --- | --- |
| Akida 1 model zoo | Examples covering the main edge task families (vision, audio, time series, detection) on AKD1500, each one a starting point for the developer's own project | #35 | P2 |
| Akida 2 model zoo: parity | Every Akida 1 example has an Akida 2 equivalent. First the same model through the Akida 2 pipeline (quantizeml, 8-bit quantization without QAT), then a model optimised for Akida 2 | #36 | P1 |
| Akida 2 model zoo: beyond Akida 1 | Examples of what only Akida 2 can run: spatiotemporal models, models with upsampling (detection, segmentation and beyond), skip connections, non-ReLU activations. Includes PyTorch-based workflows, not just Keras | #37 | P2 |
| Akida Pico | Akida Pico examples in the repo, following the same structure and principles, starting with the examples already developed internally | #38 | P2 |
| Model improvement | Existing examples get better over time (accuracy, sparsity, latency/energy), with every change measured on hardware and reflected in the model cards | #39 | P2 |
| Tutorials | A reader understands *why* models are built the way they are for each platform. Platform-specific where the topic is unique to one platform, shared where it isn't | #40 | P2 |
| Deployment | A developer can take a converted model to a running embedded / MCU system, with a gentle first route (Arduino + Python) for developers new to embedded work | #41 | P3 |
| Agent workflow | Most content work is specified as issues and executed by Claude, with human review | #42 | P1 |

## Phases

<!-- Now = committed and in flight; Next = committed, not started; Later = intended, not
committed. Priority (P1–P3) orders themes; Now items are committed whatever their theme's
priority. -->

### Now

- Akida 1 model zoo: object detection (YOLOv2 / Pascal VOC).
- Akida 2 model zoo, parity: PlantVillage, Speech Commands.
- Model improvement: a sparse variant, trained with activity regularization, alongside the
  existing model in the Akida 1 examples (VWW, PlantVillage, ECG arrhythmia).
- Tutorials: Developing sparse models (Akida 1).
- Agent workflow: repo context for Claude, issue templates and labels, the Claude Code
  GitHub Action, and CI guardrails.

### Next

- Akida 2 model zoo, beyond Akida 1: segmentation, eye tracking, and MobileNet v1 ported
  from timm (the first PyTorch-based example).
- Model improvement: HwPr mapping benchmarked on every Akida 1 example.
- Tutorials: How Akida differs, Sparsity in Akida hardware, My first Akida workflow.
- Akida Pico: integrate the examples already developed internally.
- Deployment: an Arduino + Python route on the BrainBoard 1500.
- Agent workflow: a weekly routine that proposes issues and posts status; the remaining
  content skills.

### Later

- Deployment: further embedded / MCU examples.
- Tutorials: mapping modes; quantization with cnn2snn and with quantizeml; input scaling and
  the input layer; transfer learning (classification and object detection); 1D-conv
  time-series models (Akida 1 and 2); TENNs-B and TENNs-R.

## Non-goals

- Replacing or duplicating the official documentation at doc.brainchipinc.com.
- Being a reusable library or API (`akida_models` is that).
- Publishing simulated or estimated performance numbers as hardware results.

## Open questions

<!-- Decisions the owner still needs to make; Claude may add questions here by PR. -->

- How to organise tutorials that are cross-platform in concept: a top-level `tutorials/`,
  per-platform folders with shared pages, or something else. (`concepts/` in the README is
  to be renamed `tutorials/`.)
- Where pretrained weights are hosted: Git LFS (today), Hugging Face, or both. Hosting on
  Hugging Face would change how examples fetch weights and how they're pinned to a
  toolchain version.

## Change log

<!-- YYYY-MM-DD: what changed in strategy and why (one line each). -->

- 2026-10-07: First version. Concepts merged into Tutorials; Akida 2 split into parity and
  beyond-Akida-1 themes.
- 2026-10-08: Added the Model improvement theme; brought in items from the internal
  backlog (tutorial ideas, MobileNet v1 via PyTorch, HwPr, weight hosting question).
