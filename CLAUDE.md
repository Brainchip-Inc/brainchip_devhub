# CLAUDE.md

Guidance for Claude (and humans) working in `brainchip_devhub`. Keep this file lean: rules
and pointers, not tutorials. Area-specific detail lives in nested `CLAUDE.md` files, skills
and commands.

## Purpose and audience

Practical, runnable examples for developing and deploying models on BrainChip Akida
hardware. It complements the official docs (doc.brainchipinc.com); it does not replace them.

Readers are ML engineers who already know Keras/TensorFlow but are new to Akida. They want
something they can run, numbers they can trust, and the Akida-specific "why" explained.
The repo is **public**, so everything committed (issues and ROADMAP.md included) is
customer-facing.

## Structure

```
README.md                  landing page (hand-written)
ROADMAP.md                 strategy: themes, phases, non-goals (owner-maintained)
akida1/                    Akida 1 (AKD1000 / AKD1500)
  README.md                GENERATED from docs/README.md.template + every example's zoo_card.json/metrics.json
  update_readme.py
  model_zoo/<example>/     one self-contained example per folder
akida2/model_zoo/<example> Akida 2 (AKD2500; benchmarked on the v2 FPGA today)
brainchip_utils/           shared hardware benchmark + plotting helpers (pip install -e .)
test/                      CI model tests (pytest), discover_models.py
.github/workflows/         ci.yml (float, GitHub-hosted), hardware.yml (self-hosted AKD1500 + v2 FPGA)
.claude/commands/          model zoo generators: /model-zoo-example, /model-zoo-example-akida2, /model-zoo-eval-example
.claude/skills/            repeatable content workflows (review-content, ...)
```

The anatomy of an example (`<example>_<role>.py` naming, notebooks, `docs/`,
`pretrained_models/`) is documented once, in README.md § "Anatomy of a model_zoo example".
Follow it; the canonical references are `akida1/model_zoo/vww` (training),
`akida1/model_zoo/imagenet_akidanet` (eval-only) and `akida2/model_zoo/vww` (Akida 2).

## Rules that are easy to get wrong

- **Generated READMEs.** Each example's `README.md` is generated from
  `docs/README.md.template` + `docs/metrics.json` by its `update_readme.py`, which also
  regenerates `akida1/README.md`. Never hand-edit a generated README; edit the template and
  regenerate. Templates use `str.format_map`: double every literal brace (`{{ }}`), and
  never put a dot in a metrics key (`a50_224_`, not `a0.5_224_`).
- **Numbers.** Every accuracy, latency, power or energy figure must come from
  `metrics.json` (written by `--save-metrics`) or a cited source. Never type, round or
  estimate a benchmark number. Hardware numbers are AKD1500 unless stated otherwise.
- **Akida 1 vs 2.** The default cnn2snn context is v2; v1 code must run under
  `set_akida_version(AkidaVersion.v1)`. Do not copy v1 architecture/quantization prose into
  a v2 example. `.claude/commands/model-zoo-eval-example.md` has the verified comparison.
- **Weights and data.** `.h5`/`.fbz` go through Git LFS (`.gitattributes`). Only
  `pretrained_models/` is committed; `data/` and `models/` are runtime-only and git-ignored.
- **Dataset licensing.** State the licence and citation in the README. Never commit
  imagery from datasets that can't be redistributed (ImageNet etc.).
- **Toolchain pins.** `pyproject.toml` is the source of truth. The version badges in
  `README.md` and `akida1/docs/README.md.template` are hard-coded and must move with it.
- **No machine-specific paths.** No `/mnt/...` or `/home/...` in committed code or docs.
- New `.py` files in `model_zoo/` start with `# Copyright <year> Brainchip Holdings Ltd.  Apache 2.0 License`.

## Akida knowledge base

How Akida hardware and its toolchain behave is recorded in `.claude/knowledge/akida.md`,
imported below. Check claims about hardware behaviour against it. When you're corrected
about Akida, add an entry (platform, source, confidence) in the same PR as the fix.

@.claude/knowledge/akida.md

## Style

- Second person, direct, practical. Explain the Akida-specific *why*; skip generic ML
  explanations the reader already knows.
- Notebooks run from data to deployed model (training may be skipped only where it's
  prohibitively long), but the code cells shown are mostly the Akida-specific steps.
  Import standard ML pieces (data, model definition, training loop) from the example's
  scripts rather than inlining them. Those scripts also back the `.sh` one-command
  reproduction path, so put logic in the scripts, not the notebook.
- Prose spelling leans British (normalisation, optimisation); keep `quantization`, which
  matches the tool and API names.
- Claims about efficiency need a number from this repo behind them. No unsupported
  superlatives.
- Planned or unfinished content is marked `📝 Planned` / `🚧 In review` / **COMING SOON**,
  never left as a dead link.
- Link to the top-level README for requirements and LFS setup instead of repeating them.

## Commands

```bash
pip install -v -e .                                   # Python 3.10–3.12, pinned toolchain
python akida1/model_zoo/<example>/update_readme.py    # regenerate example + akida1 READMEs
python akida1/update_readme.py                        # regenerate the akida1 landing page only
pytest test/test_hardware_utils.py                    # unit tests, no hardware
pytest test/test_models.py -m "not hardware" --models "<repo-relative model paths>"
python test/discover_models.py --all                  # list the models CI will test
python .claude/skills/review-content/check_content.py  # README drift, links, pins, LFS, paths (read-only)
```

There is no site build: GitHub renders the markdown. Markdown lint and link checking are
not set up yet (#67).

## What Claude can and can't do here

Claude can develop an example end to end: code, templates, notebooks, landing pages, CI,
and **GPU training runs** that produce `pretrained_models/` weights and `metrics.json`.
Training is Claude's to run unless the issue says otherwise. When running it:

- check the GPU is free (`nvidia-smi`) before launching, and never stop jobs you didn't start;
- run long jobs in the background and log to a file. Record the exact command, epochs,
  wall-clock time and final metrics in the PR;
- pass dataset locations with the scripts' `--data` / `-d` flags, never hard-coded paths;
- commit weights only through Git LFS, and only the final `pretrained_models/` artifacts.

**Hardware benchmarks** need a physical AKD1500 (with I²C power sensing) or the v2 FPGA.
Claude can run them when remote access to the benchmark machines is configured in a
`CLAUDE.local.md` (local-only, git-ignored). Otherwise they need a human.

These steps **need a human** and must be labelled `needs-human` on the issue:

- hardware benchmarks, when remote hardware access isn't configured;
- dataset licensing decisions and anything touching product positioning or roadmap;
- any step the issue explicitly reserves for a human (e.g. a training run the owner
  wants to do themselves).

When a task mixes both, split it: the human-only step becomes its own sub-issue and the
writing step waits for its outputs (`metrics.json`, weights, plots).

## Definition of done (content PRs)

1. Follows the example anatomy and naming; scripts and notebooks run end to end, or the
   steps that weren't run are listed in the PR.
2. READMEs regenerated and in sync (rerunning `update_readme.py` leaves no diff).
3. Every number traces to `metrics.json` or a citation; hardware and mapping are named.
4. Dataset licence and citation present; no non-redistributable data committed.
5. New example linked from the landing pages (`README.md` table, and `zoo_card.json` for
   Akida 1).
6. CI green (`models-float`, plus `models-hardware` for new or changed weights).
7. `/review-content` run on the branch, with no blocking findings left open.
8. The PR links its issue and ticks the issue's acceptance criteria.

## Planning and work tracking

- Strategy: `ROADMAP.md`. Claude may propose edits by PR but never merges them.
- Execution: GitHub Issues and the Projects board. Labels and their meaning are in
  `.github/labels.yml`. Only pick up issues labelled `agent-ready`. If an issue is
  ambiguous, comment with questions and swap the label to `needs-human`. Don't guess.
- Issue templates in `.github/ISSUE_TEMPLATE/` require acceptance criteria and source
  material; treat those as the spec.
