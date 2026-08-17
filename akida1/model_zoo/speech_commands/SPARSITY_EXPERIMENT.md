# Speech Commands Sparsity Regularization Experiment

Tracking doc for Asana task **"Akida 1: Sparsify VWW model"** — applying the same
investigation done on VWW and PlantVillage (see
[`../vww/SPARSITY_EXPERIMENT.md`](../vww/SPARSITY_EXPERIMENT.md) and
[`../plant_village/SPARSITY_EXPERIMENT.md`](../plant_village/SPARSITY_EXPERIMENT.md))
to the Speech Commands keyword-spotting example, to check whether the findings
there (Hoyer-Square, especially the element-count-normalized variant, beating
L1L2) generalize to a third model/dataset/modality (audio, not vision).

Branch: `akida1-sparsify-speech-commands`

## Important difference from VWW/PlantVillage: this example already ships with tuned Hoyer-Square

Unlike VWW and PlantVillage, which started from *no* activity regularization,
this example's `configs/training_cfg.yml` already applies raw Hoyer-Square
activity regularization by default, with strengths that look HPO-tuned (odd
full-precision values, e.g. `weight_decay: 3.176859846184903e-05`):

| config key | value |
|---|---:|
| `activity_reg_hoyer_strength` (float stage) | `2.6244807242474337e-05` |
| `activity_reg_hoyer_strength_qat` (QAT stage) | `0.0015643915309023002` |

The published `docs/metrics.json` numbers (95.69% / 92.95% / 93.06% float/QAT/
Akida accuracy, 73.19% sparsity) are **with this regularization already
applied** -- there is no existing unregularized baseline to compare against.
Establishing one is Pass 0 below.

Also note the production config uses **different** strengths for the float
and QAT stages (float:QAT ratio ≈ 1:60), whereas this sweep (like VWW's and
PlantVillage's) applies **the same** reg value to both stages for
comparability across the three experiments. This sweep's results are
therefore not directly comparable to the production config's tuned point --
see Open Questions.

## Setup

- Model: DS-CNN (`build_ds_cnn` in `speech_commands_model.py`), 22,668 params
  -- by far the smallest of the three models tested (VWW: 226,906;
  PlantVillage: 1,156,054). 5 ReLU layers, 4 at 8,000 elements (spatial
  25x5x64 before the final GAP block) and 1 at 64 elements (after GAP) -- a
  ~125x per-layer size spread, between VWW's ~186x and PlantVillage's ~792x.
- Regularizer: `regularizers_custom.py` already had `HoyerSquare` (raw). Added
  `normalize` support to it (same formula as VWW/PlantVillage: divide by
  `tf.size(x)`), and added a new `L1L2Activity` regularizer (this project had
  no L1L2 activity-regularization option before). Selected via a new
  `reg_type` config key (`l1l2` / `hoyer_square` / `hoyer_square_norm`,
  defaulting to `hoyer_square` to preserve old config behavior when the key
  is absent) in `speech_commands_train.py`.
- Pipeline per data point: float train -> float eval -> `cnn2snn quantize`
  (8/4/4-bit) -> QAT fine-tune -> QAT eval -> `cnn2snn convert` -> Akida eval ->
  sparsity measurement. Same stage sequence as `speech_commands_train.sh`.
- Sweep script: [`speech_commands_sparsity_sweep.py`](speech_commands_sparsity_sweep.py).
  Unlike VWW/PlantVillage's sweep (which pass `-reg`/`--reg-type` as CLI
  flags), this project is config-driven, so the sweep generates a temp YAML
  config per point -- copying every setting from `--base-config`
  (`configs/training_cfg.yml`: filters, dropout, augmentation, epochs, LR,
  etc.) and overriding only `activity_reg_hoyer_strength`,
  `activity_reg_hoyer_strength_qat` (both set to the same swept value), and
  `reg_type`. Sparsity is measured with
  [`speech_commands_sparsity_only.py`](speech_commands_sparsity_only.py)
  (software backend, via `akida_models.sparsity.compute_sparsity`) rather
  than `speech_commands_benchmark.py`, because **this machine has no physical
  Akida hardware attached** -- same caveat as VWW/PlantVillage.
- Environment: same dedicated conda env `vww-sparsify` used for the other two
  sweeps. Trained on GPU 1 of the shared training box
  (`CUDA_VISIBLE_DEVICES=1`).
- Data: `speech_commands` tfds dataset (~2.4GB, 12 classes: 10 keywords +
  silence + unknown), auto-downloaded via the existing
  `speech_commands_data_loader.py`.
- Epoch budget: kept at the base config's 30 float / 25 QAT epochs (much
  longer than PlantVillage's 10/2) -- this is what the existing tuned config
  uses, and unlike PlantVillage this project has no separate "known good,
  shorter schedule" to fall back to.

## Results

### Pass 1 — true baseline + coarse L1L2 sweep (30 float epochs, 25 QAT epochs)

Same three reg values as VWW/PlantVillage's Pass 1 (`0`, `1e-5`, `1e-3`).
`reg=0` here is the first genuine unregularized baseline for this example.

| reg | float acc | QAT acc | Akida acc | sparsity |
|-----:|---:|---:|---:|---:|
| 0 (baseline) | 96.03% | 95.66% | 95.73% | 59.4% |
| 1e-5 | 96.11% | 94.61% | 94.44% | 68.1% |
| 1e-3 | 62.15% | 62.15% | 62.15% | 79.7% |

**Findings:**
- **The unregularized baseline beats the published (regularized) numbers, as
  expected.** 95.73% Akida acc / 59.4% sparsity here vs. the published
  93.06% / 73.19% -- confirms the production config's tuned Hoyer-Square
  strengths do cost accuracy for sparsity, and gives a real reference point
  to measure this sweep's L1L2/Hoyer-Square results against (something
  VWW/PlantVillage didn't need, since they started unregularized).
- **`reg=1e-5` is a good trade here, unlike on VWW/PlantVillage.** Only
  ~1.3pt Akida accuracy cost (95.73% -> 94.44%) for +8.7pt sparsity (59.4% ->
  68.1%). On VWW this same value caused a QAT-stage collapse to 66.8%; on
  PlantVillage it fully collapsed training. This model's much smaller ReLU
  tensors (max 8,000 elements vs. VWW's 36,864 / PlantVillage's 401,408) mean
  the same L1L2 magnitude is proportionally far weaker here.
- **`reg=1e-3` fully collapses to a single-class floor** (62.15% at every
  stage, float through Akida -- unlike PlantVillage, conversion succeeded
  here). 62.15% is presumably this dataset's majority-class fraction (likely
  the catch-all "Unknown" class, which absorbs all non-keyword words).
- **The knee is somewhere between `1e-5` and `1e-3`** -- a much wider gap
  than VWW/PlantVillage needed to bracket (they used values an order of
  magnitude apart at most before finding the knee). Pass 2 samples this gap.

### Pass 2 — finer L1L2 sweep between 1e-5 and 1e-3 (30 float epochs, 25 QAT epochs)

| reg | float acc | QAT acc | Akida acc | sparsity |
|-----:|---:|---:|---:|---:|
| 5e-5 | 95.29% | 90.57% | 90.45% | 72.3% |
| 1e-4 | 95.09% | 88.66% | 88.71% | 73.6% |
| 3e-4 | 92.35% | 78.93% | 78.92% | 75.9% |

**Findings:**
- **`reg=1e-5` remains the clear best L1L2 point.** Every value past it
  trades a small amount more sparsity for a much larger accuracy cost:
  `5e-5` already reaches 72.3% sparsity (close to the published 73.19%) but
  at 90.45% accuracy, well below the published 93.06%.
- **Smooth, monotonic decline, not a sudden cliff** -- same shape as
  VWW/PlantVillage. QAT-stage drop from float grows steadily: `1e-5` ~1.5pt
  -> `5e-5` ~4.7pt -> `1e-4` ~6.4pt -> `3e-4` ~13.4pt -> `1e-3` (full
  collapse, float itself drops to the floor).

**L1L2 summary (all 5 points tested):**

| reg | Akida acc | sparsity |
|-----:|---:|---:|
| 0 (baseline) | 95.73% | 59.4% |
| **1e-5** | **94.44%** | **68.1%** |
| 5e-5 | 90.45% | 72.3% |
| 1e-4 | 88.71% | 73.6% |
| 3e-4 | 78.92% | 75.9% |
| 1e-3 | 62.15% (collapsed) | 79.7% |

`reg=1e-5` is L1L2's best point here -- notably, the *same* reg value that
won on both VWW and PlantVillage, despite this model being ~50x smaller than
VWW and ~50,000x smaller than PlantVillage by parameter count. Unlike the
other two models, though, this sweep's best L1L2 point (94.44% Akida acc /
68.1% sparsity) doesn't reach the published production sparsity (73.19%) --
getting there with L1L2 costs meaningfully more accuracy (90.45% at `5e-5`)
than the production Hoyer-Square config's 93.06%.

### Pass 3 — raw Hoyer-Square sweep (30 float epochs, 25 QAT epochs)

Same three reg values as VWW/PlantVillage's Hoyer-Square passes (`1e-7`,
`1e-6`, `1e-5`), for comparability.

| reg | float acc | QAT acc | Akida acc | sparsity |
|-----:|---:|---:|---:|---:|
| 1e-7 | 96.30% | 95.57% | 95.68% | 60.5% |
| 1e-6 | 96.24% | 95.76% | 95.86% | 61.3% |
| 1e-5 | 95.82% | 95.07% | 95.00% | 66.7% |

**Findings:**
- **No collapse anywhere in this range** -- consistent with VWW/PlantVillage,
  Hoyer-Square stays well-behaved.
- **But unlike VWW/PlantVillage, these values barely move sparsity at all.**
  Sparsity crawls from 60.5% to 66.7% across three orders of magnitude of
  `reg`, vs. VWW's jump from ~52% to ~69% and PlantVillage's ~54% to ~73%
  over the *same* three values. This model's raw Hoyer-Square penalty is
  bounded by its largest ReLU tensor (8,000 elements, vs. VWW's 36,864 and
  PlantVillage's 401,408) -- for the same `reg`, a smaller max tensor size
  means a smaller maximum possible penalty, so these values are
  proportionally far weaker here.
- **This matches the production config's own tuning.** The shipped
  `activity_reg_hoyer_strength` (float: `2.62e-5`) and especially
  `activity_reg_hoyer_strength_qat` (`1.56e-3`) are both well above this
  sweep's `1e-5` ceiling -- consistent with this model needing meaningfully
  larger raw Hoyer-Square strengths than VWW/PlantVillage to have a
  comparable effect. Extending the sweep range (Pass 3b) to actually find
  this model's knee, rather than assuming VWW/PlantVillage's range transfers.

### Pass 3b — extended raw Hoyer-Square sweep at larger magnitudes (30 float epochs, 25 QAT epochs)

| reg | float acc | QAT acc | Akida acc | sparsity |
|-----:|---:|---:|---:|---:|
| 1e-4 | 95.05% | 92.61% | 92.63% | 71.3% |
| 1e-3 | 62.15% (collapsed) | 62.15% | 62.15% | 81.9% |
| 1e-2 | 62.15% (collapsed) | 62.15% | 62.15% | 83.9% |

**Findings:**
- **`reg=1e-4` is by far the best raw Hoyer-Square point found** -- 71.3%
  sparsity for a ~3.1pt accuracy cost from baseline (95.73% -> 92.63%),
  closely approaching the published production numbers (73.19% / 93.06%)
  despite this sweep using the same reg value for both float and QAT stages
  rather than the production's differentiated per-stage tuning.
- **Raw Hoyer-Square does collapse eventually, just at ~100x the strength
  that collapsed L1L2.** `reg=1e-3` and `1e-2` both fully collapse to
  exactly the same floor (62.15%) L1L2 hit at its own `reg=1e-3` -- the same
  majority-class fraction, reached from a different regularizer. This
  refines the "Hoyer-Square is more stable" narrative from VWW/PlantVillage
  (where the tested range never reached collapse): it's not immune to
  collapse, it just needs proportionally more strength to get there, because
  its raw penalty is bounded by this model's much smaller max tensor size
  (8,000 elements).
- **L1L2's best point (`1e-5`: 94.44%/68.1%) and Hoyer's best point (`1e-4`:
  92.63%/71.3%) are a genuine tradeoff, not a clean win either way** --
  unlike VWW/PlantVillage, where Hoyer-Square's best point strictly beat
  L1L2's best point on both axes. Here, Hoyer trades ~1.8pt more accuracy
  cost for ~3.2pt more sparsity. Neither point has been confirmed as each
  regularizer's true knee yet (see Pass 3c).

### Pass 3c — finer raw Hoyer-Square sweep between 1e-4 and 1e-3 (30 float epochs, 25 QAT epochs)

| reg | float acc | QAT acc | Akida acc | sparsity |
|-----:|---:|---:|---:|---:|
| 2e-4 | 94.39% | 88.80% | 88.91% | 73.4% |
| 3e-4 | 94.41% | 89.21% | 89.16% | 73.7% |
| 5e-4 | 62.15% (collapsed) | 62.15% | 62.15% | 80.7% |

**Findings:**
- **`reg=1e-4` is confirmed as the true best raw Hoyer-Square point** --
  `2e-4` and `3e-4` both reach slightly more sparsity (73.4-73.7% vs. 71.3%)
  but at a much larger accuracy cost (88.9-89.2% vs. 92.63%). Once past
  `1e-4`, the exchange rate turns sharply unfavorable.
- **The actual cliff is between `3e-4` (healthy, 89.16%) and `5e-4`
  (collapsed, 62.15%)** -- much narrower and closer to `1e-4` than Pass 3b's
  wide `1e-4`-to-`1e-3` bracket suggested.

**Raw Hoyer-Square summary (all 9 points tested):**

| reg | Akida acc | sparsity |
|-----:|---:|---:|
| 0 (baseline) | 95.73% | 59.4% |
| 1e-7 | 95.68% | 60.5% |
| 1e-6 | 95.86% | 61.3% |
| 1e-5 | 95.00% | 66.7% |
| **1e-4** | **92.63%** | **71.3%** |
| 2e-4 | 88.91% | 73.4% |
| 3e-4 | 89.16% | 73.7% |
| 5e-4 | 62.15% (collapsed) | 80.7% |
| 1e-3 | 62.15% (collapsed) | 81.9% |
| 1e-2 | 62.15% (collapsed) | 83.9% |
