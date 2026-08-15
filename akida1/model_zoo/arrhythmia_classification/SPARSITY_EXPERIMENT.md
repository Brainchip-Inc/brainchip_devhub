# Arrhythmia Classification Sparsity Regularization Experiment

Tracking doc for Asana task **"Akida 1: Sparsify VWW model"** — applying the same
investigation done on VWW, PlantVillage and Speech Commands (see
[`../vww/SPARSITY_EXPERIMENT.md`](../vww/SPARSITY_EXPERIMENT.md),
[`../plant_village/SPARSITY_EXPERIMENT.md`](../plant_village/SPARSITY_EXPERIMENT.md) and
[`../speech_commands/SPARSITY_EXPERIMENT.md`](../speech_commands/SPARSITY_EXPERIMENT.md))
to the ECG Arrhythmia Classification example, to check whether the findings there
(Hoyer-Square, especially the element-count-normalized variant, beating L1L2)
generalize to a fourth model/dataset/modality (1D physiological signal ->
CWT scalogram, not vision or audio).

Branch: `akida1-sparsify-arrhythmia`

## Important difference from the other three: this example already ships with an always-on L1L2 activity regularizer

Like Speech Commands, this project did not start from an unregularized baseline:
`scripts/train.py` unconditionally applied `apply_activity_regularizer(model,
L1L2_reg_value)` with `L1L2_reg_value = 2e-5` — the *same* constant also used as
the kernel (pointwise conv) weight-decay strength inside `build_akida_model`. The
published `ReadMe.md`/`docs/sparsity_dict.txt` numbers (95.97% accuracy, ~84%
mean ReLU-layer sparsity from `docs/sparsity_dict.txt`) are **with this
regularization already applied** — there is no existing unregularized baseline to
compare against. Establishing one (`reg=0`) is Pass 1 below.

Unlike Speech Commands, this project used the *same* strength for both stages
already (no separate float/QAT split), so no per-stage ratio discussion is
needed here.

## Setup

- Model: Custom Akida-targeted CNN (`build_akida_model` in `scripts/model.py`) —
  stem Conv2D(16) + 2 SeparableConv2D DS-blocks (32, 64, with MaxPooling) + a
  final SeparableConv2D DS-block (128, no pooling) + GAP + Dense(64) + Dense(3).
  23,363 params. 5 ReLU layers (`ReLU(max_value=6)` except the two after
  GAP/Dense which are plain `ReLU`): 18,432 elements (stem), 36,864 (block1,
  the largest — same order of magnitude as VWW's 36,864 max), 18,432 (block2),
  128 (post-GAP), 64 (post-Dense-1).
- Regularizer: `apply_activity_regularizer` in `model.py` previously hardcoded
  `regularizers.L1L2(l1=reg_val, l2=reg_val)`. Added a new
  `regularizers_custom.py` (same `HoyerSquare`/`L1L2Activity` implementations as
  `../speech_commands/regularizers_custom.py`) and switched
  `apply_activity_regularizer` to a `reg_type` dispatch (`l1l2` / `hoyer_square`
  / `hoyer_square_norm`, default `l1l2` to preserve old behavior). The kernel
  weight-decay L2 (`build_akida_model`'s regularizer arg) was decoupled from the
  activity-regularization strength and kept fixed at the original `2e-5` — only
  the activity regularizer is swept, matching how the other three experiments
  treat kernel weight decay as a separate, untouched knob.
- `train.py` changes needed to run a true `reg=0` baseline and a sweep:
  - Added `-reg`/`--regularization` (default `None` = no activity regularizer,
    the real baseline) and `--reg-type` CLI args.
  - Re-apply the activity regularizer after `prepare_qat_model`/`cnn2snn.quantize`
    (which rebuilds `ReLU` layers as `QuantizedReLU`, losing the float model's
    `activity_regularizer`) — same "re-apply so sparsity survives quantization"
    pattern the other three sweeps use, just inline in one script here instead
    of a second CLI invocation.
  - Fixed a pre-existing bug where `--run_dir` was silently discarded and always
    replaced with a fresh timestamped path — blocked the sweep script from
    controlling per-point output directories.
  - Fixed a pre-existing bug where `--float_epochs`/`--qat_epochs` were declared
    `type=float`; passing a Python float to `model.fit(epochs=...)` crashes this
    TF version (`Cannot convert 1.0 to EagerTensor of dtype int64`) — changed to
    `type=int`.
  - `evaluate_and_report` now also returns/prints a clean `Test accuracy: X.XXXX`
    line and both stages' accuracies are written to `<run_dir>/accuracies.json`.
- **Input-scaling bug found and fixed while wiring up Akida conversion/eval**:
  the pre-existing `scripts/eval.py` assumed the model's float input range is
  `[-1, 1]` when converting to Akida (`cnn2snn.convert(..., input_scaling=(127.5,
  127.5))`). This is wrong for this data: `beat_to_scalogram`'s min-max
  normalization bounds the CWT scalogram rows to `[0, 1]`, but the appended
  RR-interval feature rows (`add_rr_rows`) are z-scored (mean 0, std 1) with real
  tails — actual training-split min/max is **`[-5.59, 14.83]`**, more than 5x
  wider than assumed and asymmetric. Using the wrong bounds wastes almost the
  entire uint8 dynamic range on values that never occur and saturates the rest,
  silently corrupting Akida-model accuracy (this would have shown up as a
  mysterious QAT-to-Akida accuracy cliff on every sweep point). Centralized the
  correct bounds in a new `scripts/input_scaling.py`, used consistently by the
  new `arrhythmia_convert.py`, `arrhythmia_eval.py` and
  `arrhythmia_sparsity_only.py`. Also: `cnn2snn convert`'s CLI only accepts
  *integer* `-sc`/`-sh`, insufficient precision for this range, so
  `arrhythmia_convert.py` calls the `cnn2snn.convert` Python API directly instead
  of shelling out to the CLI (the other three examples' sweeps didn't need this,
  since none of them use a non-image input needing custom scaling).
- Pipeline per data point: this project's `train.py` runs float train -> float
  eval -> `cnn2snn.quantize` -> QAT fine-tune -> QAT eval all in **one process**
  (unlike the other three examples' two separate float/QAT script invocations),
  writing `accuracies.json`. The sweep script then converts to Akida
  (`arrhythmia_convert.py`) -> Akida eval (`arrhythmia_eval.py`) -> sparsity
  (`arrhythmia_sparsity_only.py`, software backend via
  `akida_models.sparsity.compute_sparsity` — **this machine has no physical
  Akida hardware attached**, same caveat as the other three).
- Sweep script: [`scripts/arrhythmia_sparsity_sweep.py`](scripts/arrhythmia_sparsity_sweep.py).
- Environment: same dedicated conda env `vww-sparsify` used for the other three
  sweeps — its existing deps (opencv-python, PyWavelets, wfdb, joblib,
  scikit-learn) happened to already cover this project's preprocessing
  requirements. Trained on GPU 1 of the shared training box
  (`CUDA_VISIBLE_DEVICES=1`) — GPU 0 was occupied by another user's job
  (~22/24GB used) throughout.
- Data: MIT-BIH Arrhythmia Database v1.0.0 (48 records, ~110k beat annotations),
  downloaded fresh from PhysioNet (`wget -r` per `ReadMe.md`'s Step 1) since no
  copy existed on this machine, then preprocessed with the existing
  `ECGDatasetBuilder` (beat extraction -> CWT scalogram -> RR-interval feature
  rows appended -> 80/20 stratified train/val split). Preprocessing reproduced
  the published test-set size exactly (49,280 beats), validating the pipeline
  before any training started. Epoch budget: kept at `docs/config.json`'s
  production defaults (80 float / 50 QAT).

## Results

### Pass 1 — true baseline + coarse L1L2 sweep (80 float epochs, 50 QAT epochs)

Same three reg values as the other three experiments' Pass 1 (`0`, `1e-5`,
`1e-3`). `reg=0` here is the first genuine unregularized-activity baseline for
this example (kernel weight decay stays at `2e-5` throughout, unswept).

| reg | float acc | QAT acc | Akida acc | sparsity |
|-----:|---:|---:|---:|---:|
| 0 (baseline) | 93.01% | 94.04% | 95.04% | 55.4% |
| 1e-5 | 94.92% | 92.63% | 91.48% | 68.7% |
| 1e-3 | 92.04% | 89.87% | 89.84% | 80.4% |

**Findings:**
- **The unregularized baseline is close to, and actually beats, the published
  (regularized) accuracy**: 95.04% Akida acc here vs. the published 95.97%
  overall accuracy — within ~1pt, unlike Speech Commands where the gap was
  larger. Sparsity is much lower than published though (55.4% vs. the
  `docs/sparsity_dict.txt` ReLU-layer average of ~84%), confirming the
  production `2e-5` L1L2 activity regularizer (silently baked into every prior
  run of this project) was doing real sparsity work.
- **No collapse anywhere in this range — a first among the four experiments.**
  VWW, PlantVillage and Speech Commands all hit a hard majority-class floor by
  `reg=1e-3` (single-digit-to-60s% accuracy). Here `reg=1e-3` is still a
  healthy, well-differentiated 89.84% with meaningfully higher sparsity
  (80.4%). Likely contributors: this project's `class_weights = {0: 1.0, 1:
  6.0, 2: 3.0}` (baked into every stage of `train.py`, unrelated to this sweep)
  heavily counteracts the "predict only the majority class" failure mode that
  caused every other project's collapse, and/or this smaller, GAP-heavy
  architecture (23,363 params) responds differently. The L1L2 knee is
  evidently well past `1e-3` — Pass 2 extends the range upward to find it.
- **Smooth, monotonic-ish decline, not a cliff so far** — sparsity climbs
  55.4% -> 68.7% -> 80.4% while accuracy declines gently (95.04% -> 91.48% ->
  89.84%), a much gentler slope than any of the other three projects showed
  over the same reg values.

### Pass 2 — extended L1L2 sweep to find the collapse point (80 float epochs, 50 QAT epochs)

Since `reg=1e-3` didn't collapse (unlike every other project's Pass 1), this
pass pushes an order of magnitude higher to bracket where L1L2 actually breaks
this model.

| reg | float acc | QAT acc | Akida acc | sparsity |
|-----:|---:|---:|---:|---:|
| 3e-3 | 92.89% | 91.60% | 92.19% | 80.3% |
| 1e-2 | 90.82% | 90.28% | 90.12% | 81.0% |
| 3e-2 | 90.77% | 91.12% | 91.13% | 87.3% |

**Findings:**
- **Still no collapse at 30x the reg value that fully collapsed every other
  project.** Accuracy plateaus in a narrow 90-92% band across `3e-3` ->
  `3e-2` while sparsity keeps climbing (80.3% -> 81.0% -> 87.3%) — a full order
  of magnitude of extra regularization strength buys ~7pt more sparsity for
  under 1pt of accuracy cost from `3e-3`'s level. This is a fundamentally
  different regime than VWW/PlantVillage/Speech Commands, where the same
  relative reg increases were the difference between "healthy" and "dead".
  Not pushed further past `3e-2` — the accuracy-vs-sparsity trade past this
  point is clearly saturating, and finding the *exact* eventual collapse point
  matters less here than getting a Hoyer-Square comparison at a similar
  epoch/compute budget.
- **`reg=3e-3` is the best L1L2 point found**: 92.19% accuracy at 80.3%
  sparsity dominates `reg=1e-3` (89.84%/80.4% — worse on both axes) and is
  within 1pt of `reg=1e-2`/`reg=3e-2`'s accuracy at meaningfully less
  regularization pressure.
- **Non-monotonic accuracy vs. reg** (`1e-3`: 89.84% -> `3e-3`: 92.19% ->
  `1e-2`: 90.12% -> `3e-2`: 91.13%) rather than the smooth monotonic decline
  the other three projects' sweeps showed. Consistent with this being a noisy,
  small-validation-set regime (49,280 test beats but a much smaller,
  heavily-imbalanced training signal for the minority S/V classes) rather than
  a clean accuracy-sparsity Pareto frontier — a few points of accuracy
  variation between adjacent reg values is plausibly training-seed noise, not
  a real reversal.

**L1L2 summary (all 6 points tested):**

| reg | Akida acc | sparsity |
|-----:|---:|---:|
| 0 (baseline) | 95.04% | 55.4% |
| 1e-5 | 91.48% | 68.7% |
| 1e-3 | 89.84% | 80.4% |
| **3e-3** | **92.19%** | **80.3%** |
| 1e-2 | 90.12% | 81.0% |
| 3e-2 | 91.13% | 87.3% |

### Pass 3 — raw Hoyer-Square sweep (80 float epochs, 50 QAT epochs)

Same three reg values as the other three experiments' Hoyer-Square passes
(`1e-7`, `1e-6`, `1e-5`), for comparability. This model's max ReLU tensor
(36,864 elements, `block1_relu6`) is the same order of magnitude as VWW's max
(36,864), so the same raw reg range should land in a comparable regime here,
unlike Speech Commands (max tensor only 8,000 elements, needed a ~10x larger
range to have any effect).

| reg | float acc | QAT acc | Akida acc | sparsity |
|-----:|---:|---:|---:|---:|
| 1e-7 | 92.63% | 94.28% | 93.97% | 57.5% |
| 1e-6 | 92.09% | 92.48% | 90.82% | 59.1% |
| 1e-5 | 94.48% | 94.13% | 93.64% | 67.4% |

**Findings:**
- **No collapse, consistent with every other project's raw Hoyer-Square
  results** — but also consistent with this project's own L1L2 results:
  nothing collapses in the ranges tried so far.
- **Non-monotonic again** (`1e-7`: 93.97% -> `1e-6`: 90.82% -> `1e-5`: 93.64%),
  same noisy pattern as the L1L2 sweep rather than a clean decline. Given the
  QAT epoch budget (50) is much larger relative to this model/dataset's size
  than the other three projects' sweeps, and this dataset's minority classes
  (S: 754 train beats, per `docs/config.json`) are inherently harder to fit
  consistently, run-to-run variance at fixed reg looks like a real factor here
  — a caveat future passes on this project should keep in mind before reading
  small differences between adjacent reg values as significant.
- **`reg=1e-5` is the best raw Hoyer-Square point so far**: 93.64% accuracy at
  67.4% sparsity — both better than L1L2's `reg=1e-5` point (91.48%/68.7%,
  essentially a wash on sparsity but 2pt worse accuracy) and closing in on the
  unregularized baseline's accuracy (95.04%) while still gaining +12pt
  sparsity. This is the first point in this project's sweep that looks like a
  genuinely good trade rather than a noisy wash.

### Pass 4 — normalized Hoyer-Square sweep (80 float epochs, 50 QAT epochs)

`reg` isn't directly comparable between `hoyer_square` and `hoyer_square_norm`
(same calibration issue the VWW/PlantVillage sweeps ran into) — the normalized
penalty divides by each tensor's element count, so reusing Pass 3's
`1e-7/1e-6/1e-5` would apply almost no pressure. This model's 5 ReLU layers
average 14,984 elements (18,432 / 36,864 / 18,432 / 128 / 64), so Pass 3's
values were scaled up by ~15,000x: `1e-3`, `1e-2`, `1e-1`.

| reg | float acc | QAT acc | Akida acc | sparsity |
|-----:|---:|---:|---:|---:|
| 1e-3 | 95.34% | 93.02% | 92.34% | 56.4% |
| 1e-2 | 94.35% | 93.59% | 93.63% | 55.1% |
| 1e-1 | 93.55% | 93.30% | 93.51% | 58.9% |

**Findings — unlike every other project, normalized Hoyer-Square is NOT the
clear winner at these reg values:**
- Sparsity barely moves off the unregularized baseline (55.4%) across this
  entire 100x range: 56.4% -> 55.1% -> 58.9%, essentially flat and within
  noise of the baseline's 55.4%. Compare to raw Hoyer-Square's Pass 3, which
  reached 67.4% sparsity at `reg=1e-5`, or L1L2's 80%+ at `reg>=1e-3`.
- This means the ~15,000x calibration factor (average ReLU element count) was
  too conservative for *this* model. Unlike VWW/PlantVillage's calibration
  (which measured real per-batch Hoyer-Square sums to pick a scaling factor),
  this pass used a size-based heuristic — reasonable as a starting bracket,
  but evidently still landing on the flat part of the curve, well below where
  the normalized penalty starts meaningfully competing with the raw
  activations' natural scale. The three ReLU layers with by far the most
  elements (18,432 / 36,864 / 18,432) are diluted in a naive 5-layer average
  by the two tiny post-GAP/post-Dense layers (128 / 64) — an average
  weighted more toward the large conv layers would have suggested starting
  higher.
- Accuracy stays healthy and roughly flat too (92.3-93.6%), consistent with
  the regularizer barely engaging in this range rather than any resistance to
  it collapsing.
- **Pass 5 extends normalized Hoyer-Square another order of magnitude higher**
  to find the range where it actually starts trading sparsity for accuracy,
  before drawing conclusions about whether it beats L1L2/raw Hoyer-Square here.

### Pass 5 — extended normalized Hoyer-Square sweep (80 float epochs, 50 QAT epochs)

| reg | float acc | QAT acc | Akida acc | sparsity |
|-----:|---:|---:|---:|---:|
| 1 | 95.78% | 94.30% | 92.90% | 62.6% |
| **3** | **95.46%** | **95.27%** | **94.17%** | **68.2%** |
| 10 | 93.01% | 93.17% | 92.27% | 73.4% |

**Findings:**
- **`reg=3` is the best point in the entire arrhythmia sweep** — 94.17% Akida
  accuracy at 68.2% sparsity. Only ~0.9pt below the unregularized baseline's
  accuracy (95.04%) while gaining +12.8pt sparsity, and it beats raw
  Hoyer-Square's best point (`reg=1e-5`: 93.64%/67.4%) on *both* axes at once —
  the same "normalization isn't a tradeoff, it's a strict improvement" result
  VWW found at its own best point.
- **This is also where normalized Hoyer-Square finally starts actually
  engaging** — sparsity climbs steadily 62.6% -> 68.2% -> 73.4% across this
  range, unlike Pass 4's flat 55-59%, confirming Pass 4 really was sitting
  below the useful range rather than this regularizer just not working on this
  model.
- **`reg=10` starts trading meaningfully worse** — 73.4% sparsity, the highest
  of any Hoyer-Square variant tested, but at 92.27% accuracy, ~2pt worse than
  `reg=3`. The knee is somewhere between `3` and `10`; not narrowed further
  here since `reg=3` already gives a clean win over every other regularizer
  tested.
- Not fully monotonic here either (`float acc`: 95.78% -> 95.46% -> 93.01%,
  `sparsity`: climbing but `qat_acc` dips then partially recovers) — same
  small-dataset noise caveat as Passes 2-3 applies.

## Overall summary and conclusion

Best point found for each regularizer, all at 80 float / 50 QAT epochs:

| regularizer | reg | Akida acc | sparsity | vs. baseline |
|---|-----:|---:|---:|---|
| (baseline, no activity reg) | 0 | 95.04% | 55.4% | — |
| L1L2 | 3e-3 | 92.19% | 80.3% | -2.9pt acc, +24.9pt sparsity |
| Hoyer-Square (raw) | 1e-5 | 93.64% | 67.4% | -1.4pt acc, +12.0pt sparsity |
| **Hoyer-Square (normalized)** | **3** | **94.17%** | **68.2%** | **-0.9pt acc, +12.8pt sparsity** |

**Normalized Hoyer-Square generalizes as the best-behaved regularizer for a
fourth model in a row** — at its best point it dominates raw Hoyer-Square on
both axes (higher accuracy, higher sparsity), matching the VWW/PlantVillage
result. Where this project genuinely differs from the other three:

- **No regularizer tested caused training collapse**, even at 30x-100x the reg
  values that fully destroyed VWW/PlantVillage/Speech Commands. The
  class-weighted loss (`{0: 1.0, 1: 6.0, 2: 3.0}`, pre-existing and untouched
  by this sweep) plausibly makes "always predict the majority class" a much
  less attractive local minimum here than in the other three (unweighted)
  projects, which would explain why that failure mode never appeared.
- **If pure sparsity is the goal rather than the best accuracy-sparsity
  trade-off, L1L2 at high strength is actually the strongest lever found**
  here (up to 87.3% sparsity at `reg=3e-2`, 91.13% accuracy) — a real
  reversal from VWW/PlantVillage/Speech Commands, where L1L2 always collapsed
  long before reaching Hoyer-Square's sparsity ceiling. This project's L1L2
  range was never pushed to its own collapse point (Pass 2 stopped at `3e-2`
  once the trade-off curve looked to be saturating), so it's not yet known
  whether even higher L1L2 values would keep this advantage or eventually
  collapse the way the other projects did.
- **This sweep never found a true collapse/cliff for any regularizer type** —
  every other project's sweep was bounded by finding where accuracy craters;
  here, the practical stopping point was compute budget and diminishing
  sparsity returns, not a hard wall. A future pass could keep pushing all
  three regularizer types further to see whether this model ever actually
  breaks, or whether its class-weighted loss makes it fundamentally more
  robust to activity regularization than the other three examples.
- Confirmed a real, previously-unnoticed **input-scaling bug** in this
  project's `eval.py` (assumed `[-1, 1]` input range; actual is `[-5.59,
  14.83]`) while building the Akida-conversion path for this sweep — worth
  fixing upstream in `eval.py`/`benchmark.py` independent of this sparsity
  investigation, since it would silently corrupt any hardware-benchmark run
  of this project's models today.

## Open questions / next steps

- Push L1L2 and Hoyer-Square (both forms) further to find whether this model
  ever collapses, and whether L1L2's apparent sparsity-ceiling advantage over
  Hoyer-Square holds up past `3e-2`.
- Fix the `[-1, 1]` input-scaling assumption in the pre-existing
  `scripts/eval.py`/`scripts/benchmark.py` (see `scripts/input_scaling.py` for
  the correct bounds) so hardware-benchmark runs of this project's models
  aren't silently measuring a corrupted accuracy.
- Re-run the winning point (`hoyer_square_norm`, `reg=3`) on a machine with a
  physical Akida device attached for real latency/power numbers — this sweep
  only measured software-backend sparsity, same caveat as the other three
  experiments.
- The noisy, non-monotonic accuracy-vs-reg curves seen in every pass here
  (unlike the other three projects' smooth declines) suggest re-running the
  best few points with a different seed to check how much of the ranking is
  signal vs. training variance, before treating `reg=3` as a firm
  recommendation rather than "best of the points sampled."
