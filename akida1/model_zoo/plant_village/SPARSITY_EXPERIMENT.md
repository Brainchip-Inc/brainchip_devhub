# PlantVillage Sparsity Regularization Experiment

Tracking doc for Asana task **"Akida 1: Sparsify VWW model"** — applying the same
investigation done on VWW (see [`../vww/SPARSITY_EXPERIMENT.md`](../vww/SPARSITY_EXPERIMENT.md))
to the PlantVillage disease-classification example, to check whether the
findings there (Hoyer-Square, especially the element-count-normalized variant,
beating L1L2) generalize to a different model/dataset.

Branch: `akida1-sparsify-plantvillage`

## Setup

- Model: AkidaNet, alpha=0.5, 224x224 input, 38 classes (1,156,054 params) —
  same architecture as the published baseline in `docs/README.md`. Larger than
  VWW's alpha=0.25/96x96/226,906-param model in every dimension (wider, higher
  resolution input, more output classes), so ReLU activation tensors are
  substantially bigger here — relevant to the raw-vs-normalized Hoyer-Square
  comparison (see VWW's findings on why that matters).
- Regularizer: same `-reg`/`--reg-type` flag added to `plant_village_train.py`
  as for VWW: `l1l2` (`regularizers.L1L2(reg, reg)`, pre-existing), `hoyer_square`
  (raw `factor * (sum|x|)^2 / sum(x^2)`), and `hoyer_square_norm` (same,
  divided by the activation tensor's element count) — identical
  `HoyerSquare` implementation ported from `vww_train.py`.
- Pipeline per data point: float train -> float eval -> `cnn2snn quantize`
  (8/4/4-bit) -> QAT fine-tune -> QAT eval -> `cnn2snn convert` -> Akida eval ->
  sparsity measurement. Same stage sequence as `plant_village_train.sh`, with
  `-reg` applied at both the float-training and QAT stages so sparsity
  survives quantization.
- Sweep script: [`plant_village_sparsity_sweep.py`](plant_village_sparsity_sweep.py),
  ported from VWW's sweep script with PlantVillage's script names/CLI
  differences (no `-b` batch-size flag on `plant_village_eval.py`; float/QAT
  accuracy is printed as `Test accuracy:` rather than `Validation accuracy:`).
  Sparsity is measured with
  [`plant_village_sparsity_only.py`](plant_village_sparsity_only.py) (software
  backend, via `akida_models.sparsity.compute_sparsity`) rather than
  `plant_village_benchmark.py`, because **this machine has no physical Akida
  hardware attached** (`akida.devices()` returns empty) — so real
  latency/power numbers are not available from these runs, same caveat as VWW.
- Environment: same dedicated conda env `vww-sparsify` used for the VWW sweep
  (has `tensorflow_datasets`, needed for PlantVillage's tfds-based data
  loading). Trained on GPU 1 of the shared training box
  (`CUDA_VISIBLE_DEVICES=1`) — GPU 0 was occupied by another user's job at the
  time.
- Data: `plant_village` tfds dataset (Plant leaf diseases dataset without
  augmentation, ~54k images, 38 classes), downloaded via the existing
  `plant_village_data.py` (auto-downloads to `./data/plant_village` on first
  use, mirrored from `data.brainchip.com`).
- Baseline pipeline (float/QAT epochs, learning rates) matches
  `plant_village_train.sh`'s existing defaults (10 float epochs @ 1e-3, 2 QAT
  epochs @ 1e-4) rather than VWW's raised epoch counts — that model already
  reaches 99.61%/99.43%/99.43% (float/QAT/Akida) with this schedule per the
  published README, so there's no indication more epochs are needed before
  regularization is introduced. If a regularized run doesn't converge in this
  budget, epochs will be raised the same way VWW's were.

## Results

### Pass 1 — coarse L1L2 sweep (10 float epochs, 2 QAT epochs)

Same three reg values as VWW's Pass 1 (`0`, `1e-5`, `1e-3`), using
`plant_village_train.sh`'s existing epoch schedule.

| reg | float acc | QAT acc | Akida acc | sparsity |
|-----:|---:|---:|---:|---:|
| 0 (baseline) | 99.72% | 99.52% | 99.56% | 54.2% |
| 1e-5 | 97.29% | 55.30% | 52.38% | 81.75% |
| 1e-3 | 18.97% | 14.79% | conversion failed | — |

Baseline is consistent with the published reference numbers in `docs/README.md`
(99.43-99.61%, 54.58% sparsity), validating the pipeline.

**Findings:**
- **PlantVillage is far more sensitive to L1L2 regularization than VWW was.**
  VWW's `reg=1e-5` was already past its useful knee but still landed at a
  respectable 66.75-66.79% Akida accuracy; the same reg value here collapses
  QAT accuracy to 55.30% (from a 97.29% float accuracy just one stage
  earlier) — a ~42pt QAT-stage drop, more than double VWW's ~17.5pt drop at
  the same value.
- **`reg=1e-3` doesn't just collapse accuracy, it breaks conversion outright.**
  Float/QAT accuracy both fall to a ~15-19% floor (this task's majority-class
  fraction, given 38 unevenly-sized classes -- unlike VWW's binary
  51.5%-non_person floor), but `cnn2snn convert` then raises
  `ValueError: Unsupported act_step, it should be equal or greater than
  1/16.` -- the regularization drove quantized activations so degenerate that
  a valid per-tensor quantization step size couldn't even be computed. VWW's
  equivalent `reg=1e-3` point converted fine and just landed on its
  own accuracy floor (51.52%); this is a harder failure mode PlantVillage hit
  that VWW didn't.
- **The useful range is much narrower than VWW's and sits below `1e-5`.**
  Given `1e-5` already collapses QAT training here, Pass 2 needs to sample
  well below it to find where PlantVillage's accuracy/sparsity knee actually
  is, rather than reusing VWW's `1e-6`-centered range as-is.

### Pass 2 — finer sweep below the Pass 1 collapse point (10 float epochs, 2 QAT epochs)

Given `reg=1e-5` already collapsed QAT training in Pass 1, sampled an order of
magnitude and more below it: `1e-7`, `1e-6`, `3e-6`.

| reg | float acc | QAT acc | Akida acc | sparsity |
|-----:|---:|---:|---:|---:|
| 1e-7 | 99.78% | 99.39% | 99.48% | 60.1% |
| 1e-6 | 99.12% | 98.01% | 98.03% | 73.3% |
| 3e-6 | 98.84% | 91.97% | 90.55% | 77.6% |

**Findings:**
- **`reg=1e-6` is an excellent trade, much better than VWW got at the same
  value.** 73.3% sparsity (vs. 54.2% baseline) for only a ~1.5pt accuracy
  cost (99.56% -> 98.03% Akida acc). VWW's `reg=1e-6` cost ~3pt of accuracy
  for less sparsity gain (65.8%) -- PlantVillage's larger model and higher
  baseline sparsity headroom appear to make this a cheaper trade here.
- **The knee sits between `1e-6` and `3e-6`.** `1e-6`'s QAT-stage drop is
  small (~1.1pt, float 99.12% -> QAT 98.01%); `3e-6`'s QAT-stage drop is much
  larger (~6.9pt, float 98.84% -> QAT 91.97%) for only 4.3 more points of
  sparsity (73.3% -> 77.6%) -- a much worse rate of exchange, the same shape
  of "cliff right after the knee" VWW found, just compressed into a narrower
  reg range (VWW's cliff started past `1e-6`; here it's already steep by
  `3e-6`, an order of magnitude below where VWW's L1L2 fully collapsed).
- **`reg=1e-6` is the best L1L2 point found so far**, consistent with VWW's
  conclusion that `1e-6` was L1L2's best point there too -- same reg value,
  different (better) tradeoff, on a different model/dataset.

### Pass 3 — knee-region sweep (10 float epochs, 2 QAT epochs)

Denser sampling between `1e-6` and `3e-6`: `1.5e-6`, `2e-6`.

| reg | float acc | QAT acc | Akida acc | sparsity |
|-----:|---:|---:|---:|---:|
| 1.5e-6 | 99.13% | 96.76% | 95.91% | 74.4% |
| 2e-6 | 98.99% | 94.81% | 94.25% | 75.9% |

**Note (tooling bug, not a data issue):** hit a pre-existing bug in the sweep
script's directory-tag formatting (inherited from the VWW version) --
`f"{reg:.0e}"` rounds `1.5e-6` up to `2e-06`, so this pass's two points
collided on the same output directory name. Each point's full
train->eval->quantize->QAT->eval->convert->eval pipeline still runs and
records its own metrics into the CSV *before* the next point starts training
into (and overwriting) that same directory, so the numbers above are
unaffected -- only the on-disk `1.5e-6` model checkpoints were later
overwritten by `2e-6`'s. Fixed in `plant_village_sparsity_sweep.py` by
switching to `f"{reg:.1e}"` (kept as a followup for `vww_sparsity_sweep.py`,
which never hit this in practice since none of VWW's sweep points shared a
rounded value).

**Findings:**
- **`reg=1e-6` remains the best point** -- confirmed by comparison, not just
  assumption. `1.5e-6` gets barely more sparsity (74.4% vs. 73.3%, +1.1pt)
  for much worse accuracy (95.91% vs. 98.03%, -2.1pt); `2e-6` is worse still
  (75.9% sparsity, 94.25% accuracy). Every step past `1e-6` trades a little
  sparsity for a lot of accuracy.
- **Confirms a smooth, monotonic decline, not a sudden cliff** -- QAT-stage
  drop from float grows steadily through the region: `1e-6` ~1.1pt ->
  `1.5e-6` ~2.4pt -> `2e-6` ~4.2pt -> `3e-6` ~6.9pt. Same shape VWW found in
  its own knee-region pass, just compressed into a ~3x narrower reg range.

**L1L2 summary (all 8 points tested):**

| reg | Akida acc | sparsity |
|-----:|---:|---:|
| 0 (baseline) | 99.56% | 54.2% |
| 1e-7 | 99.48% | 60.1% |
| **1e-6** | **98.03%** | **73.3%** |
| 1.5e-6 | 95.91% | 74.4% |
| 2e-6 | 94.25% | 75.9% |
| 3e-6 | 90.55% | 77.6% |
| 1e-5 | 52.38% | 81.75% |
| 1e-3 | conversion failed (18.97% float / 14.79% QAT) | — |

`reg=1e-6` is L1L2's best point on PlantVillage, same reg value that won on
VWW, but a substantially better trade here: ~1.5pt accuracy cost for +19pt
sparsity, vs. VWW's ~3pt cost for +13pt sparsity at the same reg value.

### Pass 4 — raw Hoyer-Square sweep (10 float epochs, 2 QAT epochs)

Same three reg values as VWW's Pass 4 (`1e-7`, `1e-6`, `1e-5`), for direct
comparability against both VWW's Hoyer-Square results and this experiment's
own L1L2 numbers above.

| reg | float acc | QAT acc | Akida acc | sparsity |
|-----:|---:|---:|---:|---:|
| 1e-7 | 99.74% | 99.52% | 99.59% | 55.5% |
| 1e-6 | 99.48% | 99.36% | 99.37% | 62.3% |
| 1e-5 | 98.36% | 98.07% | 97.94% | 73.1% |

Head-to-head against L1L2 at the same reg values:

| reg | L1L2 Akida acc / sparsity | Hoyer-Square Akida acc / sparsity |
|---:|---:|---:|
| 1e-7 | 99.48% / 60.1% | 99.59% / 55.5% |
| 1e-6 | 98.03% / 73.3% | 99.37% / 62.3% |
| 1e-5 | 52.38% / 81.75% (collapsed) | **97.94% / 73.1%** |

**Findings:**
- **No collapse at `reg=1e-5`, where L1L2 catastrophically failed.**
  QAT-stage drop is only ~0.3pt (float 98.36% -> QAT 98.07%), nothing like
  L1L2's ~42pt collapse at the same value. Exactly the same qualitative
  result VWW found -- Hoyer-Square's scale-invariant ratio doesn't blow up
  under quantization the way L1L2's raw magnitude sum does.
- **Hoyer-Square `reg=1e-5` lands almost exactly on L1L2's best point** --
  73.1% sparsity at 97.94% accuracy, essentially matching L1L2 `reg=1e-6`'s
  73.3%/98.03%. The two land on nearly the same point on the accuracy/sparsity
  frontier, but by a fundamentally more stable path: L1L2 got there via a
  QAT-stage cliff it had already gone past (its own best point was a full
  order of magnitude of reg back at `1e-6`), while Hoyer-Square is still on
  the gentle part of its curve at `1e-5` -- room to push further without the
  sudden failure L1L2 hits.
- **Hoyer-Square hasn't found its own knee/cliff yet** -- all three points
  (1e-7 through 1e-5) show a smooth, gentle decline with no sign of the
  QAT-stage instability L1L2 showed starting at `1e-6`. Same open question as
  VWW: where's the limit, and does normalizing change it?

### Pass 5 — normalized Hoyer-Square sweep (10 float epochs, 2 QAT epochs)

**Calibration.** Ran the same one-off diagnostic as VWW's Pass 5 (probing
every ReLU layer's activations on one real training batch from the untrained
model), specific to this model's architecture -- including `fc_1/relu`, the
new dense-block ReLU layer VWW didn't have.

| | sum across all 15 ReLU layers, one batch (b=32) |
|---|---:|
| raw `hoyer_square` | 17,077,786 |
| `hoyer_square_norm` | 4.739 |

Ratio ≈ 3,603,762 -- an order of magnitude larger than VWW's ratio (362,203),
because PlantVillage's per-layer imbalance is itself much larger: raw values
ranged **5,434 to 4,305,795** (~792x) between the smallest (`fc_1/relu`,
`n=512`) and largest (`conv_1/relu`, `n=401,408`) ReLU layers -- a much wider
spread than VWW's ~186x, driven by the 224x224 input producing far larger
early-layer activation maps. Normalized values stayed within **0.069 to
0.488**, bounded regardless of layer size as expected.

Scaled Pass 4's three `reg` values by that ratio for matched total
regularization-loss magnitude:

| hoyer_square reg | hoyer_square_norm reg (scaled) |
|---:|---:|
| 1e-7 | 0.36 |
| 1e-6 | 3.6 |
| 1e-5 | 36 |

**Results** (`sweep_results_hoyer_norm/`, `sweep_results_hoyer_norm.log`):

| reg (norm) | matched raw reg | float acc | QAT acc | Akida acc | sparsity |
|---:|---:|---:|---:|---:|---:|
| 0.36 | 1e-7 | 99.65% | 99.47% | 99.37% | 55.8% |
| 3.6 | 1e-6 | 99.52% | 99.43% | 99.39% | 63.2% |
| 36 | 1e-5 | 98.29% | 98.16% | 98.08% | 78.2% |

Head-to-head against Pass 4's raw values at matched regularization-loss
magnitude:

| matched reg | raw `hoyer_square` Akida acc / sparsity | `hoyer_square_norm` Akida acc / sparsity |
|---:|---:|---:|
| weakest (1e-7 / 0.36) | 99.59% / 55.5% | 99.37% / 55.8% |
| middle (1e-6 / 3.6) | 99.37% / 62.3% | 99.39% / 63.2% |
| strongest (1e-5 / 36) | 97.94% / 73.1% | **98.08% / 78.2%** |

**Findings:**
- **Same pattern as VWW, and a bigger win here.** At the two weaker matched
  points the formulations are within noise of each other. At the strongest
  matched point, normalized wins on *both* axes at once -- higher accuracy
  (98.08% vs. 97.94%) **and** meaningfully higher sparsity (78.2% vs. 73.1%,
  +5.1pt) -- a larger normalized-vs-raw gap than VWW saw at its strongest
  point (+1.5pt sparsity there). Consistent with the calibration: PlantVillage's
  raw-formulation imbalance (~792x across layers) is over 4x larger than
  VWW's (~186x), so there was more distortion for normalization to fix, and a
  bigger payoff from fixing it.
- **`hoyer_square_norm` at `reg=36` is the best point found across the
  entire experiment** -- 78.2% sparsity for a ~1.5pt accuracy cost from
  baseline (99.56% -> 98.08%). It beats every other candidate: raw
  Hoyer-Square's `1e-5` (73.1% sparsity, ~1.6pt cost) and L1L2's `1e-6`
  (73.3% sparsity, ~1.5pt cost) both reach meaningfully less sparsity for a
  comparable or worse accuracy cost.
- **Still no knee found for either Hoyer-Square formulation** -- `reg=36`'s
  decline (99.65% -> 99.52% -> 98.29% float acc) is as gradual as raw
  Hoyer-Square's was; a followup sweep further out (e.g. `100`, `300`) would
  be needed to find where normalized Hoyer-Square's own cliff is on this
  model.

## Overall conclusion

Full picture across all 8 L1L2 points, 3 raw Hoyer-Square points, and 3
normalized Hoyer-Square points:

| reg | L1L2 Akida acc / sparsity | raw Hoyer-Square Akida acc / sparsity | normalized Hoyer-Square Akida acc / sparsity |
|-----:|---:|---:|---:|
| 0 (baseline) | 99.56% / 54.2% | 99.56% / 54.2% | 99.56% / 54.2% |
| 1e-7 / 0.36 | 99.48% / 60.1% | 99.59% / 55.5% | 99.37% / 55.8% |
| 1e-6 / 3.6 | **98.03% / 73.3%** | 99.37% / 62.3% | 99.39% / 63.2% |
| 1.5e-6 | 95.91% / 74.4% | — | — |
| 2e-6 | 94.25% / 75.9% | — | — |
| 3e-6 | 90.55% / 77.6% | — | — |
| 1e-5 / 36 | 52.38% / 81.75% (collapsed) | 97.94% / 73.1% | **98.08% / 78.2%** |
| 1e-3 | conversion failed | — | — |

(Hoyer-Square columns are aligned by matched regularization-loss magnitude,
not equal `reg` values -- see the Pass 5 calibration above.)

**`hoyer_square_norm` at `reg=36` is the overall winner: 78.2% sparsity for
a ~1.5pt accuracy cost from baseline (99.56% -> 98.08%).** This mirrors VWW's
conclusion exactly (normalized Hoyer-Square won there too), and the
generalization holds up on a substantially different model (2.3x more
alpha, 5.4x more input pixels, 38 classes vs. 2) and dataset -- with the
normalized-vs-raw gap actually *widening* here, consistent with this model's
larger per-layer size imbalance. Two qualitative differences from VWW worth
flagging:
- **PlantVillage is far more sensitive to raw/unnormalized regularization.**
  L1L2 needed 10x less regularization strength here to hit the same collapse
  VWW needed `reg=1e-3` for, and additionally broke `cnn2snn convert` outright
  at that strength -- a harder failure VWW never hit.
- **Hoyer-Square's stability advantage over L1L2 is correspondingly larger
  here.** Where VWW's best Hoyer-Square point beat L1L2's best by a modest
  margin, PlantVillage's normalized-Hoyer-Square point reaches +4.9pt more
  sparsity than L1L2's best (`1e-6`) for about the same accuracy cost.

Whether even this best accuracy cost is worth adopting at all is still a
product call, not resolvable from these numbers alone without real hardware
latency figures (see below).

## Open questions / next steps

- Get real hardware latency/power numbers for baseline vs `hoyer_square_norm`
  `reg=36` (the current best candidate) on an AKD1500 -- this is the missing
  piece to know whether the sparsity gain actually moves latency enough to
  justify the accuracy cost. Software-measured sparsity is not on the same
  footing as hardware-measured numbers, so this can't be answered from
  software-only measurements. (No published hardware baseline exists yet for
  a PlantVillage regularized model, unlike VWW.)
- Neither Hoyer-Square formulation's knee/cliff has been found -- both Pass 4
  and Pass 5 only went as far as their strongest tested point (`1e-5` raw /
  `36` normalized) and were still on the well-behaved part of the curve
  there. A followup sweep further out would locate where it starts trading
  accuracy away, and might turn up an even better tradeoff point.
- Fix the same directory-tag-collision bug (`f"{reg:.0e}"` rounding) in
  `../vww/vww_sparsity_sweep.py` -- already fixed here in
  `plant_village_sparsity_sweep.py`, but the VWW original still has it
  (harmless in practice there since none of VWW's sweep points shared a
  rounded tag, but worth fixing for robustness).
