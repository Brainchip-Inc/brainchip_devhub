# VWW Sparsity Regularization Experiment

Tracking doc for Asana task **"Akida 1: Sparsify VWW model"** — investigating the
accuracy/sparsity trade-off when adding activity regularization (L1L2, then
Hoyer-Square) to the VWW (Visual Wake Words) model's ReLU layers during
training.

Branch: `akida1-sparsify-vww`

## Setup

- Model: AkidaNet, alpha=0.25, 96x96 input (226,906 params) — same architecture as
  the published baseline in `docs/README.md`.
- Regularizer: existing `-reg` flag in `vww_train.py`, which applies an
  `activity_regularizer` to every ReLU layer. Two types, selected via
  `--reg-type`:
  - `l1l2` (default, pre-existing): `regularizers.L1L2(reg, reg)`.
  - `hoyer_square` (added for this task, no prior implementation existed in
    the repo): a `HoyerSquare` regularizer added to `vww_train.py`, computing
    `factor * (sum|x|)^2 / sum(x^2)` per activation tensor. Scale-invariant
    sparsity-inducing penalty (Yang, Wen & Li, "DeepHoyer", ICLR 2020) — unlike
    L1/L2's uniform shrinkage, minimizing this ratio favors a few large
    activations over many small ones for the same L1 norm.
  - `hoyer_square_norm`: same formula divided by the activation tensor's
    element count (`tf.size(x)`), bounding the penalty to `(0, 1]` regardless
    of tensor size. Added after checking an unrelated prior project (`DPLS`,
    a personal Hoyer-sparsity research repo) that hit **training collapse**
    with the raw formulation and traced it to exactly this: raw Hoyer-Square
    scales with element count `n` (ranges `[1, n]`), so a single fixed `factor`
    applied uniformly across layers of different sizes exerts wildly uneven
    pressure — the DPLS repo measured raw values around 797,000 vs. ~0.5 once
    normalized, on the same batch. See "Pass 5" below for whether the same
    imbalance was actually distorting the `hoyer_square` results here.
- Pipeline per data point: float train -> float eval -> `cnn2snn quantize`
  (8/4/4-bit) -> QAT fine-tune -> QAT eval -> `cnn2snn convert` -> Akida eval ->
  sparsity measurement. Same stage sequence as `vww_train.sh`, with `-reg` applied
  at both the float-training and QAT stages so sparsity survives quantization.
- Sweep script: [`vww_sparsity_sweep.py`](vww_sparsity_sweep.py). Sparsity is
  measured with [`vww_sparsity_only.py`](vww_sparsity_sweep.py) (software backend,
  via `akida_models.sparsity.compute_sparsity`) rather than `vww_benchmark.py`,
  because **this machine has no physical Akida hardware attached** (`akida.devices()`
  returns empty) — so real latency/power numbers are not available from these runs.
  Getting those would require re-running the benchmark step on a machine with an
  AKD1500 attached.
- Environment: dedicated conda env `vww-sparsify` (isolated from the shared base
  env to avoid an `akida_models` version conflict with another project). Passes
  1-4 trained on GPU 0 of the shared training box (`CUDA_VISIBLE_DEVICES=0`);
  Pass 5 switched to GPU 1 (`CUDA_VISIBLE_DEVICES=1`) since GPU 0 was occupied
  by another user's job (23/24GB used) at the time.
- Data: `vw_coco2014_96` (~110k images, person/non_person), downloaded from
  Silicon Labs' public benchmark dataset mirror per the repo's README.

## Note: sparsity numbers here are not directly comparable to the README's 68.29%

The published baseline sparsity in `docs/README.md`/`docs/metrics.json` is
**68.29%**, but our baseline (same architecture, same `-reg=0`) measures **52.4%**.
This looked at first like a training discrepancy, so it was checked directly:

- Ran `vww_sparsity_only.py` on the actual checked-in
  `pretrained_models/akidanet_vww_qat.fbz` (not a freshly-trained model) ->
  **52.26%** sparsity, essentially identical to our retrained baseline.
- Ran `vww_eval.py` on that same shipped model -> **88.42%** accuracy, an exact
  match to the documented `akida_acc`. Same weights, so no training/version drift.
- Ruled out sample count: `vww_benchmark.py` calls
  `compute_sparsity(ak_model, samples=samples)` without a `batch_size` arg
  (defaults to 100, silently truncating the 1000-sample array it generated),
  while our script explicitly uses all 1000. Re-ran ours with `-n 100` to match
  -> still ~52.2%. Not the cause.
- Root cause, from reading `akida_models.sparsity._compute_sparsity_ak` directly:
  for each target layer it walks `layer.inbounds` to collect every layer feeding
  it, builds a **brand-new** `akida.Model(layers=...)` from just that subset, and
  calls `.forward()` on *that* sub-model. There's no `.map()` call inside
  `compute_sparsity` itself -- whether that `forward()` executes on real
  hardware or the software simulator depends entirely on whether the `Layer`
  objects being reused already carry mapping state from before the call.
  `vww_benchmark.py` calls `ak_model.map(device, mode=Minimal, hw_only=True)`
  *before* calling `compute_sparsity`, so its sub-models run on the physical
  AKD1500. Our fallback (`vww_sparsity_only.py`) never maps anything (no device
  exists on this machine), so every sub-model forward pass runs on the software
  behavioral simulator instead.
- Hardware and simulator are two separately-implemented execution engines for
  the same quantized weights. They agree closely enough to preserve final
  classification accuracy (confirmed identical, 88.42%, above), but aren't
  necessarily bit-exact at every intermediate step -- particularly around each
  layer's activation threshold/saturation logic, which is exactly what decides
  whether a given output is *exactly* zero. That doesn't move the argmax
  prediction, but it compounds into a real, systematic gap in the sparsity count.

**Practical takeaway:** the sweep's sparsity numbers (52% / 78% / 100% below) are
internally consistent and safe to compare *against each other*, but not against
the README's hardware-measured 68.29% baseline. Getting a number on the same
footing as the README requires re-running `vww_benchmark.py` on hardware with a
real AKD1500 attached.

## Results

### Pass 1 — coarse sweep (20 float epochs, 5 QAT epochs)

| reg | float acc | QAT acc | Akida acc | sparsity |
|-----:|---:|---:|---:|---:|
| 0 (baseline) | 89.18% | 88.27% | 88.47% | 52.4% |
| 1e-5 | 85.32% | 67.75% | 66.75% | 77.7% |
| 1e-3 | 51.52% | 51.52% | 51.52% | 100% |

Baseline is consistent with the published reference numbers in `docs/README.md`
(88.42-89.33%), validating the pipeline.

**Findings:**
- `reg=1e-3` fully kills the model: 100% sparsity, and 51.52% accuracy at every
  stage is exactly the "always predict the majority class" floor (`non_person` is
  51.5% of the dataset) — every activation regularized to zero.
- `reg=1e-5` (the smallest value tested) causes a modest float-accuracy dip
  (89.18% -> 85.32%) but a much larger drop after quantization/QAT (85.32% ->
  67.75%, ~17.5pt), versus baseline's QAT drop of ~0.9pt. Sparsity does increase
  meaningfully (77.7%), but 5 QAT epochs isn't enough to recover accuracy once
  quantization interacts with the regularization-induced activation pattern.
- The accuracy cliff is much steeper than expected — even reg values an order of
  magnitude below anything used successfully on larger models collapse this
  model's accuracy, consistent with the task's own note that VWW may already be
  too small/sparse (baseline is already 52-68% sparse) for this technique to pay
  off cleanly.

### Pass 2 — finer sweep (20 float epochs, **10** QAT epochs)

Tested finer regularization values below the collapse point, and doubled QAT
epochs (5->10) to check whether the QAT-stage collapse seen at `reg=1e-5` above
is a QAT-epoch-budget problem rather than a fundamental regularization/
quantization incompatibility.

| reg | float acc | QAT acc | Akida acc | sparsity |
|-----:|---:|---:|---:|---:|
| 1e-7 | 89.30% | 88.49% | 88.17% | 55.5% |
| 1e-6 | 88.40% | 85.36% | 85.36% | 65.8% |
| 1e-5 (10 QAT epochs, vs 5 above) | 85.32% | 67.78% | 66.79% | 77.8% |

**Findings:**
- **QAT epochs is not the fix.** `reg=1e-5` at 10 QAT epochs (67.78%/66.79%) is
  essentially identical to the same reg value at 5 QAT epochs from Pass 1
  (67.75%/66.75%) — doubling QAT training time recovered nothing. This rules out
  "QAT just needs more time" and points to a structural incompatibility between
  this regularization strength and 4-bit weight/activation quantization, not an
  undertrained QAT stage.
- **`reg=1e-7` is indistinguishable from baseline** — sparsity barely moves
  (52.4% -> 55.5%) and accuracy is unchanged within noise (88.47% -> 88.17%
  Akida acc). Too weak to matter.
- **`reg=1e-6` is the interesting middle ground.** Sparsity rises meaningfully
  (52.4% -> 65.8%, in the same ballpark as the README's hardware-measured 68.29%
  baseline, though not measured the same way -- see note above) for a real but
  moderate accuracy cost (88.47% -> 85.36% Akida acc, ~3pt drop). This is the
  best candidate found so far for an actual accuracy/sparsity tradeoff point,
  rather than a full collapse.
### Pass 3 — knee-region sweep (20 float epochs, 5 QAT epochs)

Denser sampling between `1e-6` and `1e-5` to check whether Pass 1/2's apparent
"knee then cliff" shape was real, or an artifact of sparse sampling.

| reg | float acc | QAT acc | Akida acc | sparsity |
|-----:|---:|---:|---:|---:|
| 2e-6 | 87.72% | 80.63% | 80.66% | 71.0% |
| 3e-6 | 87.44% | 76.49% | 76.47% | 72.4% |
| 5e-6 | 86.51% | 67.48% | 67.17% | 79.2% |

**Findings:**
- **The "knee then cliff" framing was wrong** — with denser sampling this is a
  smooth, steep, roughly monotonic decline starting right at `1e-6`, not a
  plateau that suddenly collapses at `1e-5`. Akida accuracy drops steadily:
  85.36% (`1e-6`) -> 80.66% (`2e-6`) -> 76.47% (`3e-6`) -> 67.17% (`5e-6`) ->
  66.75-66.79% (`1e-5`, effectively a floor -- `5e-6` and `1e-5` land within
  0.4pt of each other).
- **`reg=1e-6` remains the best point found across the entire sweep.** Every
  step past it trades accuracy for sparsity at a worse rate: `1e-6`->`5e-6`
  costs 18pts of accuracy (85.36%->67.17%) for only 13 more points of sparsity
  (65.8%->79.2%), versus baseline->`1e-6`'s 3pts of accuracy for 13 points of
  sparsity. There's no better middle ground hiding in this range.

### Pass 4 — Hoyer-Square sweep (20 float epochs, 5 QAT epochs)

Same three-point sweep as Pass 1 (`1e-7`, `1e-6`, `1e-5`), but with
`--reg-type hoyer_square` instead of `l1l2`, to see whether the scale-invariant
penalty finds a better accuracy/sparsity trade than L1L2 did at the same reg
magnitudes.

| reg | float acc | QAT acc | Akida acc | sparsity |
|-----:|---:|---:|---:|---:|
| 1e-7 | 89.12% | 88.70% | 88.68% | 52.4% |
| 1e-6 | 88.55% | 88.39% | 88.36% | 55.1% |
| 1e-5 | 87.43% | 85.56% | 85.27% | 68.9% |

**Findings:**
- **No collapse at any tested value.** Unlike L1L2, which was already past its
  best tradeoff point by `1e-6` and fully collapsed to the majority-class floor
  by `1e-3`, Hoyer-Square stays well-behaved across this entire range — even
  its largest value (`1e-5`) lands at 85.27% Akida accuracy, not far off L1L2's
  *best* point (`1e-6`, 85.36%).
- **At matched reg values, Hoyer-Square strictly dominates L1L2:**
  | reg | L1L2 Akida acc / sparsity | Hoyer-Square Akida acc / sparsity |
  |-----:|---:|---:|
  | 1e-7 | 88.17% / 55.5% | 88.68% / 52.4% |
  | 1e-6 | 85.36% / 65.8% | 88.36% / 55.1% |
  | 1e-5 | 66.75-66.79% / 77.7-77.8% | 85.27% / 68.9% |

  At `1e-6`, Hoyer-Square gets a smaller sparsity gain than L1L2 (55.1% vs
  65.8%) but at essentially zero accuracy cost (88.36% vs baseline's 88.47%,
  L1L2's 85.36%). At `1e-5`, Hoyer-Square reaches *higher* sparsity than L1L2's
  `1e-6` point (68.9% vs 65.8%) while still beating L1L2's own `1e-6` accuracy
  (85.27% vs 85.36% is roughly a wash, but L1L2 needed 10x less regularization
  strength to get there and had nothing better past it, whereas Hoyer-Square at
  `1e-5` isn't showing signs of the cliff yet).
- **This sweep didn't find Hoyer-Square's own knee/cliff** — `1e-5` is still on
  the well-behaved part of the curve. The useful next step, if pursued, is
  pushing `reg` higher (`3e-5`, `1e-4`, ...) to find where Hoyer-Square starts
  trading accuracy the way L1L2 did past `1e-6`.

### Pass 5 — normalized Hoyer-Square sweep (20 float epochs, 5 QAT epochs)

Re-ran the Pass 4 sweep with `--reg-type hoyer_square_norm` to check whether
`hoyer_square`'s per-layer size imbalance (see Setup, and the DPLS precedent)
was distorting Pass 4's results.

**Calibration.** `reg` isn't directly comparable between `hoyer_square` and
`hoyer_square_norm` — the normalized penalty is smaller by whatever the
average per-layer element count is, so reusing Pass 4's `1e-7/1e-6/1e-5`
values for the normalized variant would apply almost no regularization
pressure. To pick comparable values, ran a one-off diagnostic (probing every
ReLU layer's activations on one real training batch from the untrained model)
measuring both forms directly:

| | sum across all 14 ReLU layers, one batch (b=32) |
|---|---:|
| raw `hoyer_square` | 2,075,193 |
| `hoyer_square_norm` | 5.730 |

Ratio ≈ 362,203. Also confirms the imbalance directly: per-layer raw values
ranged **3,022 to 560,892** (~186x) between the smallest (`n=256`) and largest
(`n=36,864`) ReLU layers, while normalized values all stayed within **0.16 to
0.68** — bounded and comparable regardless of layer size, as expected.

Scaled Pass 4's three `reg` values by that ratio to get matched total
regularization-loss magnitude:

| hoyer_square reg | hoyer_square_norm reg (scaled) |
|---:|---:|
| 1e-7 | 0.036 |
| 1e-6 | 0.36 |
| 1e-5 | 3.6 |

**Results** (`sweep_results_hoyer_norm/`, `sweep_results_hoyer_norm.log`):

| reg (norm) | matched raw reg | float acc | QAT acc | Akida acc | sparsity |
|---:|---:|---:|---:|---:|---:|
| 0.036 | 1e-7 | 89.11% | 88.37% | 88.20% | 52.7% |
| 0.36 | 1e-6 | 88.82% | 87.65% | 87.78% | 55.5% |
| 3.6 | 1e-5 | 86.61% | 86.22% | 85.90% | 70.4% |

Head-to-head against Pass 4's raw values at matched regularization-loss
magnitude:

| matched reg | raw `hoyer_square` Akida acc / sparsity | `hoyer_square_norm` Akida acc / sparsity |
|---:|---:|---:|
| weakest (1e-7 / 0.036) | 88.68% / 52.4% | 88.20% / 52.7% |
| middle (1e-6 / 0.36) | 88.36% / 55.1% | 87.78% / 55.5% |
| strongest (1e-5 / 3.6) | 85.27% / 68.9% | **85.90% / 70.4%** |

**Findings:**
- **No collapse here either** — like raw Hoyer-Square, the normalized version
  stays well-behaved across this whole range (float/QAT/Akida acc all decline
  smoothly, no majority-class floor). The DPLS-style catastrophic collapse
  from skipping normalization didn't materialize on this model at these reg
  magnitudes; VWW's per-layer size spread (256 to 36,864 elements, ~144x) and
  DeepHoyer's dynamics on a much smaller model apparently didn't combine into
  the same failure mode DPLS hit on its CNNs.
- **Normalizing is a small, one-directional win, not a wash.** At the two
  weaker matched points the two formulations are within noise of each other.
  At the strongest matched point, normalized *wins on both axes at once* —
  higher sparsity (70.4% vs 68.9%) **and** higher accuracy (85.90% vs
  85.27%) — rather than trading one for the other. Consistent with the
  calibration diagnostic: at this reg magnitude the raw formulation was still
  putting disproportionate pressure on the largest ReLU layers (`conv_1`,
  `n=36,864`) while under-regularizing the smallest (`separable_13`, `n=256`);
  spreading pressure evenly via normalization lets the strong layers push
  further before accuracy suffers.
- **`hoyer_square_norm` at `reg=3.6` is the best point found across the
  entire experiment** — 70.4% sparsity for a 2.6pt accuracy cost from
  baseline (88.47% -> 85.90%), beating every other candidate: raw
  Hoyer-Square's `1e-5` (68.9% / ~3.2pt cost) and L1L2's `1e-6` (65.8% /
  ~3.1pt cost) both cost more accuracy for less sparsity.
- **Still no knee found** — `reg=3.6`'s decline (89.11% -> 88.82% -> 86.61%
  float acc) is as gradual as raw Hoyer-Square's was; a followup sweep
  further out (e.g. `10`, `20`) would be needed to locate where normalized
  Hoyer-Square's own cliff is.

## Overall conclusion

Full picture across all 8 L1L2 values, 3 raw Hoyer-Square values, and 3
normalized Hoyer-Square values:

| reg | L1L2 Akida acc / sparsity | raw Hoyer-Square Akida acc / sparsity | normalized Hoyer-Square Akida acc / sparsity |
|-----:|---:|---:|---:|
| 0 (baseline) | 88.47% / 52.4% | 88.47% / 52.4% | 88.47% / 52.4% |
| 1e-7 / 0.036 | 88.17% / 55.5% | 88.68% / 52.4% | 88.20% / 52.7% |
| 1e-6 / 0.36 | **85.36% / 65.8%** | 88.36% / 55.1% | 87.78% / 55.5% |
| 2e-6 | 80.66% / 71.0% | — | — |
| 3e-6 | 76.47% / 72.4% | — | — |
| 5e-6 | 67.17% / 79.2% | — | — |
| 1e-5 / 3.6 | 66.75-66.79% / 77.7-77.8% | 85.27% / 68.9% | **85.90% / 70.4%** |
| 1e-3 | 51.52% / 100% | — | — |

(Hoyer-Square columns are aligned by matched regularization-loss magnitude,
not equal `reg` values — see the Pass 5 calibration above for the raw/norm
scaling factor.)

**`hoyer_square_norm` at `reg=3.6` is the overall winner: 70.4% sparsity for a
2.6pt accuracy cost from baseline (88.47% -> 85.90%).** It beats every other
candidate tested:
- vs. raw Hoyer-Square `reg=1e-5` (68.9% sparsity, ~3.2pt cost): higher
  sparsity *and* higher accuracy — normalization isn't a tradeoff here, it's a
  strict improvement at this end of the range.
- vs. L1L2 `reg=1e-6` (65.8% sparsity, ~3.1pt cost, L1L2's best point): more
  sparsity for about the same accuracy cost.

Neither Hoyer-Square formulation found its own knee/cliff within the range
tested (both decline gradually through `1e-5`/`3.6`, unlike L1L2's sharp
falloff past `1e-6`), so there is likely a still-better point further out —
see next steps. Whether even the current best accuracy cost is worth adopting
at all is still a product call, not resolvable from these numbers alone
without real hardware latency figures (see below).

## Open questions / next steps

- Get real hardware latency/power numbers for baseline vs `hoyer_square_norm`
  `reg=3.6` (the current best candidate) on an AKD1500 -- this is the missing
  piece to know whether the sparsity gain actually moves latency enough to
  justify the accuracy cost. Software-measured sparsity is not on the same
  footing as the README's hardware numbers (see note above), so this can't be
  answered from software-only measurements.
- Neither Hoyer-Square formulation's knee/cliff has been found — both Pass 4
  and Pass 5 only went as far as their strongest tested point (`1e-5` raw /
  `3.6` normalized) and were still on the well-behaved part of the curve
  there. A followup sweep further out (e.g. normalized `reg=10`, `20`) would
  locate where it starts trading accuracy away, the way L1L2 did past `1e-6`,
  and might turn up an even better tradeoff point.
