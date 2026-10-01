# Camera-Less Eye Tracking on Akida

BrainChip Model Zoo-style reference project demonstrating gaze estimation from
photodiode time-series data using Akida-compatible neural networks.

Asana task: https://app.asana.com/1/754105566748081/task/1217459201908303

## Background

The wearables strategy review identified camera-less eye tracking as a
promising application for Akida/Pico. Photodiode-based eye tracking replaces
image sensors with a small number of infrared photodiodes that generate
low-bandwidth temporal signals, creating an attractive ultra-low-power
workload for Akida.

## Objective

- Identify a suitable public photodiode-based eye-tracking dataset.
- Develop a lightweight temporal model suitable for QuantizeML and Akida
  conversion.
- Benchmark latency, memory footprint, model size, and accuracy.
- Provide a reproducible reference implementation for future customer
  demonstrations and partner evaluations.

## Dataset

Full survey and reasoning in `docs/dataset_report.md`. Bottom line: **no
public dataset of real raw-photodiode signals with gaze ground truth
exists** (confirmed across GazeBase, GazeBaseVR, JuDo1000, TEyeD, the Texas
State PS-OG dataset, and every real photodiode-hardware paper found —
LiGaze, NextGaze/LED2Gaze, EssilorLuxottica's prototype). This project uses:

- **GazeBase** (Griffith et al. 2021, CC BY 4.0, 322 subjects) as the real
  gaze-trajectory source — chosen over the license-cleaner-but-non-commercial
  Texas State PS-OG dataset (CC BY-NC-SA 4.0) because this project's stated
  purpose (customer demos, partner evaluations) is commercial use.
- A simulated 8-channel PSOG/photodiode response (`data/psog_simulate.py`),
  following the Gaussian-windowed virtual-sensor model from Rigas, Raffle &
  Komogortsev (arXiv:1707.05413) and the Texas State PS-OG pipeline, applied
  to GazeBase's `Random_Saccades` task (the closest paradigm to real PSOG
  calibration tasks) across both sessions, all 322 Round_1 subjects.
- Recorded gaze outside the task's actual stimulus range (x: ±16°, y: ±11°,
  measured from the `xT`/`yT` target columns) is filtered out as measurement
  artifact before simulation — see `data/build_dataset.py`.

Caveat carried forward from the dataset report: this is a real, honest
proxy for validating the pipeline and model architecture, not a substitute
for real photodiode hardware data. Numbers here should be read as "does the
approach work," not "this is what an Akida deployment will see in the wild."

## Models

- **Tiny Temporal CNN** (`models/tiny_temporal_cnn.py`): the README's
  originally specced architecture — Input (64 timesteps x 8 channels) ->
  Conv1D(16, k=5) -> BN -> ReLU -> Conv1D(32, k=3) -> BN -> ReLU ->
  GlobalAveragePooling1D -> Dense(32) -> ReLU -> Dense(2). **3,538 params**
  (target was < 50K). Used for the dataset-quality and shift-robustness work
  in **Results so far** below. **Not Akida-deployable as-is** — see
  Deployment flow.
- **Tiny Temporal CNN, causal/streaming form**
  (`models/tiny_temporal_cnn_streaming.py`): the version that actually
  quantizes and converts to Akida. Same conv sizes and **3,490 params**, but
  restructured as causal Conv3D (-> BufferTempConv after quantization) and
  predicting gaze continuously at every timestep instead of once per window.
  Its own accuracy numbers are in **Deployment flow** below, not the
  Results table (different eval protocol - per-timestep vs. per-window).
- **Tiny MLP baseline** (`models/tiny_mlp.py`): Flatten -> Dense(64) ->
  Dense(32) -> Dense(16) -> Dense(2). **35,474 params** (target was
  ~10-20K; the flatten-to-Dense(64) layer alone accounts for most of it
  given the fixed architecture spec). Not attempted on Akida.
- **Optional / not yet done**: event-based encoding
  (`event(t) = |x(t) - x(t-1)| > threshold`) compared against the raw-signal
  CNN, and CNN2SNN conversion of the MLP.

## Results so far

All numbers are mean angular error (degrees) on a subject-disjoint test set
(225 train / 48 val / 49 test subjects, no subject in more than one split).

| Experiment | CNN (3.5K params) | MLP (35K params) |
|---|---|---|
| Zero-shift (idealized, no sensor misalignment) | **0.214°** | 0.243° |
| Under sensor shift (0-4°, one random shift per test recording, models unchanged) | 1.965° | 1.928° |
| Trained WITH shift augmentation, evaluated zero-shift | 0.659° (3x worse) | 0.574° (2.4x worse) |
| Trained WITH shift augmentation, evaluated under shift | 1.876° (only 4.5% better) | 1.866° (only 3.2% better) |
| Zero-shift model + per-session bias calibration (1 calib point), evaluated under shift | 0.341° | 0.401° |
| Zero-shift model + per-session bias calibration (20 calib points), evaluated under shift | 0.238° | 0.320° |

Key findings (details in commit history and `docs/figures/`):

1. **Data quality matters more than architecture here.** Filtering
   out-of-range gaze artifacts (~1-4% of windows, identified by comparing
   against the task's actual stimulus range, not just eyeballing) took CNN
   error from 0.82° to 0.21° and fixed unstable training for both models.
2. **The CNN is not shift-invariant** — it convolves over time, not over
   the sensor/channel axis, so it gets none of the spatial shift-invariance
   the Griffith et al. PS-OG CNN paper describes (theirs convolves over an
   actual 2D sensor grid). Under shift, CNN and MLP degrade almost
   identically.
3. **Shift augmentation is the wrong fix.** The per-recording shift is a
   hidden nuisance parameter with no signal in any single window that would
   let the model infer it, so training under random shift just produces a
   worse compromise mapping, not a shift-aware one.
4. **Per-session calibration is the right fix**, and requires no retraining:
   because the simulation math makes a shifted-array reading for gaze `g`
   identical to a zero-shift reading for gaze `g - shift`, a zero-shift model
   responds to shift with a near-constant bias, correctable from as few as
   one known-gaze calibration point (this mirrors how real eye trackers are
   calibrated).
5. **The idealized zero-shift numbers should not be presented as the
   project's headline accuracy** — the shifted numbers (~1.9-2.0°) are the
   ones comparable to real hardware papers (NextGaze 1.68-2.67°, LED2Gaze
   1.57°, Katrychuk MLP 1.07° under shift), and even those still lack real
   sensor noise, cross-talk, and individual anatomy variation.

## Deployment flow

Dataset preprocessing -> TensorFlow/Keras training -> QuantizeML quantization
-> Akida conversion: **done, end to end** (software backend - no physical
AKD1000/AKD1500 attached in this environment). Full writeup, including two
real toolchain bugs found and fixed along the way, in
`docs/benchmark_report.md`.

The original Conv1D + GlobalAveragePooling1D spec **cannot be deployed on
Akida at all** - quantizeml only supports Conv2D-family, Dense, and
BufferTempConv (FIFO-buffered 1D temporal) layers. Two attempts:

1. Reshape Conv1D as Conv2D over a (64, 1, 8) "tall image" - trains and
   quantizes fine, but `cnn2snn.convert()` rejects it: Akida's 2D conv
   hardware only supports **square kernels**, and a (5,1) kernel isn't one.
2. **Causal streaming redesign** (`models/tiny_temporal_cnn_streaming.py`):
   the actual supported primitive, matching BrainChip's own eye-tracking
   reference model. Predicts gaze continuously at every timestep from
   past-only context (not one value per whole window) - this is what
   actually converts and runs on Akida.

| Stage | Mean angular error (last-timestep) |
|---|---|
| Float (causal streaming) | 0.224° |
| Quantized int8 | 0.247° |
| **Akida-converted** (software backend, n=300 windows) | **0.253°** |

Model: 3,490 params, 8.7 KB `.fbz` file, ~52,000 fps / ~19 µs per frame on
the software backend. Power is **not measurable** without physical
hardware - not estimated, not reported as a number.

**The Tiny MLP also deploys** — and needed no architecture rework at all
(plain Flatten+Dense, no Conv2D-kernel or BufferTempConv complications).
Same signed-int8-input constraint applied here too.

| Stage | Mean angular error |
|---|---|
| Float (tf_keras rebuild) | 0.238° |
| Quantized int8 | 0.352° |
| **Akida-converted** (full 301,808-window test set) | **0.352°** |

Model: 35,474 params, 46 KB `.fbz` file, ~94,000 fps (validly batched — no
buffered state to corrupt, unlike the CNN). Quantization hurts the MLP more
than the CNN (+48% vs +10% relative error) but Akida conversion itself adds
zero further error, versus the CNN's small extra drop. Full comparison and
all four toolchain bugs/constraints found across both models in
`docs/benchmark_report.md`.

## Repo layout

- `data/` — dataset download/preprocessing scripts (raw data and generated
  `.npz` files are git-ignored, not committed):
  - `psog_simulate.py` — gaze -> simulated photodiode array signal.
  - `gazebase_loader.py` — GazeBase CSV loader.
  - `build_dataset.py` — builds the zero-shift train/val/test windows.
  - `build_dataset_shift_aug.py`, `build_shift_eval.py` — shift-robustness
    dataset variants.
  - `visualize.py` — scanpath/signal/window visualization (used for
    `docs/figures/pipeline_demo_*.png`).
- `models/` — Keras model definitions: `tiny_temporal_cnn.py`/`tiny_mlp.py`
  (original spec, TF/Keras 3, GPU-trained — not Akida-deployable),
  `tiny_temporal_cnn_akida.py` (Conv2D reshape — dead end, documented for
  the record), `tiny_temporal_cnn_streaming.py` (causal streaming, the one
  actually deployed), `tiny_mlp_akida.py` (deploys with no architecture
  rework — tf_keras/Keras 2, required by quantizeml/cnn2snn).
- `training/` — `train.py` (original spec, supports `clean`/`shift_aug`),
  `train_akida.py`/`train_akida_streaming.py`/`train_mlp_akida.py`
  (tf_keras versions), `eval_shift.py`, `eval_calibration.py`,
  `streaming_eval_utils.py` (per-timestep sequential evaluation - required
  for the CNN, since Akida's buffered layers only maintain correct state
  for batch_size=1; not needed for the MLP), and `runs/` (trained models +
  metrics, committed since they're small).
- `quantize/` — `quantize_model.py`/`quantize_model_streaming.py`/
  `quantize_model_mlp.py` (QuantizeML 8-bit quantization) and their output
  `.h5` models.
- `deploy/` — `convert_akida.py`/`convert_akida_streaming.py`/
  `convert_akida_mlp.py` (cnn2snn conversion + evaluation) and the final
  `model_streaming.fbz`/`model_mlp.fbz`.
- `docs/` — `dataset_report.md`, `benchmark_report.md`, `figures/` (all
  plots referenced above).

## Deliverables

- [x] Dataset evaluation report with recommendation.
- [x] Training and preprocessing pipeline.
- [x] TensorFlow reference implementation.
- [x] Quantized model (`quantize/model_streaming_quantized.h5`).
- [x] Akida-converted model (`deploy/model_streaming.fbz`).
- [x] Deployment scripts (`deploy/convert_akida_streaming.py`).
- [x] Benchmark report (`docs/benchmark_report.md`) — accuracy and
      software-backend throughput/latency covered; power is explicitly
      not reported (unmeasurable without physical hardware, not estimated).

## Success criteria

- [x] End-to-end TensorFlow -> QuantizeML -> Akida deployment demonstrated
      (software backend; see `docs/benchmark_report.md` for the two real
      toolchain bugs found and fixed to get there).
- [x] Model size below 50K parameters (streaming CNN: 3,490).
- [~] Real-time inference achieved on Akida hardware — ~52,000 fps on the
      **software backend**; not measured on physical AKD1000/AKD1500
      (none attached in this environment).
- [~] Competitive gaze-estimation accuracy on a public dataset — competitive
      against published photodiode-hardware numbers under the shift stress
      test (~1.9-2.0° vs. 1.07-2.67° in the literature), though on a
      simulated rather than real sensor signal.
