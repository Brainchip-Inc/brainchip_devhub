# Dataset Evaluation Report: Camera-Less / Photodiode Gaze Estimation

**Project:** Camera-Less Eye Tracking on Akida (Tiny Temporal CNN / MLP, 64 timesteps x 8 channels -> gaze x,y)
**Date:** 2026-08-24
**Author:** Research pass for `docs/dataset_report.md` (BrainChip reference project)

## Bottom line up front

**No public dataset of genuine raw photodiode/PSOG intensity signals with (x,y) gaze
ground truth exists today.** This is a narrow, largely proprietary/industrial sensing
modality (Meta Reality Labs, Google, NVIDIA Research, EssilorLuxottica/Politecnico di
Milano, and academic groups have all built photodiode or single-pixel-detector
prototypes), and none of the hardware papers reviewed below released their raw sensor
recordings. The one genuine exception — a real, downloadable dataset that is
*purpose-built for PSOG research* — is the **Texas State University PS-OG dataset**
(Katrychuk, Griffith & Komogortsev, ETRA/ICCV 2019): it contains real infrared
eye-camera recordings from 23 subjects plus released code that deterministically
simulates a low-dimensional photosensor array (a 3x5 = 15-channel virtual PSOG grid)
from those images, with real gaze ground truth. It is the most defensible starting
point, with one important caveat: it is licensed **CC BY-NC-SA 4.0** (non-commercial).

Everything else usable for this project is either (a) a large, high-quality,
permissively-licensed **video-oculography (VOG/camera-derived) eye-movement** dataset
(GazeBase, GazeBaseVR, JuDo1000) that supplies excellent real gaze-trajectory ground
truth but *no* sensor signal resembling a photodiode array, or (b) a real photodiode
*hardware* paper (NVIDIA/UW NextGaze & LED2Gaze, Dartmouth LiGaze, EssilorLuxottica
"Hidden Photodetectors") with small in-house subject pools and **no released dataset**.

---

## 1. GazeBase

- **Paper:** Griffith, Lohr, Abdulin & Komogortsev, "GazeBase, a large-scale,
  multi-stimulus, longitudinal eye movement dataset," *Scientific Data* 8, 184 (2021).
  https://www.nature.com/articles/s41597-021-00959-y (preprint:
  https://arxiv.org/pdf/2009.06171)
- **Data:** https://doi.org/10.6084/m9.figshare.12912257 (figshare, "GazeBase Data
  Repository")
- **Sensor modality:** Video-oculography (VOG). Recorded on an **SR Research EyeLink
  1000** (pupil–corneal-reflection tracker), i.e. camera-derived gaze, not photodiode.
  What is released is the *derived* gaze signal (angle in degrees), not eye images.
- **Subjects / duration:** 322 college-aged participants, 12,334 monocular
  recordings, 9 rounds of collection over 37 months, 2 sessions/round, 7 tasks per
  session (fixation, horizontal saccade, random oblique saccade, reading, video
  free-viewing x2, gaze-driven gaming).
- **Sampling rate:** 1,000 Hz.
- **Format / dimensionality:** Per-recording CSV with columns `n` (timestamp, ms),
  `x`, `y` (gaze position, degrees of visual angle), `val` (validity flag), `xT`,
  `yT` (target position where applicable), `dP` (uncalibrated pupil area), `lab`
  (fixation/saccade/blink label). This is a **1-2 channel time series** (x, y), not
  an 8-channel sensor-array signal.
- **License / access:** CC BY 4.0, open download, no request form.
- **Suitability:** Excellent ground-truth gaze *trajectories* at very high sampling
  rate and huge subject diversity — the best available substrate for **driving a
  photodiode-response simulation** (see Recommendation). But by itself it contains no
  sensor-array-like signal; heavy simulation work (a rendering/response model, as in
  Rigas et al. 2017 or Katrychuk et al. 2019 below) is required to turn its (x,y)
  trajectories into an 8-channel pseudo-photodiode input. A companion tool,
  **SP-EyeGAN** (https://github.com/aeye-lab/sp-eyegan, pretrained on GazeBase), can
  generate additional synthetic gaze trajectories/scanpaths via GANs if more
  volume/diversity is needed, but it also only produces (x,y)-style traces, not
  sensor signals.

## 1b. GazeBaseVR (companion dataset)

- **Paper:** Lohr et al., "GazeBaseVR, a large-scale, longitudinal, binocular
  eye-tracking dataset collected in virtual reality," *Scientific Data* 10, 177
  (2023). https://www.nature.com/articles/s41597-023-02075-5 (arXiv:
  https://arxiv.org/abs/2210.07533)
- **Data:** https://doi.org/10.6084/m9.figshare.21308391
- **Sensor modality:** VOG, binocular, collected inside a VR headset (still
  camera-based, not photodiode).
- **Subjects / duration:** 407 participants, 5,020 binocular recordings, up to 6
  sessions/participant over 26 months, 5 tasks (vergence, horizontal smooth pursuit,
  video viewing, self-paced reading, random oblique saccades).
- **Sampling rate:** 250 Hz.
- **Format:** Same style CSV as GazeBase (x/y per eye, degrees visual angle).
- **License:** CC BY 4.0, open download.
- **Suitability:** Same profile as GazeBase — a high-quality, permissively licensed
  gaze-trajectory source for driving a photosensor simulation, with the added
  relevance that it is VR-headset-collected (closer to the near-eye geometry a
  photodiode array would actually see) and binocular. Still no raw sensor data.

## 2. JuDo1000

- **Paper:** Makowski, Jäger & Scheffer describe the collection; used widely in
  Kasneci-group oculomotor biometrics work (e.g., "DeepEyedentificationLive," Jäger
  et al.).
- **Data:** OSF, https://osf.io/5zpvk/ (DOI 10.17605/OSF.IO/5ZPVK). Also
  downloadable programmatically via the `pymovements` Python package
  (https://pymovements.readthedocs.io/en/latest/datasets/JuDo1000.html).
- **Sensor modality:** VOG — **EyeLink Portable Duo**, binocular, pixel/gaze
  coordinates. Not photodiode.
- **Subjects / duration:** 150 participants, 4 sessions each, at least one week
  apart. Task: following a randomly jumping dot on screen.
- **Sampling rate:** 1,000 Hz.
- **Format:** Binocular gaze-position time series (screen pixel / degree
  coordinates); exact column schema is documented in the `pymovements` loader.
- **License / access:** Custom/unlabeled OSF license ("Other" — see `license.txt` in
  the OSF storage; not a standard CC license as of this check). Freely downloadable,
  no request form, but **verify the license text before any commercial/internal-use
  redistribution**.
- **Suitability:** Same category as GazeBase — good real eye-movement dynamics
  (smooth pursuit of a jumping target, i.e. exactly the paradigm PSOG hardware
  papers use for calibration), but pure VOG (x,y) output, no sensor-array signal.
  Smaller and less diverse than GazeBase; only useful here as an additional
  trajectory source for simulation, not as a primary dataset.

## 3. TEyeD (now "TüEyeD")

- **Paper:** Fuhl, Kasneci & Kasneci, "TEyeD: Over 20 million real-world eye images
  with Pupil, Eyelid, and Iris 2D and 3D Segmentations, 2D and 3D Landmarks, 3D
  Eyeball, Gaze Vector, and Eye Movement Types," arXiv:2102.02115 (checked at v5,
  Jan 2026).
- **Data:** https://es-cloud.cs.uni-tuebingen.de/d/8e2ab8c3fdd444e1a135/?p=%2FTEyeDS&mode=list
  (mirror: https://hctlsrva.edu.sot.tum.de/TEyeDS/)
- **Sensor modality — VERIFIED CAMERA-BASED, NOT PHOTODIODE.** TEyeD is explicitly
  "the world's largest unified public data set of eye **images** taken with
  head-mounted devices," acquired with **seven different head-mounted eye trackers**
  (two integrated into VR/AR devices), covering car rides, simulator rides, outdoor
  sports, and daily indoor activities. Gaze vector, eyeball model, and eye-movement
  type (fixation/saccade) are derived by fitting 2D/3D landmarks and segmentation
  masks to the eye images — a classic video-oculography pipeline, not a photosensor
  reflectance measurement.
- **Subjects / duration:** ~132 participants; per-recording video lengths from a
  few minutes to several hours; frame rates **25–200 Hz** depending on the source
  eye tracker.
- **Format / dimensionality:** Full eye images plus dense per-frame annotations
  (segmentation, landmarks, 3D eyeball, gaze vector) — i.e., high-dimensional image
  data, the opposite of the low-bandwidth signal this project targets.
- **License / access:** Open download links are provided, but explicit standard
  license terms (CC BY/BY-NC/etc.) were not confirmed from the paper or hosting
  pages during this review — check the dataset's own terms page before use.
- **Suitability: poor fit, as anticipated.** This confirms the concern raised in the
  task: **TEyeD is a camera/pupil-video dataset, not a photodiode dataset.** It could
  theoretically serve as raw material for a from-scratch photosensor-response
  simulation (windowing/aggregating pixel intensities over image sub-regions, exactly
  as Katrychuk et al. did with their own IR camera images below), but doing so
  requires downloading and processing a 20-million-image corpus and re-deriving a
  new sensor simulation and calibration pipeline — a large, self-directed
  undertaking with no precedent in the literature specifically using TEyeD this way.
  It is not recommended as a first choice.

## 4. PSOG (Photosensor Oculography) datasets from the literature

This is the modality that actually matches the project (a handful of photodiode
channels -> temporal signal -> gaze x,y). It is a small research niche, dominated by
two groups: **Google/Google-affiliated + academic collaborators (Rigas, Raffle,
Komogortsev)** and the **Texas State University Komogortsev lab** (Katrychuk,
Griffith). No large public "PSOG dataset" comparable to GazeBase exists; the field
mostly relies on ad-hoc, non-released hardware studies or model-based simulation.

### 4a. Texas State PS-OG dataset — the closest thing to a real, downloadable PSOG dataset

- **Papers:**
  - Katrychuk, Griffith & Komogortsev, "Power-efficient and shift-robust eye-tracking
    sensor for portable VR headsets," ETRA 2019.
    https://dl.acm.org/doi/10.1145/3314111.3319821 /
    https://par.nsf.gov/servlets/purl/10101300
  - Griffith, Katrychuk & Komogortsev, "Assessment of Shift-Invariant CNN Gaze
    Mappings for PS-OG Eye Movement Sensors," ICCV 2019 OpenEDS Workshop.
    https://openaccess.thecvf.com/content_ICCVW_2019/papers/OpenEDS/Griffith_Assessment_of_Shift-Invariant_CNN_Gaze_Mappings_for_PS-OG_Eye_Movement_ICCVW_2019_paper.pdf
    (arXiv:1909.05655)
- **Data + code:**
  - Dataset landing page: https://digital.library.txstate.edu/handle/10877/8574
  - Full raw dataset (9.15 GB, per-subject IR image sequences + gaze-signal files):
    https://digital.library.txstate.edu/bitstream/handle/10877/7955/UnprocessedDataset.zip
  - "Slim" preprocessed dataset (59 MB, simulated PSOG sensor outputs, no raw images):
    https://digital.library.txstate.edu/bitstream/handle/10877/7955/SlimDataset.zip
  - Code: https://github.com/pseudowolfvn/psog_nn (branch `etra2019`) — full
    preprocessing (image -> simulated sensor), ML training, and plotting pipeline.
- **Sensor modality:** **Real IR eye images**, from which PSOG-style photosensor
  output is **deterministically simulated** post-hoc (not raw photodiode
  measurements, but a rigorously documented, code-reproducible camera-to-sensor
  simulation, using real human eye motion and real image data as the source signal
  rather than a rendered 3D model). Hardware: custom eye tracker with an 850 nm IR
  illuminator, ThorLabs DCC1545M camera, hot mirror, and EyeLink 1000 chin-rest.
- **Simulation method (reproducible from the released code):** each IR frame is
  cropped/aligned, then a virtual **3x5 grid of 15 photosensors** is overlaid; each
  sensor's "reading" is the average pixel intensity within a 121x121-pixel window
  after Gaussian windowing (sigma = 1/4 window size) to emulate angular sensor
  sensitivity falloff — precisely the same simulation principle used in the PSOG
  survey paper (4b, below) and in the NVIDIA/UW single-pixel-detector work (Sec. 5b).
  Because the grid geometry is a configurable parameter in the released code, it can
  be re-run to produce an **8-channel** grid (e.g. 2x4 or a custom layout) directly
  matching this project's target input shape instead of the paper's native 15
  channels.
- **Subjects / duration:** 23 subjects recorded (1 excluded due to a head-tracking
  marker failure), each performing a random horizontal/vertical step-stimulus
  saccade task over a target range of ±19.1° (H) x ±16.7° (V) degrees of visual
  angle.
- **Sampling rate:** 120 fps (camera-limited) — well below the 250–1,000 Hz native
  rate of real PSOG/VOG hardware, but sufficient to populate 64-timestep windows
  (64 samples ≈ 0.53 s at 120 Hz).
- **Ground truth gaze (x,y):** Yes — real VOG-derived gaze position in degrees of
  visual angle, recorded synchronously with the IR images, for every frame.
- **License:** **CC BY-NC-SA 4.0** ("Copyright (c) 2019, Texas State University. All
  rights reserved... licensed under a Creative Commons
  Attribution-NonCommercial-ShareAlike 4.0 ... For all other uses, please contact
  the Office for Commercialization and Industry Relations at Texas State
  University"). **This is a non-commercial license** — a real constraint for a
  BrainChip commercial reference project; internal research/demo use is likely fine,
  but any public/commercial redistribution of derived models or data would need
  explicit permission from Texas State's OCIR.
- **Suitability: best available match.** Real gaze ground truth, real human eye
  motion, a fully open and reproducible sensor-simulation pipeline that can be
  re-parameterized to exactly 8 channels, and prior published baselines (MLP: 1.07°
  mean error for ±2 mm shifts; CNN variants improving on that) to benchmark against.
  Limitations: modest scale (23 subjects, single saccade task, no smooth-pursuit or
  naturalistic viewing), 120 Hz ceiling, and the non-commercial license.

### 4b. Rigas, Raffle & Komogortsev — PSOG simulation survey (no dataset released)

- **Paper:** "Photosensor Oculography: Survey and Parametric Analysis of Designs
  Using Model-Based Simulation," arXiv:1707.05413 (also ETRA-adjacent venue).
- **Method:** Pure model-based simulation — a parametric 3D eyeball model rendered
  in Blender, with a Gaussian-windowed virtual photosensor response computed from
  the rendered images (the same simulation principle later reused in 4a). The real
  eye-movement trajectory used to drive the simulated eyeball (a jumping-dot task,
  36 s, amplitudes to ±10°) came from a **single subject** recorded on an EyeLink
  1000 at 1,000 Hz — used only as a motion-driving ground truth, not distributed.
- **Data/code availability:** None found. This is a design-space exploration paper
  (varying detection-area size/shape/count and reporting accuracy/crosstalk
  trade-offs for 4 idealized PSOG sensor layouts); no dataset or simulation code
  release was located.
- **Suitability:** Valuable as a **methodology reference** (it documents exactly how
  to build a photosensor-response simulator from an eyeball model, and gives
  quantitative sensor-geometry design guidance — e.g. detection-area size vs.
  accuracy/crosstalk trade-offs) if the project ends up building its own simulator
  from GazeBase trajectories rather than reusing 4a's dataset. Not usable directly
  as a dataset.

### 4c. Other PSOG hardware papers (real prototypes, real subjects, data not released)

- **"Developing photo-sensor oculography (PS-OG) system for virtual reality
  headsets"** (Rigas, Raffle, Komogortsev, ETRA 2018,
  https://dl.acm.org/doi/10.1145/3204493.3208341) — the "Hybrid PS-V" system fusing
  a 2x2 simulated IR-receiver grid with a low-speed (5 Hz) VOG correction signal;
  simulated on a **single synthetic subject**, not released.
- **"Design and Implementation of an Eyewear-Integrated Infrared Eye-Tracking
  System"**, *Sensors* 26(7):2065 (2026),
  https://pmc.ncbi.nlm.nih.gov/articles/PMC13074884/ — a real PSOG prototype (4 NIR
  LEDs + 4 photodiodes/eye, 16-feature vector, 12.8 kHz photodiode sampling, LED
  multiplexing 400–3200 Hz), but **tested only on artificial eye models** (a doll's
  eye and an OEMI-7 model eye on a motorized gimbal), explicitly because "this
  prototype has not been certified as eye-safe [and] could not be tested on
  humans." Ground truth is gimbal-encoder angle, not human gaze. Data explicitly
  **not released** ("not publicly available due to confidentiality obligations and
  NDAs with the industrial partner").
- **"Development of a Low-Power Wearable Eye Tracker based on Hidden
  Photodetectors"** (ETRA 2025, EssilorLuxottica / Politecnico di Milano,
  https://dl.acm.org/doi/10.1145/3715669.3727347) — 4 photodetectors per lens edge
  (16 signals/lens), 70 Hz, ~4° accuracy on an artificial eye plus a small number of
  human volunteers; no evidence of a public dataset release.
- **ETRA 2026 (very recent, Aug 2026):**
  - "Toward Always-On PSOG: Wearable Gaze Tracking System via a PPG-Derived Optical
    Analog Front-End," https://doi.org/10.1145/3797246.3806215 — repurposes the
    MAX86181 photoplethysmography (PPG) analog front-end for PSOG sensing.
  - "A Low-Power Distance-Based Approach to Gaze Classification in Pervasive
    Smart-Eyewear Eye Tracking," https://doi.org/10.1145/3797246.3806216 — a
    32-dimensional feature vector at 1 kHz from sequentially illuminated IR
    LED/photodiode pairs in a smart-eyewear frame, using distance-based
    classifiers for discrete gaze-zone estimation (not continuous x,y regression).
  - Both are behind the ACM paywall; full text was not accessible during this
    review, so **no confirmation either way on data release** — worth periodically
    rechecking, since this is the most current work in the space and closest in
    spirit (32-dim photodiode feature vector at 1 kHz) to the project's target
    format.

## 5. Other camera-less / photodiode gaze benchmarks found in the literature

### 5a. LiGaze (Dartmouth, SenSys 2017)

- **Paper:** Li, Liu & Zhou, "Ultra-Low Power Gaze Tracking for Virtual Reality,"
  ACM SenSys 2017. https://dl.acm.org/doi/10.1145/3131672.3131682
  (https://www.cs.dartmouth.edu/~xia/papers/sensys17-ligaze.pdf)
- **Sensor modality:** Real photodiodes placed around a VR lens, sensing **reflected
  screen light** (no active IR emitters) modulated by the pupil's light-absorption
  property.
- **Performance (from the project's own reporting, not this project's testing):**
  128 Hz, 6.1° within-user / 10.1° cross-user mean accuracy, 791 µW total power
  (solar-cell powerable); a follow-up reduced this to 395 µW at 120 Hz.
- **Data availability:** **No dataset release found** on the paper, the lab project
  page (http://dartnets.cs.dartmouth.edu/ligaze), or the related battery-free-eye-
  tracker page (http://dartnets.cs.dartmouth.edu/eye). Materials are explicitly
  shared "on a non-commercial basis" with no data links.
- **Suitability:** Real, relevant hardware precedent; not usable as a dataset
  source.

### 5b. NextGaze / LED2Gaze (University of Washington + NVIDIA Research, ISMAR 2020)

- **Paper:** Li, Whitmire, Stengel, Boudaoud, Kautz, Luebke, Patel & Akşit, "Optical
  Gaze Tracking with Spatially-Sparse Single-Pixel Detectors,"
  arXiv:2009.06875 / IEEE ISMAR 2020, https://ieeexplore.ieee.org/document/9284794/
- **Sensor modality:** Real photodiode + IR LED hardware (two prototypes).
  - **NextGaze:** 8 photodiodes + 3 LEDs per eye, time-multiplexed illumination,
    **400 Hz**. The gaze model is literally an **8-input, 2-output fully-connected
    network** (4 hidden layers x 64 units, tanh activations) — i.e. almost exactly
    the architecture and channel count this project targets, minus the temporal
    convolution.
  - **LED2Gaze:** 6 LEDs per eye used as *both* emitter and detector (no separate
    photodiodes), **250 Hz**, mapped to gaze via a Gaussian Process Regression
    model.
- **Subjects / accuracy:** NextGaze evaluated on 6 subjects (4 M/2F): 1.68° mean
  error on smooth pursuit, 2.67° on a ±20° fixation grid, 16 mW power. LED2Gaze
  evaluated on 14 subjects (10 M/4F): 1.57° mean / 1.12° median error, reported
  power draw of ~800 mW (higher than NextGaze despite the simpler sensing scheme,
  per the paper's own reporting).
- **Ground truth gaze (x,y):** Yes — screen-space calibration targets during a
  smooth-pursuit + fixation-grid protocol.
- **Data availability:** **No dataset or code release found.** The paper is purely
  a hardware/algorithm contribution; there is no GitHub link or supplementary data
  archive referenced in the text. This is the single closest architectural match to
  the project's envisioned model (small photodiode vector -> small MLP -> 2D gaze)
  but is not obtainable as a dataset — only as a reproduction target (i.e., a
  reason to eventually build equivalent hardware).

### 5c. Event-based ("camera-less" in the frame-based-imaging sense) gaze tracking

- **Dataset:** Angelopoulos et al., "Event-based, Near-Eye Gaze Tracking Beyond
  10,000 Hz" — dataset repo https://github.com/aangelopoulos/event_based_gaze_tracking
- **Sensor modality:** Dynamic Vision Sensor (DAVIS346b event camera) under IR
  illumination — an *event stream*, not a fixed small photodiode array. This is a
  different low-power/high-speed modality (sparse pixel-level brightness-change
  events) rather than a handful of aggregate photodiode intensities.
- **Subjects:** 27 recorded (24 used in the paper; 3 excluded for a suboptimal
  setup).
- **Format:** Raw event `.aerdat` files (timestamp, polarity, row, column) plus
  25 fps grayscale reference frames (346x260, 8-bit); ground-truth gaze position
  is embedded as pixel row/column in each frame's filename.
- **License:** MIT (code + data).
- **Suitability:** Good precedent for "genuinely camera-less, ultra-low-bandwidth,
  high-temporal-resolution, openly released, real ground truth" — but the sensor
  format (sparse spatial events from a 346x260 array) is architecturally different
  from a small fixed photodiode array (8 scalar channels). Useful as a fallback
  benchmark if the project's scope is broadened to "any non-frame-based sensing,"
  but it would require its own reformatting (e.g., binning events into coarse
  spatial buckets) to resemble an 8-channel photodiode vector, and doing so would
  be a research contribution in itself rather than a straightforward reuse.

---

## 6. Comparison table

| Dataset / source | Real sensor modality | Subjects | Sampling rate | Native channel format | Gaze (x,y) GT | License / access | Photodiode-signal fidelity |
|---|---|---|---|---|---|---|---|
| **GazeBase** | VOG (EyeLink 1000) | 322 | 1,000 Hz | 1–2 ch (x,y + pupil area) | Yes (native output) | CC BY 4.0, open | None — trajectory only, needs full simulation pipeline |
| **GazeBaseVR** | VOG (VR headset) | 407 | 250 Hz | 1–2 ch/eye (x,y) | Yes (native output) | CC BY 4.0, open | None — trajectory only |
| **JuDo1000** | VOG (EyeLink Portable Duo) | 150 | 1,000 Hz | Binocular x,y | Yes (native output) | Custom/"Other" OSF license — verify | None — trajectory only |
| **TEyeD** | Camera (7 HMD eye trackers) | ~132 | 25–200 Hz | Full eye images + derived landmarks | Yes (derived gaze vector) | Open download, license unconfirmed | None — would need bespoke image-to-sensor simulation on 20M images |
| **Texas State PS-OG dataset** (Katrychuk/Griffith/Komogortsev) | Real IR eye images -> **simulated** 3x5 (15-ch) photosensor grid, code-reproducible | 23 | 120 Hz (camera-limited) | 15 channels (reconfigurable, incl. to 8) | Yes (real VOG, deg. visual angle) | **CC BY-NC-SA 4.0** (non-commercial) | High — closest reproducible approximation of real photodiode-array data |
| Rigas/Raffle/Komogortsev PSOG survey (2017) | Fully simulated (Blender render + Gaussian sensor model), driven by 1-subject EyeLink trace | 1 (motion source) | 1,000 Hz (motion source) | 4 idealized designs, up to 9 detectors | Yes (simulation ground truth) | No dataset/code release found | Methodology only, not usable directly |
| PS-OG eyewear feasibility study (Sensors 2026) | Real photodiode hardware, tested on **artificial eyes only** (no humans, not eye-safety certified) | 0 human | 12.8 kHz | 4 photodiodes -> 16-feature vector | Gimbal-encoder angle (not human gaze) | Not public (industrial NDA) | Real signal, but no human data and not released |
| ETRA 2025 "Hidden Photodetectors" (EssilorLuxottica/PoliMi) | Real photodiode hardware | Artificial eye + a few humans | 70 Hz | 16 signals/lens | Partial | Not released | Real signal, tiny scale, not released |
| **LiGaze** (Dartmouth) | Real photodiodes (passive, screen-light reflection) | Not disclosed publicly | 120–128 Hz | Few photodiodes | Likely yes (in-paper only) | Not released | Real hardware precedent only |
| **NextGaze / LED2Gaze** (UW/NVIDIA) | Real photodiodes/LEDs, 8-ch / 6-ch | 6 / 14 | 400 Hz / 250 Hz | **8 channels (NextGaze) — closest architectural match** | Yes (screen calibration targets) | Not released | Real signal, ideal architecture, but no data available |
| Event-based gaze tracking (DAVIS346, Angelopoulos et al.) | Event camera (DVS), not photodiode array | 24 (of 27 recorded) | >10,000 Hz (event), 25 fps (frames) | Sparse spatial event stream, 346x260 | Yes (pixel position in filename) | MIT, open | Different modality (spatial event stream, not scalar photodiode vector) |

---

## 7. Recommendation

**No off-the-shelf, permissively-licensed, real raw-photodiode dataset with (x,y)
gaze ground truth exists.** Be direct about this with stakeholders: the project's
premise (a public photodiode dataset for training) does not hold as stated, and the
plan should shift from "find a dataset" to "construct a defensible
photodiode-like dataset from the best available real eye-movement data," exactly as
the two existing PSOG research groups have done.

### Primary choice: Texas State PS-OG dataset + `psog_nn` simulation code

Use the Katrychuk/Griffith/Komogortsev dataset
(https://digital.library.txstate.edu/handle/10877/8574) and its released code
(https://github.com/pseudowolfvn/psog_nn) as the **primary training/validation
source**:

- It is the only dataset found that pairs **real human eye motion**, **real gaze
  ground truth**, and a **documented, reproducible, code-available simulation** of a
  small photosensor array — not an ad hoc synthesis invented for this project.
- Re-run the released preprocessing with the sensor-grid parameter changed from 3x5
  (15 channels) to an 8-channel layout (e.g. 2x4, or drop to the horizontal/vertical
  differential pairs used by classic PSOG designs) to match the project's declared
  (64, 8) input shape exactly. The 120 fps frame rate comfortably supports 64-sample
  windows (~0.53 s of context).
- Published baselines (MLP ~1.07° mean error under simulated ±2 mm sensor shift;
  CNN improvements in the ICCV workshop paper) give a literature anchor to sanity-
  check the Tiny Temporal CNN / Tiny MLP models described in this project's README.
- **Caveat to resolve before any external/customer-facing use:** the license is
  **CC BY-NC-SA 4.0**, held by Texas State University, with an explicit note to
  contact their Office for Commercialization and Industry Relations for any use
  beyond non-commercial research. Confirm with legal/BD whether this reference
  project (internal demo vs. distributed customer artifact) requires that
  outreach, since BrainChip is a commercial entity. If it does not clear that bar,
  fall back to the synthesis path below using GazeBase instead, which is fully
  commercial-friendly (CC BY 4.0).

### Fallback: Build a photosensor simulation on top of GazeBase (or GazeBaseVR)

If the Texas State dataset's non-commercial license is disqualifying, the most
defensible alternative is to reproduce the **published PSOG simulation methodology**
(Rigas, Raffle & Komogortsev 2017, and the same Gaussian-windowed-sensor-response
approach used in the Texas State pipeline) on top of **GazeBase** or **GazeBaseVR**
(both CC BY 4.0, both far larger and more diverse than the Texas State dataset: 322
and 407 subjects respectively, multiple task types including smooth pursuit and
naturalistic viewing):

1. Take GazeBase's real (x,y) gaze-angle trajectories as motion ground truth (as
   Rigas et al. did with a single EyeLink recording).
2. Drive a parametric 3D eyeball/eyelid model (their approach used Blender with a
   published eyeball model) or a lighter-weight analytic PSOG response function
   (a documented option in the same paper, since it also reports closed-form
   accuracy/crosstalk trade-offs for idealized sensor geometries) to synthesize an
   8-channel photodiode-array response for each trajectory sample.
3. Calibrate/validate the synthetic-sensor pipeline's channel outputs and gaze
   error against the Texas State study's reported numbers and Rigas et al.'s
   accuracy/crosstalk curves before trusting it for model development, to avoid
   silently overstating fit to the real problem.

This path is more engineering-heavy (there is no off-the-shelf simulator to
download; only a documented methodology) but yields a much larger, permissively
licensed, fully-owned dataset, and directly follows precedent set by peer-reviewed
work rather than inventing a synthesis method from scratch.

### What NOT to use as a primary dataset, and why

- **TEyeD**: confirmed camera/pupil-video data, not photodiode; adopting it would
  require the same from-scratch simulation effort as the GazeBase fallback but on a
  much larger, harder-to-process (20M image) corpus with less clean gaze-angle
  ground truth than GazeBase's native (x,y) columns, and unconfirmed licensing.
  Reserve only if a much larger raw-image corpus for simulation becomes necessary.
- **JuDo1000**: smaller and less thoroughly documented than GazeBase, ambiguous
  ("Other") license — no advantage over GazeBase for this project's purposes.
- **NextGaze/LED2Gaze, LiGaze, EssilorLuxottica prototypes**: the best
  *architectural* precedents (especially NextGaze's literal 8-photodiode-to-MLP
  design), but none released data. They are a strong argument for eventually
  building in-house photodiode hardware to collect a proprietary dataset once the
  simulated-data model is validated — not usable today.

### Honest summary of confidence

- High confidence: GazeBase/GazeBaseVR/JuDo1000 are VOG, not photodiode (directly
  stated in their own methods sections); TEyeD is camera-based (directly stated in
  its own abstract/intro).
- High confidence: the Texas State PS-OG dataset is real, downloadable today, and
  license terms are explicit (CC BY-NC-SA 4.0) — verified directly from the
  `LICENSE.md` in the released code repository.
- Medium confidence: no dataset release exists for LiGaze, NextGaze/LED2Gaze, or
  the 2025/2026 ETRA photodiode-eyewear papers — based on absence of any
  data/code link in the papers, project pages, and searches; a direct email to the
  authors is the only way to fully rule out an unlisted release (e.g. available
  "on request").
- Lower confidence: exact license status of TEyeD and JuDo1000, and full technical
  detail of the two ETRA 2026 papers, since their full texts sit behind an ACM
  paywall this review could not access.
