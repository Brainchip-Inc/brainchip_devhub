---
description: Generate a brainchip_devhub model zoo example that evaluates and benchmarks existing pretrained models (no training stage)
---

# Model Zoo Evaluation Example Generator

Generate a model zoo example that **evaluates and benchmarks already-trained models**,
rather than training one. Use this when training is impractical to reproduce (ImageNet-scale
datasets) and the point of the example is to publish reference numbers and reusable
backbones.

For examples that *do* train, use `/model-zoo-example` instead — that command is the
training-pipeline generator and this one deliberately drops its `_train.py` / `_train.sh`
steps.

**Canonical reference: `akida1/model_zoo/imagenet_akidanet/`.** Read it before generating.
It is the worked version of everything below, covering six models (3 widths × 2 resolutions)
on ImageNet.

## Usage

```
/model-zoo-eval-example <name> [--akida-version 1|2]
```

Parse from `$ARGUMENTS`: `NAME`, `AKIDA_VERSION` (default 1). Target directory is
`akida<AKIDA_VERSION>/model_zoo/<NAME>/`. Note `akida2/` may not exist yet — create it if
targeting v2.

---

## Step 1 — Establish the inputs before writing anything

Ask or determine:

1. **Which models.** The set of variants the example covers — width multipliers, input
   resolutions, or architecture variants. Each combination becomes a row in the Model Card.
2. **Where the model files are.** Usually already downloaded into `pretrained_models/`, or
   fetchable from `https://data.brainchip.com/models/AkidaV<N>/<arch>/`.
3. **The dataset**, and critically **whether it can be redistributed** (see Step 6).
4. **The Akida version target**, which changes more than you would expect (Step 5).

Then verify the models actually load and score sensibly *before* building the example
around them. A five-minute smoke test prevents a day of building on a wrong assumption.

---

## Step 2 — File set

```
<NAME>_preprocessing.py   self-contained copy of the eval preprocessing
<NAME>_data.py            dataset loader + small sample pack + label helpers
<NAME>_model.py           model file resolution + backbone loader
<NAME>_eval.py            accuracy (+ activation sparsity) per variant
<NAME>_benchmark.py       hardware latency/power benchmark
<NAME>_eval.sh            driver (replaces <NAME>_train.sh)
update_readme.py          copied from any sibling example; extend it only to
                          regenerate figures derived from metrics.json
<NAME>_summary_plot.py    optional: cross-model summary figure, drawn from metrics
colab_setup.py            adapted from vww/colab_setup.py
docs/README.md.template   README source of truth
docs/metrics.json         all template keys, measured or "TBD"
docs/sample_mosaic.png    dataset figure
<NAME>_notebook_evaluation.ipynb
<NAME>_notebook_benchmark.ipynb
pretrained_models/        committed via Git LFS, no .gitignore
data/.gitignore           the standard 88-byte ignore-all-but-self file
README.md                 GENERATED - never hand-edit
```

**No `models/` directory and no `_train.py` / `_train.sh`** — there is no training stage, so
there are no working artifacts. Say so explicitly in the README and point at a sibling
training example (`plant_village` runs in ~20 minutes) for readers who want the full
train → quantize → convert pipeline.

---

## Step 3 — Multi-model conventions

This is the main structural difference from the single-model training examples.

### Model selection

Scripts select a model with `-a/--alpha` and `-i/--input-resolution` style arguments plus
`--variant {float,qat,akida}`, **not** the sibling `-l/--loadmodel` path argument — the
metrics key prefix must be derived from the model identity anyway.

### Uniform filenames

Normalise the published filenames into a regular scheme so that path construction is one
f-string rather than a lookup table. Upstream naming is typically irregular (default variants
omit their parameter, others use integer percentages). Write a `model_fetch.sh` at the repo
root that downloads and renames, so the mapping stays reproducible:

```
<arch>_<dataset>_<RES>_alpha_<A>.h5          full precision
<arch>_<dataset>_<RES>_alpha_<A>_qat.h5      quantized
<arch>_<dataset>_<RES>_alpha_<A>_qat.fbz     converted to Akida
```

Keep `_qat.fbz` rather than `_akida.fbz` — the conversion tool names its output after its
input, and fighting that creates more inconsistency than it removes.

### Metrics keys

Every model shares one `docs/metrics.json`, so namespace each key by model identity:
`a50_224_float_t1`, `a100_160_akida_t5`, and so on. Derive the prefix in code
(`metrics_prefix()`), never by hand.

> **Critical gotcha:** `update_readme.py` uses `str.format_map`, which reads a **dot in a
> field name as attribute access**. `{a0.5_224_float_t1}` parses as `a0` → attribute
> `5_224_float_t1` and fails. Metrics keys must scale the alpha to an integer (`a50_`) even
> when the *filenames* use decimals (`alpha_0.5`). The two schemes differ deliberately —
> document this in `metrics_prefix()` so nobody later "fixes" it.

Also: **every literal brace in the template must be doubled** (`{{`, `}}`). Prefer writing
code snippets that contain no braces at all.

### README tables

- **One accuracy table**, with a spanning header row per group
  (`<tr><td colspan="N"><b>224 × 224 input</b></td></tr>`), primary configuration first.
- **One hardware benchmark table per model**, one row per mapping mode — a single combined
  table becomes unreadable past two or three models. With three modes and nine columns the
  per-model table is already at the width a README can carry; resist adding a fourth axis.
- **One cross-model summary figure**, above the per-model tables. Six tables of nine
  columns do not show a reader the shape of the trade-off; one scatter of accuracy against
  latency and against energy does. Derive it from `metrics.json` and regenerate it from
  `update_readme.py`, not from the benchmark script — then it stays in step with the tables
  without needing hardware, and a prose edit cannot leave a stale figure behind.
  **Plot the models as points, not as joined curves.** A handful of widely spaced
  configurations has nothing to interpolate between; a connecting line asserts a smooth
  trade-off that was never measured, and the gap between two such lines invents a difference
  between the series. Say what the measured points support and no more — with six models the
  useful claim was that none is dominated and both cost measures rank them identically, which
  needs no line at all.
- Verify placeholders against metrics before generating:
  ```python
  import string; {f[1] for f in string.Formatter().parse(template) if f[1]}
  ```

---

## Step 4 — Script specifics

### `<NAME>_preprocessing.py`

A self-contained copy of the `akida_models` inference preprocessing — the example should
document in one readable place what happens to an input before it reaches the model.

**Then prove it is a faithful copy**: assert bit-equality against the `akida_models`
implementation over real samples at every supported resolution. A silently drifting copy is
the worst failure mode this file has.

Call out in the docstring that normalisation lives **inside** the model (a `Rescaling` layer
from `input_scaling`), so the pipeline delivers raw uint8 and must not normalise. This is the
most common cause of a pretrained model scoring at chance.

### `<NAME>_eval.py`

- Report the metrics the dataset's field actually uses (for ImageNet: top-1 **and** top-5).
- Keras path: `model.evaluate` with `['accuracy', SparseTopKCategoricalAccuracy(k=5)]`.
- Akida path: manual batch loop; `predict` returns `(B, 1, 1, C)`, so `squeeze(axis=(1,2))`.
- Support `-n/--num-samples` to cap, and a `--samples` mode running the small sample pack
  with per-image predicted-vs-true output. Make `--save-metrics` **refuse** to record a
  sample-pack run.
- **Compute activation sparsity here, not in `_benchmark.py`.** It needs no hardware, and
  gating it behind the benchmark script (which exits early without a device) leaves the
  column empty for no reason. This is a deliberate deviation from the training-example
  template.
- Do **not** call `tf.config.experimental.enable_op_determinism()` — evaluation has no
  shuffle or augmentation so it is already deterministic, and it materially slows large runs.

### `<NAME>_benchmark.py`

Structurally a copy of `plant_village_benchmark.py`. Keep the `brainchip_utils` imports and
call sequence identical. Differences for a multi-model example: namespace the output plot
filenames per model (`ref_benchmark_results_full_<tag>.png`), and default the samples to the
small redistributable pack with `-d` opting into the full dataset.

Benchmark samples must be **real images** — Akida is event-driven, so latency and power
depend on activation sparsity, which depends on the input. Random noise gives wrong numbers.
A handful of real images cycled is fine; say so in the README rather than implying more.

Drive the mapping modes from one list (`map_modes = ['Minimal', 'AllNps', 'HwPr']`) and derive
the metrics key segment as `mm.lower()`, so adding or removing a mode needs no other code
change. Two things then follow:

- **Guard the `None` return.** `full_model_benchmark()` catches a mapping failure and returns
  `None`; the caller immediately does `full_results[mm]['num_nps'] = ...` and dies with a
  `TypeError`. Skip the mode instead — a benchmark sweep is long and expensive, and one mode
  that will not map onto a single hardware sequence should cost you that mode's row, not the
  run. A skipped mode leaves its keys at `TBD` and drops its column from the plot, which
  `plot_full_model_results()` already handles.
- **Do not write `{prefix}sparsity` from this script.** It computes sparsity for the per-layer
  plot, but persisting it means a benchmark run silently overwrites the eval script's
  validation-set figure with one derived from the handful of benchmark samples. Compute here,
  persist only in `_eval.py`.

Seed every mode's keys in `metrics.json` as `"TBD"` **before** the first hardware run.
`format_map` raises `KeyError` on a missing key, and the four power/energy keys are only
written when the INA sensor is actually detected.

---

## Step 5 — Akida version differences (do not hard-bake v1)

**The default `cnn2snn` context is v2.** Any v1 model construction or conversion must be
wrapped in `with set_akida_version(AkidaVersion.v1):` or it silently builds the wrong
architecture and fails to match the weights. Always set the context explicitly, whichever
version you target.

Verified differences (checked against akida_models 1.14.0 / akida 2.19.2):

| | Akida 1 | Akida 2 |
|---|---|---|
| `get_params_by_version()` → | `(fused=True, post_relu_gap=False, 'ReLU6')` | `(fused=False, post_relu_gap=True, 'ReLU3.75')` |
| Separable conv | fused single `SeparableConv2D` | separate `DepthwiseConv2D` + `Conv2D` |
| Global avg pooling | **before** the neighbouring ReLU | **after** the ReLU |
| Quantization tool | `cnn2snn quantize -i 8 -w 4 -a 4` | `quantizeml` |
| Default bit-width | 4 | 8 (`get_default_bitwidth()`) |
| Published weights | `data.brainchip.com/models/AkidaV1/…`, `_iq8_wq4_aq4.h5` | `…/AkidaV2/…`, `_i8_w4_a4.h5` / `_i8_w8_a8.h5` |
| Reference device | AKD1500, `CLOCK_FREQUENCY = 400e6` | AKD2500 — **verify the clock frequency** |

`akida.MapMode` currently exposes three strategies, and the canonical example measures all
three: `Minimal` minimises the hardware resources used, `AllNps` maximises the NPs used while
keeping the *minimum* number of hardware passes, and `HwPr` maximises the NPs used while 
letting the pass count grow. On AKD1500, `HwPr` was the fastest *and* the lowest-energy 
mapping on all six ImageNet models, so measure it rather than assuming two modes bracket 
the range. Confirm which modes are meaningful for the target device rather than assuming 
all three apply — and note that some sibling training examples (`vww`, `plant_village`) 
still measure only two.

**Write the architecture prose from the version you are targeting.** The AkidaNet
description in `imagenet_akidanet` (fused separables, pre-ReLU pooling) is *v1-specific* and
becomes wrong if copied into a v2 example. Generate the layer listing from the actual loaded
model and let it correct you — that is how the pooling/BN ordering error in the original
example was caught.

---

## Step 6 — Dataset licensing

Many benchmark datasets (ImageNet among them) permit research use but **prohibit
redistribution**. Never commit their imagery to the repo.

Order of preference:

1. A redistributable sample pack already mirrored by BrainChip —
   `https://data.brainchip.com/dataset-mirror/…`. The `imagenet_like` pack is 10 labelled
   JPEGs, enough for benchmarking and a genuine end-to-end smoke test.
2. A new BrainChip mirror, if you can upload one.
3. Manual setup by the user, documented with a link to the authoritative instructions
   (e.g. the TFDS catalog page) and the exact directory layout expected.

State the licence position plainly in the README, and be explicit that a 10-image pack is a
pipeline check and **not** an accuracy measurement. Never let a tiny-sample number reach the
Model Card.

---

## Step 7 — Measure before committing to a long run

Before launching a full-dataset sweep, time a single model to estimate the total. For the
ImageNet example the Akida software backend was assumed to be prohibitively slow; measuring
showed 2–17 minutes per model, so all six models × three variants ran on the full 50,000
images with no subsetting.

If a full run genuinely is infeasible, record the sample count in metrics
(`{prefix}{variant}_n`) and footnote it in the README. **Never present a subset number as a
full-dataset number.**

---

## Step 8 — Verification checklist

```bash
python -c "import ast, glob; [ast.parse(open(f).read()) for f in glob.glob('*.py')]"
bash -n <NAME>_eval.sh
python -c "import json, glob; [json.load(open(f)) for f in glob.glob('*.ipynb')]"
python update_readme.py
```

Plus these, which catch the failures that actually happen:

- [ ] Local preprocessing is **bit-identical** to `akida_models` at every resolution
- [ ] Every model path resolves (loop all variant combinations)
- [ ] Template placeholders exactly match metrics keys, both directions; rendered README
      contains no unresolved `{`
- [ ] Backbone loader weights match the `akida_models` pretrained helper where both exist
- [ ] `--samples` smoke test gives sensible per-image predictions
- [ ] Benchmark script exits cleanly with no device attached
- [ ] Every mapping mode has its full key set seeded in `metrics.json`, so the README renders
      before any hardware run
- [ ] After the sweep, no mode was silently skipped (grep the run logs for `did not map`) and
      no `_sparsity` value changed
- [ ] Notebook cells that introspect layers were actually executed, not just written
- [ ] Model files are LFS-tracked (`git check-attr filter -- <file>`)

---

## Gotchas worth carrying forward

- **Model metadata can lie.** The ImageNet 224 checkpoints carry an internal Keras name
  saying `160`, because they were produced by rescaling the 160 models. Read resolution from
  `model.input_shape`, never `model.name`.
- **Rename model files before the first commit.** Once they are in Git LFS, renaming means
  history churn. Normalising names is free while the files are still untracked.
- **Verify claims about the architecture against the loaded model.** Prose written from
  memory or from a description gets layer ordering wrong; a printed layer list does not.
- **The `pretrained_models/` directory gets no `.gitignore`** — that is the one model
  directory whose contents are meant to be committed.
- **Do not write the mapping-mode prose before you have the numbers.** Two claims that read as
  obvious turned out to be wrong on AKD1500, and both had been written into the ImageNet
  README before the hardware run:
  - *"More NPs costs power, and you buy latency with it."* Power does rise, but **total energy
    per inference falls**. Dynamic energy is near-constant across modes (within ~4%) because
    the events the model generates are fixed by the model and the input; what the mapping
    changes is elapsed time, and therefore how long the static floor (~111–114 mW on AKD1500)
    is paid for. Finishing sooner wins on energy even at higher power.
  - *"`AllNps` always spreads wider than `Minimal`."* For the two largest ImageNet models the
    mapper returned an identical solution for both, so those rows agree to within noise. The
    Minimal-vs-AllNps distinction only says something when the model is small relative to the
    mesh.

  Generate the tables first, read them, then write the paragraph that explains them.
