---
description: Generate a complete brainchip_devhub Akida 2 model zoo example from a source example
---

# Akida 2 Model Zoo Example Generator

Generate a self-contained **Akida 2** model zoo example in `brainchip_devhub`, ported from a
source example. This skill targets Akida 2 only; for Akida 1 use the separate
`model-zoo-example` skill.

The canonical reference is `akida2/model_zoo/vww/` — every generated file follows the
structure of its VWW counterpart. Read the reference files (Step 2) and mirror them.

## Usage

```
/model-zoo-example-akida2 <NAME>
```

- `<NAME>` — subdirectory name under `akida_models/scripts/` (e.g. `kws`, `mnist`, `face`).

Parse this from `$ARGUMENTS`. Set:
- `NAME` = the example name
- `SOURCE_DIR` = `akida_models/scripts/<NAME>/`
- `TARGET_DIR` = `brainchip_devhub/akida2/model_zoo/<NAME>/`

Resolve the `brainchip_devhub` and `akida_models` repo locations before doing anything else,
in this order:
1. Environment variables `BRAINCHIP_DEVHUB` / `AKIDA_MODELS`, if set.
2. Paths passed as `--devhub <path>` / `--akida-models <path>` in `$ARGUMENTS`.
3. Sibling / ancestor directories of the current working directory (if invoked inside a
   `brainchip_devhub` checkout, look for a sibling `akida_models`).

If a repo cannot be found, stop and ask rather than guessing. Do not hardcode any absolute
user path.

> **Prerequisite:** `akida2/model_zoo/vww/` must exist in the checkout as the reference
> example. If it is not present, stop and confirm with the user — do not generate against a
> missing reference.

---

## Step 1 — Read source scripts

Read every `.py` and `.sh` file under `SOURCE_DIR = akida_models/scripts/<NAME>/`. For each
file, identify:

- **Model file**: the script that builds and returns the model. Note where the model-building
  code is and what its builder function is called, and whether it builds from scratch or from
  pretrained weights (this only affects the `_untrained` save-path suffix). You do **not** need
  to dissect the architecture (pretrained URL, `include_top`, custom head layers, etc.) — in
  3a this file's model definition is lifted essentially verbatim, not reconstructed. The source
  is authoritative for the architecture.

- **Data file**: the script that loads the dataset. Record the dataset name, directory
  structure (does it have `train/`+`val/` subdirs, a TFDS dataset, or something else?),
  input resolution, any augmentation applied, and whether data is uint8 or float.

- **Training file**: the script that trains the model. Record the loss function, optimizer,
  number of epochs, learning rate schedule, and any callbacks.

- **Pipeline structure**: read the source `.sh` file as the authoritative pipeline
  reference. Record: the number of training/fine-tuning epochs, the LR used, whether the
  training action is full training from scratch or fine-tuning from a pretrained starting
  point, any multi-phase training. The reference `akida2/model_zoo/vww/` provides structural
  patterns only; all epoch counts, LR values, and training modes come from the source.

- **Quantization**: look in `.sh` files for the quantization command for reference. Note the
  bit-widths the source uses. For **Akida 2** targets this skill quantizes with `quantizeml`
  (not `cnn2snn quantize`): an 8-bit `i8/w8/a8` variant (no QAT) and a 4-bit `i8/w4/a4`
  variant (QAT only). The input layer weights are always 8-bit. See 3f for the exact pipeline
  — the source is reference-only for quantization; the v2 scheme above is fixed.

- **Calibration samples** (required — see 3f): `quantizeml quantize` calibrates while
  quantizing, and it must be given real samples. The source `.sh` fetches them with a `wget`
  that is normally the **first command in the script**, pointing at the dataset mirror:
  ```bash
  wget -N https://data.brainchip.com/dataset-mirror/samples/<dataset>/<file>.npz
  ```
  Record the **exact URL and filename** — the naming is per-dataset and not derivable
  (`vww/vww_batch1024.npz`, `kws/kws_batch1024.npz`, `imagenet/imagenet_batch1024_160.npz`,
  `voc/voc20_384_batch1024.npz`, `eye_tracking/eye_tracking_bs100.npz`, …). Also note any
  `-e` (calibration epochs) / `-bs` (calibration batch size) arguments the source passes to
  `quantizeml quantize`.

  If the source has no such `wget`, or it is present but commented out (as in
  `detection/train_voc.sh`), **stop and ask the user** for the sample pack to use. Do not
  silently fall back to random calibration.

- **Any notebooks** in `SOURCE_DIR` — read their content if they exist.

---

## Step 2 — Read the reference example

Read every file in `akida2/model_zoo/vww/` — it is the pattern each generated file follows:

- `vww_model.py`, `vww_data.py`, `vww_train.py`, `vww_eval.py`, `vww_benchmark.py`
- `vww_train.sh`, `update_readme.py`, `colab_setup.py`
- `vww_notebook_training.ipynb`, `vww_notebook_benchmark.ipynb`, `vww_notebook.py`
- `docs/README.md.template`, `docs/metrics.json`

Also read the top-level `pyproject.toml` to see which packages are already available
(`quantizeml`, `cnn2snn`, `akida_models`, …) versus needing a note.

Akida 2 status note: the reference hardware is an **FPGA at 25 MHz**; AKD2500 production
silicon does not exist yet. The software pipeline (train → quantize → convert →
software-backend eval → sparsity) is fully real, but the hardware benchmark is FPGA-based and
**latency-only** (the power path is still under development). Do not fabricate power numbers
or a production clock.

Reference clock frequencies for Akida devices/platforms (FPGA and AKD2500 included) live in
`brainchip_utils.hardware_utils.AKIDA_CLOCKS_HZ` — a single dict shared across model zoo
examples. Import and use it rather than hardcoding clock values (see 3e).

---

## Step 3 — Generate target files

Create `TARGET_DIR` and the files below. For each, the VWW file is the direct structural
template — preserve every pattern and only substitute `NAME`-specific content.

### 3a. `<NAME>_model.py`

The model definition is **ported from the source example, not designed or derived**. The
source normally includes the model-building code — lift it rather than reasoning about the
architecture.

- Locate the model-definition code in the source (the function/section that builds and
  returns the Keras model) and copy it into `<NAME>_model.py` essentially verbatim.
- Adapt only the mechanical wrappers, never the architecture:
  - imports,
  - the builder function name → `build_<NAME>_model(...)`,
  - the `set_akida_version(AkidaVersion.v2)` context,
  - a `-s/--savepath` CLI (default `./models/<short_model_name>_<NAME>[_untrained].h5`; use
    the `_untrained` suffix if the source builds from scratch, omit it if the source starts
    from pretrained weights) and `model.save(..., include_optimizer=False)`,
  - `--seed` + `set_random_seed(args.seed)` if the source doesn't already seed.
- **Preserve whatever the source does** — pretrained backbone vs. `include_top`, custom head,
  `input_scaling`, `rescale`, layer names, input dtype. Do not add, remove, or "improve"
  layers. The source is authoritative. Input handling (e.g. uint8 input with on-graph
  scaling) is carried over exactly as the source has it.
- The `akida_models` factories are version-aware inside the `set_akida_version(AkidaVersion.v2)`
  context, so that context is the whole version switch — there is no per-layer
  v2-compatibility work.

`akida2/model_zoo/vww/vww_model.py` shows what a lifted-and-adapted result looks like, but the
source example — not this reference — dictates the architecture.

### 3b. `<NAME>_data.py`

Follow the reference `vww_data.py` structure. Key rules:
- Expose exactly two public functions: `get_data(...)` and `get_samples(data_path,
  input_shape, num_samples=1024)`. `get_samples` **always** returns a `np.ndarray` of
  `dtype=uint8` (required by the benchmark utilities), regardless of dataset type.
- `get_data`'s **signature and return arity follow the dataset**, not a fixed shape:
  - Directory dataset → `get_data(data_path, input_shape, batch_size, seed=42)` returning
    `(train_dataset, val_dataset)`.
  - TFDS dataset → `get_data(data_path, input_shape, batch_size, dtype=tf.uint8, seed=42)`
    returning `(train_dataset, val_dataset, test_dataset)`.
  Match whichever the source/dataset naturally provides, and make sure `<NAME>_train.py` /
  `<NAME>_eval.py` unpack the matching number of values. Include a `seed=` parameter and call
  `set_random_seed(seed)`.
- Adapt the internals to the dataset **type** discovered in Step 1. The two real cases:
  - **Directory dataset** (`train/`+`val/` subdirs): `ImageDataGenerator` +
    `flow_from_directory` with `class_mode='sparse'`.
  - **TFDS dataset**: `tfds.load(name, split=[...], as_supervised=True, data_dir=data_path)`.
    Split strings define the partition (e.g. `['train[:80%]','train[80%:90%]','train[90%:]']`);
    map a `resize_and_cast` that resizes to `input_shape` and casts to `dtype` (uint8 by
    default); augment only the train split; call `tfds.disable_progress_bar()`. If the source
    overrides the dataset download URL to a BrainChip mirror
    (`..._dataset_builder._URL = "https://data.brainchip.com/dataset-mirror/..."`), replicate
    that override.
  - (No `.npz` branch — if a genuinely different source format appears, adapt from the closest
    reference.)
- Preserve augmentation from the source where it applies (spatial augmentation for images;
  none for spectrograms/1D signals unless the source does it). In the TFDS case, augment only
  the training dataset and cast back to the model's input dtype afterward.
- `get_samples()` mechanics follow the dataset type: directory examples glob/sample image
  files; TFDS examples do `tfds.load(split=...).take(num_samples)`, resize, and stack. Both
  return uint8 arrays.
- Default `data_path` should be `./data/<dataset_dir_name>`.

### 3c. `<NAME>_train.py`

Follow the reference `vww_train.py` structure. Key rules:
- The training function is named `train_<NAME>(model, train_ds, val_ds, epochs,
  learning_rate, regularization=None, seed=42)`.
- Use `SparseCategoricalCrossentropy(from_logits=True)` for sparse integer labels. Use
  `CategoricalCrossentropy` only if the source uses one-hot labels.
- Optimizer is `Adam` (legacy: `tf_keras.optimizers.legacy.Adam`).
- **LR schedule: match the source — do not mandate one.** (The v1 references use different
  schedules — `CosineDecay` with warmup, or exponential decay via a `get_custom_scheduler()`
  helper — so there is no single correct one.) Pick whichever fits the source's intent and
  wrap it in a scheduler callback. Do NOT invent a step-decay schedule.
- **Do not add a `RestoreBest` callback** — the references don't use one (one even has it as a
  dead import; don't propagate that).
- The optional activity-regularization path (`-reg` adding `L1L2` on `ReLU` layers to increase
  sparsity) is real — carry it over as in the reference.
- Carry over `tf.config.experimental.enable_op_determinism()` at import time if the
  source/reference does (small throughput cost, aids reproducibility).
- CLI: `-l`, `-s`, `-d`, `-b`, `-e`, `-lr`, `-reg`, `--seed`, with epoch/LR defaults from the
  source.
- Unpack `get_data(...)` with the arity the data module provides (2 or 3 values — see 3b).
- `<NAME>_train.py` is also used for the 4-bit QAT fine-tuning step (see 3f); the training
  loop is the same, it just loads a quantized model to fine-tune. `load_quantized_model` loads
  both float and quantizeml-quantized `.h5` files.

### 3d. `<NAME>_eval.py`

Copy the reference `vww_eval.py` and substitute only:
- `vww_data` → `<NAME>_data`.
- The default `--data` path.
- The `evaluate_akida_model` function is identical; copy verbatim.
- `--save-metrics` uses the **variant-keyed** scheme — copy it verbatim from
  `akida2/model_zoo/vww/vww_eval.py`. It writes `float_acc`/`params` for the float model and
  `<variant>_quant_acc` (from `.h5`) / `<variant>_akida_acc` (from `.fbz`) per stored variant,
  with the variant inferred from the filename (`i8_w8_a8` → `w8a8`; `i8_w4_a4` → `w4a4_qat`).

### 3e. `<NAME>_benchmark.py`

Copy the reference `vww_benchmark.py` and substitute only:
- `from vww_data import get_samples` → `from <NAME>_data import get_samples`.
- The default `--data` path.
- All `brainchip_utils` imports, benchmark calls, and plotting calls are identical.

Akida 2 benchmark specifics (already in the reference — preserve them):
- Import `AKIDA_CLOCKS_HZ` from `brainchip_utils.hardware_utils` and use
  `MEASURED_CLOCK = AKIDA_CLOCKS_HZ['AKIDA2_FPGA']` (25 MHz) rather than hardcoding the value.
- A **projected latency** at a higher clock: `projected_ms = mean_inf_clk / PROJECTED_CLOCK *
  1000` (cycle count is clock-independent, so this is exact), with
  `PROJECTED_CLOCK = AKIDA_CLOCKS_HZ['AKD2500']` (1 GHz — BrainChip's current target for
  AKD2500 production silicon). AKD2500 silicon does not exist yet, so this is a target, not a
  measured value, and it may change — but it is not a per-example placeholder to invent; it
  comes from the shared dict.
- **Latency-only**: no power measurement (the FPGA power path is WIP); no power columns/keys.
- `--save-metrics` writes variant-keyed metrics: `<variant>_sparsity`, and per map-mode
  `<variant>_<mode>_{nps,passes,cycles,latency_ms,projected_ms}`, with `<variant>` ∈
  {`w8a8`, `w4a4_qat`} inferred from the `.fbz` filename.

### 3f. `<NAME>_train.sh`

The pipeline. Follow `akida2/model_zoo/vww/vww_train.sh`. Epoch counts and LR values come
from the source (read in Step 1); the reference provides structural shape only.

Data-path forwarding at the top:
```bash
DATADIR="${1:-}"
DATA_ARG=${DATADIR:+-d "$DATADIR"}
```

Immediately after it, the calibration-samples download, using the URL recorded in Step 1:
```bash
# Download batch of samples for calibration
wget -N <SAMPLES_URL> \
     -P data/
```
Note the `-P data/`: the source scripts drop the `.npz` in the working directory, but devhub
examples keep it inside the example's own `data/` folder alongside the dataset. `-N` makes
the download idempotent across reruns.

Quantization uses **`quantizeml`** (not `cnn2snn quantize`). The **input layer weights are
always 8-bit** (`-i 8`), regardless of the target precision. Both quantize calls pass the
downloaded samples via `--samples`. Two quantized variants are produced:

| Variant | quantize command | QAT? |
|---|---|---|
| 8-bit | `quantizeml quantize -m <float>.h5 -i 8 -w 8 -a 8 -s <name>_i8_w8_a8.h5 --samples data/<samples>.npz` | No — 8-bit PTQ is accurate enough |
| 4-bit | `quantizeml quantize -m <float>.h5 -i 8 -w 4 -a 4 -s <name>_i8_w4_a4_pretmp.h5 --samples data/<samples>.npz`, then QAT-fine-tune | Yes — **QAT only** |

Pipeline order:
1. Build untrained/starting model: `python <NAME>_model.py -s models/<short>_<NAME>[_untrained].h5`
2. Float-train → `models/<short>_<NAME>.h5`, then `python <NAME>_eval.py -l ...<NAME>.h5 $DATA_ARG`
3. **8-bit**: `quantizeml quantize ... -i 8 -w 8 -a 8 -s ...i8_w8_a8.h5 --samples data/<samples>.npz`
   → eval `.h5` → `cnn2snn convert -m ...i8_w8_a8.h5` → eval `.fbz` →
   `python <NAME>_benchmark.py -l ...i8_w8_a8.fbz $DATA_ARG`
4. **4-bit (QAT only)**: `quantizeml quantize ... -i 8 -w 4 -a 4 -s ...i8_w4_a4_pretmp.h5 --samples data/<samples>.npz`
   → `python <NAME>_train.py -l ...i8_w4_a4_pretmp.h5 -s ...i8_w4_a4_qat.h5 -e <N> -lr <LR> $DATA_ARG`
   → eval `.h5` → `cnn2snn convert -m ...i8_w4_a4_qat.h5` → eval `.fbz` → benchmark `.fbz` →
   `rm -f models/<short>_<NAME>_i8_w4_a4_pretmp.h5`

Key points:
- **4-bit PTQ accuracy is poor** (as seen on Akida 1). Do not store, evaluate, convert, or
  record metrics for the 4-bit PTQ model — it is a throwaway on the way to QAT. Write it to a
  clearly-temporary `_pretmp.h5` name and `rm` it at the end.
- `quantizeml` has **no `convert` subcommand**. Conversion to `.fbz` is always
  `cnn2snn convert -m <model>.h5` (it accepts quantizeml-quantized models). Output filename is
  `<input_stem>.fbz`.
- **Real calibration samples are mandatory, not a per-example decision.** `quantizeml
  quantize` calibrates during quantization; left to itself it calibrates on random data, which
  quantizeml itself warns is inaccurate for per-axis activation quantization. The `.sh`
  pipeline therefore always passes the published sample pack downloaded above. `-sa` and
  `--samples` are the same flag (the reference spells it `--samples`); `-ns/--num_samples` is
  ignored once samples are supplied. Do **not** use `--per_tensor_activations` as a substitute
  for real samples — it changes the quantization scheme rather than fixing the calibration.
- `-e` / `-bs` (calibration epochs and batch size) default to `1` / `None`. Carry over the
  source's values only if the source sets them explicitly.

Model naming (keep bit-widths explicit; 8-bit needs no suffix, 4-bit always carries `_qat`):
- Float: `<short>_<NAME>_untrained.h5` → `<short>_<NAME>.h5`
- 8-bit: `<short>_<NAME>_i8_w8_a8.h5` → `<short>_<NAME>_i8_w8_a8.fbz`
- 4-bit throwaway: `<short>_<NAME>_i8_w4_a4_pretmp.h5` (deleted at pipeline end)
- 4-bit QAT: `<short>_<NAME>_i8_w4_a4_qat.h5` → `<short>_<NAME>_i8_w4_a4_qat.fbz`

Float-training and QAT epoch/LR values are provisional if the source doesn't specify them —
mark them confirm-on-first-run.

### 3g. `update_readme.py`

Copy verbatim from `akida2/model_zoo/vww/update_readme.py`. No example-specific content.

### 3h. `docs/README.md.template`

Follow `akida2/model_zoo/vww/docs/README.md.template` exactly (same sections, same order),
rewriting content for this dataset/model. Sections in order:

1. Logo image line — copy verbatim.
2. `# <DISPLAY_NAME>` — readable title.
3. `## Model Card` — the model card and benchmark tables:
   - **Model card**: rows-per-variant. One row per stored quantized variant (8-bit,
     4-bit QAT), columns `Variant | Weights/Acts | QAT | Quantized acc. | Akida acc. | Sparsity`.
     The 8-bit row shows `-` in the QAT column; the 4-bit row shows `yes`. Float accuracy +
     params are stated once above the table.
   - **Benchmark table**: the two stored variants × {Minimal, AllNps} = 4 rows, latency-only,
     with `Latency @ 25 MHz` and `Projected @ 1000 MHz (target)` columns (values from
     `AKIDA_CLOCKS_HZ`, see Step 2). No power columns.
   - A short architecture description.
4. `## Requirements` — copy from the reference; add example-specific deps if any.
5. `## Dataset` — describe the dataset from Step 1.
6. `## Dataset setup` — download/obtain instructions; include a URL + wget/extract if known,
   else a `<!-- TODO -->`. End the section with a short **calibration samples** paragraph:
   give the mirror URL of the `.npz`, and state that `<NAME>_train.sh` fetches it into `data/`
   automatically, while the training notebook instead builds its samples from the dataset with
   `get_samples()` (see 3m).
7. `## Pipeline` — copy the pipeline table from the reference, adapting the quantization rows
   to this example. Both quantization rows must say that quantization is calibrated on real
   samples.
8. `## Usage` → two subsections:
   - `### Notebook` — link the two notebooks (`<NAME>_notebook_training.ipynb` and
     `<NAME>_notebook_benchmark.ipynb`) with a short description of each, and place the
     **Colab badge** immediately after the training-notebook description (see 3m). Add the
     hardware-benchmark note after the benchmark-notebook description.
   - `### Script` — the `bash <NAME>_train.sh [DATADIR]` intro, adapting filenames.
9. `## Contributing and Maintenance` — copy from the reference, substituting `<NAME>` and the
   per-variant `--save-metrics` command list (float + 8-bit + 4-bit-QAT).

### 3i. `docs/metrics.json`

Every `{key}` in the template must appear here or `update_readme.py` crashes. Take the key
set from `akida2/model_zoo/vww/docs/metrics.json` and adapt to any keys you introduced. The
reliable method: regex-extract all `{...}` placeholders from the template you wrote and emit
exactly that set, each `"TBD"`.

**Verify the three-way bijection**: the template placeholders, the metrics.json keys, and the
union of keys the eval + benchmark `--save-metrics` blocks write must be **equal** — no extras
on any side. Check programmatically (regex the template, compare to the script-written set),
not by eye. Then run `update_readme.py` — it must render with no `KeyError`.

### 3j. `README.md`

Generate by running `update_readme.py`. The result has `"TBD"` wherever metrics go — correct
and expected until training runs complete.

### 3l. `colab_setup.py`

A module with a single `setup()` function that makes the "Open in Colab" badge work; it is a
no-op on local runs. Follow `akida2/model_zoo/vww/colab_setup.py`:
- Module constants: `REPO_URL`, `REPO_DIR = 'brainchip_devhub'`,
  `EXAMPLE_SUBDIR = 'akida2/model_zoo/<NAME>'`, and (for a directory dataset) `DATA_DIR` +
  `DATASET_URL`.
- `setup()` returns immediately with a friendly message if `'google.colab' not in sys.modules`
  — this is what makes it safe to leave the notebook's first cell in permanently.
- On Colab: clone the repo with `GIT_LFS_SKIP_SMUDGE=1` (pretrained weights aren't needed for
  the default train-from-scratch path), `os.chdir` into `EXAMPLE_SUBDIR`, put the repo root and
  the example dir on `sys.path`, then `pip install -q akida_models==1.14.0 tf_keras quantizeml`
  (add any extra imports the notebook/modules need — check by grep; e.g. `pooch` for some
  examples, not VWW).
- **Dataset handling depends on the data type** (from Step 1):
  - **Directory dataset** (VWW): include a `wget` + `tar -xzf` block guarded by
    `if not os.path.exists(DATA_DIR)` (needs a public `DATASET_URL`).
  - **TFDS dataset**: omit the download block entirely — the notebook's data cell auto-downloads
    via `tfds.load`.
- End with a note to restart the runtime if TensorFlow was just (re)installed.

### 3m. Notebooks

Generate two notebooks plus a Jupytext mirror, following the `akida2/model_zoo/vww/`
notebooks:

- **`<NAME>_notebook_training.ipynb`** — the training walkthrough. Cell structure:
  1. Markdown: logo (absolute `raw.githubusercontent.com` URL) + title + overview.
  2. Code: the Colab-only setup cell — `if 'google.colab' in sys.modules:` → `wget`
     `colab_setup.py` (from `akida2/model_zoo/<NAME>/`) if absent → `import colab_setup;
     colab_setup.setup()`.
  3. Setup (imports, `DATA_PATH`, `MODELS_DIR`, `SEED`, `RUN_FLOAT_TRAINING = True`,
     `enable_op_determinism()`).
  4. Dataset (`get_data`, unpacking the right arity), Model (`build_<NAME>_model`), Float
     training (train from scratch), evaluate float.
  5. **Quantization with `quantizeml`** — two variants: 8-bit
     `QuantizationParams(input_weight_bits=8, weight_bits=8, activation_bits=8)` → `quantize`
     → eval; and 4-bit `(…weight_bits=4, activation_bits=4)` → `quantize` → **QAT fine-tune
     via `train_<NAME>`** → eval. (Do not use `cnn2snn.quantize` — that is Akida 1.)

     **Calibration samples are generated in the notebook, not downloaded.** This is a
     deliberate difference from `<NAME>_train.sh`, which uses the published `.npz`: the
     notebook is teaching material, so the reader sees how samples are built for their own
     use case. Concretely, in the first quantization cell:
     ```python
     from quantizeml.models import quantize, QuantizationParams

     from <NAME>_data import get_samples

     NUM_SAMPLES = 1024
     samples = get_samples(DATA_PATH, INPUT_SHAPE, num_samples=NUM_SAMPLES)
     ```
     Import `quantize` **and** `QuantizationParams` from `quantizeml.models` — a single
     import; do not import `QuantizationParams` from `quantizeml.layers`. Pass
     `samples=samples` to **both** `quantize(...)` calls. Precede the cell with a markdown
     note that `quantizeml` quantization requires calibration samples, and that these should
     be representative samples for the task drawn from the **training** split, to avoid data
     leakage.
  6. Conversion: `cnn2snn.convert` for both variants → `.fbz`.
  7. Akida software-backend eval for both variants; activation sparsity for both. The sparsity
     section **reuses the `samples` array already built for calibration** — do not re-import
     `get_samples` or rebuild the array, and say in the markdown that the calibration samples
     are being reused (sparsity needs real activations, same as calibration).
  8. Summary table (float vs 8-bit vs 4-bit-QAT).
  Train from scratch by default (`RUN_FLOAT_TRAINING = True`); do not add a pretrained-load
  fast path unless committed pretrained models exist.
- **`<NAME>_notebook_benchmark.ipynb`** — accuracy + hardware benchmark. Device-guarded
  (`get_akida_device` → `None` when absent → skip latency, still compute sparsity on the
  software backend). Latency-only, `MEASURED_CLOCK = 25e6` + provisional `PROJECTED_CLOCK`.
  This notebook is **not** Colab-fied (needs hardware) — no setup cell, no badge.
- **`<NAME>_notebook.py`** — the Jupytext `py:percent` mirror of the training notebook.
  Generate with `jupytext --to py:percent --opt comment_magics=false` (so the `!wget` magic
  stays uncommented), then remove the `comment_magics: false` line from the header for
  consistency. Verify it round-trips back to the `.ipynb` with no cell mismatches.

Validate every generated notebook with `nbformat.validate` and give each cell an `id`.

### 3n. Directory stubs

Create `models/` and `data/` each containing a `.gitignore` that ignores everything except
itself (copy verbatim from the reference — it is not a `.gitkeep`):
```
# Git to Ignore everything in this directory
*
# Except this .gitignore file
!.gitignore
```

---

## Step 4 — Report

Summarise:
1. Files created (paths).
2. TODOs left for the user (dataset URL/hash, the provisional epoch/LR values). The projected
   clock is not a per-generation TODO — it comes from `AKIDA_CLOCKS_HZ`.
3. Verification commands:
   ```bash
   cd TARGET_DIR
   python -c "import ast; [ast.parse(open(f).read()) for f in ['<NAME>_model.py','<NAME>_data.py','<NAME>_train.py','<NAME>_eval.py','<NAME>_benchmark.py']]"
   bash -n <NAME>_train.sh
   python update_readme.py   # must render with no KeyError
   python -c "import nbformat; [nbformat.validate(nbformat.read(f, as_version=4)) for f in ['<NAME>_notebook_training.ipynb','<NAME>_notebook_benchmark.ipynb']]"
   wget --spider -q <SAMPLES_URL> && echo "samples URL OK"
   ```
   Also confirm the `--save-metrics` key set matches the template exactly (extract `{...}`
   placeholders from the template, compare against the union of keys the eval + benchmark
   scripts write — the two sets must be equal).
4. A reminder that the pipeline was never executed — all accuracy/latency numbers in
   `docs/metrics.json` and the README are `"TBD"` until the user runs the real pipeline and
   the `--save-metrics` maintenance commands.

---

## Key invariants

- `get_samples()` must always return `np.ndarray` of `dtype=uint8` — required by
  `per_layer_benchmark` and `full_model_benchmark`.
- The model definition is **ported from the source example, not designed or reconstructed**.
  Preserve the source's architecture exactly (backbone, head, `input_scaling`, layer names,
  input dtype); adapt only mechanical wrappers. The source is authoritative.
- The **input layer weights are always 8-bit** (`-i 8`), regardless of the rest of the
  model's target precision.
- Quantization is `quantizeml quantize`: an 8-bit `i8/w8/a8` variant (no QAT — PTQ is accurate
  enough) and a 4-bit `i8/w4/a4` variant (QAT only — 4-bit PTQ accuracy is poor, so the PTQ
  model is a throwaway, never stored/evaluated). Both convert to `.fbz` with the same
  `cnn2snn convert` — there is no `quantizeml convert`.
- **Quantization always calibrates on real samples — never random.** The two paths get their
  samples differently, and that split is intentional:
  `<NAME>_train.sh` downloads the published pack from the dataset mirror
  (`wget -N <url> -P data/`) and passes `--samples data/<file>.npz` to every quantize call;
  the training notebook builds its own with `samples = get_samples(DATA_PATH, INPUT_SHAPE,
  num_samples=1024)` and passes `samples=samples` to every `quantize(...)` call, so the
  reader can see how to produce samples for their own data. If no published pack exists for
  the source example, stop and ask — do not fall back to random calibration.
- Do not fabricate Akida 2 hardware numbers: AKD2500 production silicon does not exist yet;
  the reference platform is a 25 MHz FPGA, benchmarking is latency-only (no power). Both the
  FPGA (25 MHz) and AKD2500 target (1 GHz) clocks come from
  `brainchip_utils.hardware_utils.AKIDA_CLOCKS_HZ` — the single source of truth, since the
  AKD2500 value is a pre-production target that may still change. The software pipeline is
  fully real regardless.
- The metrics.json / template / `--save-metrics` key sets must be in exact three-way bijection
  (variant-prefixed keys). Verify programmatically, not by eye.
