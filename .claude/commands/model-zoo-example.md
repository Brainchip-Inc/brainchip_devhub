---
description: Generate a complete brainchip_devhub model zoo example from an akida_models source example
---

# Model Zoo Example Generator

Generate a self-contained model zoo example in `brainchip_devhub`, adapted from the source
scripts in `akida_models`. The new example follows the VWW reference example for the target
Akida version:
- **Akida 1** → `akida1/model_zoo/vww/` is the canonical reference.
- **Akida 2** → `akida2/model_zoo/vww/` is the canonical reference. Akida 2 differs from
  Akida 1 in several substantial ways (quantization tool, multi-variant model set, benchmark
  clock, README table shape). Those deltas are collected in the dedicated
  **"Akida 2 targets"** section below; when `AKIDA_VERSION == 2`, that section overrides the
  Akida-1-oriented instructions in Step 3 wherever they conflict.

## Usage

```
/model-zoo-example <name> [--akida-version 1|2]
```

- `<name>` — subdirectory name under `akida_models/scripts/` (e.g. `kws`, `mnist`, `face`)
- `--akida-version` — target hardware generation (default: `1`)

Parse these from `$ARGUMENTS` now. Set:
- `NAME` = the example name
- `AKIDA_VERSION` = `1` or `2` (default `1`)
- `SOURCE_DIR` = `akida_models/scripts/<NAME>/`
- `TARGET_DIR` = `brainchip_devhub/akida<AKIDA_VERSION>/model_zoo/<NAME>/`

Resolve the two repo locations before doing anything else, in this order:
1. Environment variables `BRAINCHIP_DEVHUB` and `AKIDA_MODELS`, if set.
2. Paths passed as `--devhub <path>` / `--akida-models <path>` in `$ARGUMENTS`.
3. Sibling directories of the current working directory (e.g. if invoked inside a
   `brainchip_devhub` checkout, look for `../akida_models`).
4. Search upward from the current directory for folders named `brainchip_devhub` and
   `akida_models`.

If neither repo can be found, stop and ask for the paths rather than guessing. Do not
hardcode any absolute user path.

---

## Step 1 — Read source scripts

Read every `.py` and `.sh` file under `SOURCE_DIR`. For each file, identify:

- **Model file**: the script that instantiates the model architecture (look for calls to
  `akidanet_imagenet`, `ds_cnn_kws`, `mobilenet`, or similar `akida_models` factory functions).
  Record the function name, all arguments (input_shape, classes, alpha, etc.), and any
  imports from `akida_models`.

- **Data file**: the script that loads the dataset. Record the dataset name, directory
  structure (does it have `train/`+`val/` subdirs, or a `.npz` file, or something else?),
  input resolution, any augmentation applied, and whether data is uint8 or float.

- **Training file**: the script that trains the model. Record the loss function, optimizer,
  number of epochs, learning rate schedule, and any callbacks.

- **Transfer learning**: check if the model script loads pretrained weights (look for
  `fetch_file`, `.load_weights(`, `by_name=True`). If so, record:
  - The pretrained weights URL from `fetch_file`
  - Whether `include_top=False` is used on the base model
  - All custom top layers added after the base model (layer types, units, names)
  - Any `get_params_by_version()` calls for version-specific activations

- **Pipeline structure**: read the source `.sh` file as the authoritative pipeline
  reference. Record: the number of training/fine-tuning epochs, the LR used, whether the
  training action is full training from scratch or fine-tuning from a pretrained starting
  point, any multi-phase training (e.g. initial float tune followed by QAT). The VWW
  template provides structural patterns only; all epoch counts, LR values, and training
  modes come from the source.

- **Quantization**: look in `.sh` files for the `quantizeml quantize` or `cnn2snn quantize`
  command. Note which tool is used and the `-i`, `-w`, `-a` bit-width arguments. This is for
  reference only — for **Akida 1** targets, quantization is **always**
  `cnn2snn quantize -i 8 -w 4 -a 4` and QAT is always required, regardless of what the
  source uses (the source may target Akida 2 with `quantizeml`).

- **Any notebooks** in `SOURCE_DIR` — read their content if they exist.

---

## Step 2 — Read the canonical VWW reference

Read the VWW reference files for the target Akida version — they are the pattern every
generated file must follow. Let `REF = akida<AKIDA_VERSION>/model_zoo/vww/`.

- `REF/vww_model.py`
- `REF/vww_data.py`
- `REF/vww_train.py`   *(present for Akida 1; for Akida 2 confirm whether it exists —
  the v2 reference may reuse the v1 training loop; see the Akida 2 section)*
- `REF/vww_eval.py`
- `REF/vww_benchmark.py`
- `REF/vww_train.sh`
- `REF/update_readme.py`
- `REF/docs/README.md.template`
- `REF/docs/metrics.json`

For **Akida 2**, read the `akida2/model_zoo/vww/` versions specifically — do not assume they
match the Akida 1 files. The differences are the whole point of the Akida 2 section below.
Also read the top-level `pyproject.toml` so you know which packages are already available
(`quantizeml`, `cnn2snn`, `akida_models`, etc.) versus needing a note.

---

## Step 3 — Generate target files

Create `TARGET_DIR` and the files below. For each, the VWW file is the direct structural
template — preserve every pattern and only substitute `NAME`-specific content.

### 3a. `<NAME>_model.py`

Follow `vww_model.py` exactly. Key rules:
- The build function is named `build_<NAME>_model()`.
- Wrap the model instantiation in `with set_akida_version(AkidaVersion.v<AKIDA_VERSION>):`.
- Use the architecture factory and arguments discovered in Step 1.
- If `input_scaling=(255, 0)` is appropriate for this model (i.e. the model expects uint8
  inputs), include it. Include it by default unless the source explicitly normalises inputs
  before feeding to the model.
- CLI: `-s/--savepath` defaulting to `./models/<short_model_name>_<NAME>_untrained.h5`
  (derive `short_model_name` from the architecture, e.g. `akidanet`, `dscnn`).
- Always include `--seed` and call `set_random_seed(args.seed)`.

For **Akida 2** targets, the model-construction pattern differs from the transfer-learning
recipe below — see the **"Akida 2 targets"** section (it builds directly from
`akidanet_imagenet(include_top=True)` rather than grafting a head onto a pretrained
backbone). Follow the v2 reference `akida2/model_zoo/vww/vww_model.py` and that section
instead of the transfer-learning block here.

**Transfer learning models (Akida 1)**: if Step 1 found that the source uses transfer learning,
generate `<NAME>_model.py` following the real `plant_village_model.py` / `vww_model.py`
pattern (they use the `_pretrained` factory helper, not a manual `fetch_file` +
`load_weights`):

- Create the pretrained base model in one call:
  `akidanet_imagenet_pretrained(alpha=<ALPHA>, quantized=False)` — this fetches the
  ImageNet-pretrained backbone weights internally (no separate `fetch_file`/`load_weights`).
- Tap the backbone at the appropriate feature layer:
  `x = base_model.get_layer('separable_13/relu').output` (plant_village), then add the
  source's custom head (e.g. `dense_block(units=..., add_batchnorm=True,
  relu_activation='ReLU6.0')`, `Dropout(...)`, a final `dense_block(units=classes,
  add_batchnorm=False, relu_activation=False)`). Copy the head layer structure and names
  from Step 1.
- **Input rescaling is conditional on resolution:**
  - If the source changes the input resolution from the backbone's native size (VWW: 96×96),
    apply `rescale(model, (H, W))` from `akida_models.imagenet.imagenet_train` after building.
  - If the source keeps the backbone's native resolution (plant_village: 224×224), do **not**
    call `rescale()` — the `akidanet_imagenet_pretrained` model already handles input scaling
    (uint8 → /255) internally, so the preprocessing pipeline must deliver raw uint8.
- Build with `tf_keras.Model(base_model.input, x, name='<name>')`.
- Wrap the construction in `with set_akida_version(AkidaVersion.v1):`.
- Default save path: `./models/<short_model_name>_<NAME>.h5` (no `_untrained` suffix —
  the backbone is already pretrained; this file is the fine-tuning starting point).
- Add imports for `tf_keras.Model`, `akidanet_imagenet_pretrained`, `dense_block`, and any
  layer types used in the head (e.g. `Dropout`).

(For **Akida 2** the model is built differently — from `akidanet_imagenet(include_top=True)`
with no pretrained-backbone graft; see the Akida 2 section.)

### 3b. `<NAME>_data.py`

Follow the reference `vww_data.py` / `plant_village_data.py` structure. Key rules:
- Expose exactly two public functions: `get_data(...)` and `get_samples(data_path,
  input_shape, num_samples=1024)`. `get_samples` **always** returns a `np.ndarray` of
  `dtype=uint8` (required by the benchmark utilities), regardless of dataset type.
- `get_data`'s **signature and return arity follow the dataset**, not a fixed shape:
  - VWW (directory) → `get_data(data_path, input_shape, batch_size, seed=42)` returning
    `(train_dataset, val_dataset)` — **two** datasets.
  - plant_village (TFDS) → `get_data(data_path, input_shape, batch_size, dtype=tf.uint8,
    seed=42)` returning `(train_dataset, val_dataset, test_dataset)` — **three** datasets.
  Match whichever the source/dataset naturally provides, and make sure `<NAME>_train.py` /
  `<NAME>_eval.py` unpack the matching number of values. Both reference examples take a
  `seed=` parameter and call `set_random_seed(seed)` — include it.
- Adapt the internals to the dataset **type** discovered in Step 1. The two real cases are:
  - **Directory dataset** (has `train/` + `val/` subdirs, e.g. VWW): use `ImageDataGenerator`
    + `flow_from_directory` with `class_mode='sparse'`, exactly as in `vww_data.py`.
  - **TFDS dataset** (loaded via `tensorflow_datasets`, e.g. plant_village): use
    `tfds.load(name, split=[...], as_supervised=True, data_dir=data_path)`. Follow
    `plant_village_data.py`:
    - Split strings define the train/val/test partition (e.g.
      `['train[:80%]', 'train[80%:90%]', 'train[90%:]']`).
    - Map a `resize_and_cast(image, label)` that resizes to `input_shape` and casts to the
      requested `dtype` (uint8 by default).
    - Apply augmentation only to the train split (see below); call
      `tfds.disable_progress_bar()` to keep output clean.
    - If the source overrides the dataset download URL to a BrainChip mirror (plant_village
      sets `..._dataset_builder._URL = "https://data.brainchip.com/dataset-mirror/..."`),
      replicate that override — it's how the dataset is fetched without the upstream source.
  - (No `.npz` branch — neither reference example uses one. If a genuinely different source
    format appears, adapt from the closest reference rather than assuming npz.)
- Preserve augmentation from the source where it applies (spatial augmentation for images;
  none for spectrograms/1D signals unless the source does it). In the TFDS case, augment only
  the training dataset, cast back to the model's input dtype afterward.
- `get_samples()` mechanics also follow the dataset type: directory examples glob/sample
  image files; TFDS examples do `tfds.load(split=...).take(num_samples)`, resize, and stack.
  Both return uint8 arrays.
- Default `data_path` should be `./data/<dataset_dir_name>`.

### 3c. `<NAME>_train.py`

Follow the reference `<NAME>_train.py` structure (`vww_train.py` or
`plant_village_train.py`). Key rules:
- The training function is named `train_<NAME>(model, train_ds, val_ds, epochs,
  learning_rate, regularization=None, seed=42)`.
- Use `SparseCategoricalCrossentropy(from_logits=True)` for sparse integer labels (the
  standard case). Use `CategoricalCrossentropy` only if source uses one-hot labels.
- Optimizer is `Adam` (legacy: `tf_keras.optimizers.legacy.Adam`) in both references.
- **LR schedule: match the source — do not mandate one.** The two reference examples
  deliberately use *different* schedules, so there is no single "correct" one to copy:
  - VWW: `CosineDecay` (from `tf_keras.optimizers.schedules`) with ~10% warmup.
  - plant_village: exponential decay to ~1% of the initial LR over `n_epochs`, implemented
    in a `get_custom_scheduler()` helper wrapping a `LearningRateScheduler` callback.
  Pick whichever fits the source's intent and wrap it in a scheduler callback. Do NOT invent
  a step-decay schedule (that was an older, inaccurate instruction).
- **Do not add a `RestoreBest` callback.** Neither reference uses it in the training loop.
  Note that `plant_village_train.py` *imports* `RestoreBest` but never uses it — that is a
  dead import; do not propagate it, and do not add the callback.
- The optional activity-regularization path (the `-reg` flag adding `L1L2` on `ReLU`
  layers to increase sparsity) is real — carry it over as in the reference.
- Some references call `tf.config.experimental.enable_op_determinism()` at import time for
  reproducibility (plant_village does; it has a small throughput cost). Carry it over if the
  source/reference does.
- CLI: `-l`, `-s`, `-d`, `-b`, `-e`, `-lr`, `-reg`, `--seed` with the same defaults as the
  reference except adapt epochs and LR defaults to match the source.
- Unpack `get_data(...)` with the arity the data module provides (2 or 3 values — see 3b).
- For **Akida 2**, `<NAME>_train.py` is also used for the QAT fine-tuning of the 4-bit
  variant (see the Akida 2 section); the training loop itself is version-agnostic.

### 3d. `<NAME>_eval.py`

Copy the reference `vww_eval.py` and make the following substitutions only:
- Replace all occurrences of `vww_data` → `<NAME>_data`.
- Replace the default `--data` path to match the dataset for this example.
- The `evaluate_akida_model` function is identical; copy verbatim.
- **Akida 1** `--save-metrics`: keep the reference's filename-based key inference
  (`float_acc`, `qat_acc`, `akida_acc`, `params`).
- **Akida 2** `--save-metrics`: the key scheme is variant-keyed, not the single-QAT scheme —
  copy it from `akida2/model_zoo/vww/vww_eval.py` verbatim (see the Akida 2 section for why:
  three quantized variants each need their own keys).

### 3e. `<NAME>_benchmark.py`

Copy the reference `vww_benchmark.py` and make the following substitutions only:
- Replace `from vww_data import get_samples` → `from <NAME>_data import get_samples`.
- Replace the default `--data` path to match the dataset for this example.
- All `brainchip_utils` imports, benchmark calls, and plotting calls are identical; do not
  change them.
- **Akida 1**: the reference hardcodes `CLOCK_FREQUENCY = 400e6` (AKD1500) inline and
  measures latency + power. Keep as-is.
- **Akida 2**: the benchmark differs materially (25 MHz FPGA clock, a projected-latency
  computation, latency-only with the power path removed, and variant-keyed `--save-metrics`).
  Follow `akida2/model_zoo/vww/vww_benchmark.py` and the Akida 2 section — do not just swap
  the clock constant.

### 3f. `<NAME>_train.sh`

The source `train.sh` (read in Step 1) defines the pipeline details. The VWW reference
provides the structural shape only (DATA_ARG forwarding, cnn2snn quantize/convert steps,
eval/benchmark steps). All epoch counts, LR values, and training modes come from the source.

Start with the same two-line data-path forwarding:
```bash
DATADIR="${1:-}"
DATA_ARG=${DATADIR:+-d "$DATADIR"}
```

Nine steps in order — read the source `train.sh` to fill in epochs and LR:

**Step 1 — Build starting model**:
- Transfer learning: `python <NAME>_model.py -s models/<short_model_name>_<NAME>.h5`
- From scratch: `python <NAME>_model.py -s models/<short_model_name>_<NAME>_untrained.h5`

**Step 2 — Train/fine-tune** (use epochs and LR from the source train.sh, not VWW defaults):
- Transfer learning: `-l models/<short>_<NAME>.h5 -s models/<short>_<NAME>.h5 -e <N> -lr <LR> $DATA_ARG`
  (load and save to the same path — the pretrained starting point is overwritten by the
  fine-tuned result)
- From scratch: `-l models/<short>_<NAME>_untrained.h5 -s models/<short>_<NAME>.h5 -e <N> -lr <LR> $DATA_ARG`

**Step 3**: `python <NAME>_eval.py -l models/<short_model_name>_<NAME>.h5 $DATA_ARG`

**Step 4 — Quantize (Akida 1 always)**: `cnn2snn quantize -m models/<short_model_name>_<NAME>.h5 -i 8 -w 4 -a 4`
This is fixed for all Akida 1 targets regardless of what the source train.sh uses.

**Step 5 — QAT**: use QAT epochs and LR from source if specified; default to 2 epochs at
LR 1e-4 if the source does not include a separate QAT fine-tuning step.

**Step 6**: `python <NAME>_eval.py -l models/<short_model_name>_<NAME>_qat.h5 $DATA_ARG`

**Step 7**: `cnn2snn convert -m models/<short_model_name>_<NAME>_qat.h5`

**Step 8**: `python <NAME>_eval.py -l models/<short_model_name>_<NAME>_qat.fbz $DATA_ARG`

**Step 9**: `python <NAME>_benchmark.py -l models/<short_model_name>_<NAME>_qat.fbz $DATA_ARG`

Model naming convention:
- Transfer learning: `<short>_<NAME>.h5` → `<short>_<NAME>_iq8_wq4_aq4.h5` → `<short>_<NAME>_qat.h5` → `<short>_<NAME>_qat.fbz`
- From scratch: `<short>_<NAME>_untrained.h5` → `<short>_<NAME>.h5` → `<short>_<NAME>_iq8_wq4_aq4.h5` → `<short>_<NAME>_qat.h5` → `<short>_<NAME>_qat.fbz`

### 3g. `update_readme.py`

Copy the file verbatim from `akida1/model_zoo/vww/update_readme.py`. It has no VWW-specific
content.

### 3h. `docs/README.md.template`

Write a full README template for the new example. Follow the VWW template structure exactly
(same sections in the same order), but rewrite the content for this dataset and model:

**Sections (in order):**

1. **Logo image line** — copy verbatim:
   `<img src="../../../docs/assets/0.-BC-dev-hub-LOGO-flicker.svg" alt="BrainChip Dev Hub" width="200"/>`

2. **`# <DISPLAY_NAME>`** — use a readable title (e.g. "Keyword Spotting (KWS)").

3. **`## Dataset`** — describe the dataset from what you read in Step 1. Include:
   the name, what it contains, the classes (or number of classes), input resolution,
   and approximate split sizes if known.

4. **`## Model`** — the performance table (copy the HTML table structure verbatim from VWW,
   using the same `{float_acc}`, `{qat_acc}`, `{akida_acc}`, `{sparsity}`, `{params}`
   placeholder keys). Then the hardware benchmark table:

   For **Akida 1**: use the exact same AKD1500 benchmark HTML table as VWW, with the same
   placeholder keys: `{minimal_nps}`, `{minimal_passes}`, `{minimal_cycles}`,
   `{minimal_latency_ms}`, `{minimal_total_P}`, `{minimal_total_E}`, `{minimal_dyn_P}`,
   `{minimal_dyn_E}`, `{allnps_nps}`, `{allnps_passes}`, `{allnps_cycles}`,
   `{allnps_latency_ms}`, `{allnps_total_P}`, `{allnps_total_E}`, `{allnps_dyn_P}`,
   `{allnps_dyn_E}`. Include the two `<img>` lines for `ref_benchmark_results_full.png`
   and `ref_benchmark_results_layers.png`.

   For **Akida 2**: the model card and benchmark table have a **different shape** — a
   rows-per-variant model card (one row per quantized variant) and a latency-only benchmark
   table with measured + projected latency columns and no power columns. Do not reuse the
   Akida 1 table. Follow `akida2/model_zoo/vww/docs/README.md.template` and the Akida 2
   section for the exact structure and key scheme.

   Then write a short description of the model architecture (name, key hyperparameters).

5. **`## Pipeline`** — copy the four-row pipeline table from VWW verbatim (the pipeline
   stages are the same for all Akida examples).

6. **`## Requirements`** — copy from VWW verbatim (versions are the same). If this example
   has additional dependencies (e.g. librosa for audio), add them.

7. **`## Dataset setup`** — describe how to obtain this dataset. If a URL is known from the
   source scripts, include it with the wget + extract commands. If not known, write
   `<!-- TODO: add dataset download instructions -->`.

8. **`## Usage`** — two subsections:
   - `### Notebook` — link the two notebooks (`<NAME>_notebook_training.ipynb` and
     `<NAME>_notebook_benchmark.ipynb`) with a short description of each, matching the
     reference template's wording. (Colab badges are intentionally omitted for now.)
   - `### Script` — copy the reference structure (the `bash <NAME>_train.sh [DATADIR]`
     intro), adapting filenames, epochs, and LR values.

9. **`## Contributing and Maintenance`** — copy verbatim from VWW, substituting `<NAME>`
   for `vww` and the model file paths accordingly.

### 3i. `docs/metrics.json`

Write a JSON file with every placeholder key from the template set to `"TBD"`:

```json
{
    "float_acc": "TBD",
    "qat_acc": "TBD",
    "akida_acc": "TBD",
    "sparsity": "TBD",
    "params": "TBD",
    "minimal_cycles": "TBD",
    "minimal_latency_ms": "TBD",
    "minimal_total_P": "TBD",
    "minimal_total_E": "TBD",
    "minimal_dyn_P": "TBD",
    "minimal_dyn_E": "TBD",
    "allnps_cycles": "TBD",
    "allnps_latency_ms": "TBD",
    "allnps_total_P": "TBD",
    "allnps_total_E": "TBD",
    "allnps_dyn_P": "TBD",
    "allnps_dyn_E": "TBD",
    "minimal_nps": "TBD",
    "minimal_passes": "TBD",
    "allnps_nps": "TBD",
    "allnps_passes": "TBD"
}
```

**Important:** every `{key}` in the template must appear in this JSON or `update_readme.py`
will crash. Audit the template you just wrote and add any extra keys you introduced. The
reliable way to do this: extract all `{...}` placeholders from the template with a regex and
emit exactly that set, each set to `"TBD"`. Verify by running `update_readme.py` — it must
render with no `KeyError`.

The **Akida 2** key set is different and larger (variant-prefixed keys like `w8a8_*`,
`w4a8_ptq_*`, `w4a8_qat_*`, plus projected-latency keys, and no power keys). Take it from
`akida2/model_zoo/vww/docs/metrics.json` and keep it in exact correspondence with both the
v2 template and the variant-keyed `--save-metrics` blocks in the v2 eval/benchmark scripts.

### 3j. `README.md`

Generate the initial README by running `update_readme.py` inline (read the template and
format it with the metrics dict). The result will have "TBD" everywhere metrics should be —
that is correct and expected until training runs are complete.

### 3k. Notebooks

The reference examples ship **two** notebooks plus a Jupytext mirror, not a single
`<NAME>_notebook.ipynb`:
- `<NAME>_notebook_training.ipynb` — the training→quantization→conversion→eval walkthrough.
- `<NAME>_notebook_benchmark.ipynb` — accuracy-on-Akida + hardware benchmark (requires a
  device; never runs in Colab).
- `<NAME>_notebook.py` — a Jupytext `py:percent` mirror paired with the notebooks (the
  notebook metadata declares `jupytext: formats: ipynb,py:percent`). Regenerate it from the
  notebook with `jupytext` rather than hand-writing it.

> **Deferred:** notebook generation is intentionally out of scope for the current version of
> this skill while the non-notebook pipeline is being validated. When implementing it, use
> the cell-structure guidance below as a starting point, but the **Akida 2** notebooks must
> reflect the multi-variant pipeline (three quantized models, `quantizeml` quantization, no
> QAT for 8-bit, FPGA benchmark) rather than the single-model Akida 1 flow — mirror
> `akida2/model_zoo/vww/` once its notebooks exist.

Cell-structure reference (Akida 1 single-model flow; use `nbformat` 4.5):

Cell sequence (each as a separate element in `"cells"`):

1. **Markdown** — title + one-paragraph overview of the example.
2. **Code** — imports and configuration flags:
   ```python
   import os, pathlib
   import numpy as np
   import akida
   import pooch
   # ... other imports discovered in Step 1
   
   RUN_FLOAT_TRAINING = False
   RUN_QAT_TRAINING = False
   DATA_PATH = './data/<dataset_dir_name>'
   MODELS_DIR = pathlib.Path('./models')
   MODELS_DIR.mkdir(exist_ok=True)
   ```
3. **Markdown** — "## Dataset" — one paragraph describing the dataset and how to obtain it.
4. **Code** — data loading:
   ```python
   from <NAME>_data import get_data
   train_ds, val_ds = get_data(DATA_PATH, input_shape=(<H>, <W>, <C>), batch_size=32)
   ```
5. **Markdown** — "## Model" — describe the architecture and why it suits Akida.
6. **Code** — model creation:
   ```python
   from <NAME>_model import build_<NAME>_model
   model = build_<NAME>_model()
   model.summary()
   ```
7. **Markdown** — "## Float Training" — explain the training process and LR schedule.
8. **Code** — conditional float training or Pooch download:
   ```python
   from <NAME>_train import train_<NAME>
   if RUN_FLOAT_TRAINING:
       train_<NAME>(model, train_ds, val_ds, epochs=<N>, learning_rate=<LR>)
       model.save(MODELS_DIR / '<short_model_name>_<NAME>.h5', include_optimizer=False)
   else:
       model = pooch.retrieve(
           url='https://data.brainchip.com/models/AkidaV<AKIDA_VERSION>/<arch>/<short_model_name>_<NAME>.h5',
           known_hash=None,  # TODO: add hash after first successful training run
           path=MODELS_DIR,
           fname='<short_model_name>_<NAME>.h5',
       )
   ```
   (If no pretrained URL is known, just train unconditionally and note that.)
9. **Code** — float evaluation:
   ```python
   from cnn2snn import load_quantized_model
   model = load_quantized_model(str(MODELS_DIR / '<short_model_name>_<NAME>.h5'))
   model.compile(metrics=['accuracy'])
   _, float_acc = model.evaluate(val_ds)
   print(f'Float accuracy: {float_acc:.4f}')
   ```
10. **Markdown** — "## Quantization" — explain PTQ and bit-width choice.
11. **Code** — quantization:
    ```python
    import cnn2snn
    quantized_model = cnn2snn.quantize(model, input_weight_quantization=<I>,
                                       weight_quantization=<W>, activ_quantization=<A>)
    ```
12. **Markdown** — "## Quantization-Aware Training (QAT)" — explain fine-tuning.
13. **Code** — conditional QAT or Pooch download:
    ```python
    if RUN_QAT_TRAINING:
        train_<NAME>(quantized_model, train_ds, val_ds, epochs=2, learning_rate=1e-4)
        quantized_model.save(MODELS_DIR / '<short_model_name>_<NAME>_qat.h5', include_optimizer=False)
    else:
        quantized_model = pooch.retrieve(
            url='https://data.brainchip.com/models/AkidaV<AKIDA_VERSION>/<arch>/<short_model_name>_<NAME>_iq<I>wq<W>aq<A>.h5',
            known_hash=None,
            path=MODELS_DIR,
            fname='<short_model_name>_<NAME>_qat.h5',
        )
    ```
14. **Code** — quantized model evaluation:
    ```python
    quantized_model = load_quantized_model(str(MODELS_DIR / '<short_model_name>_<NAME>_qat.h5'))
    quantized_model.compile(metrics=['accuracy'])
    _, qat_acc = quantized_model.evaluate(val_ds)
    print(f'QAT accuracy: {qat_acc:.4f}')
    ```
15. **Markdown** — "## Conversion to Akida Format".
16. **Code** — conversion:
    ```python
    akida_model = cnn2snn.convert(quantized_model)
    akida_model.save(str(MODELS_DIR / '<short_model_name>_<NAME>_qat.fbz'))
    ```
17. **Markdown** — "## Hardware Device Detection".
18. **Code** — device detection:
    ```python
    from brainchip_utils.hardware_utils import get_akida_device
    device = get_akida_device(target_version=akida_model.ip_version)
    if device is not None:
        print(f'Akida hardware found: {device}')
    else:
        print('No hardware found — using software backend')
    ```
19. **Markdown** — "## Akida Evaluation".
20. **Code** — Akida evaluation (copy the manual iteration loop from `vww_eval.py`'s
    `evaluate_akida_model` function, adapted for this example's label format).
21. **Markdown** — "## Activation Sparsity".
22. **Code** — sparsity analysis:
    ```python
    from akida_models.sparsity import compute_sparsity
    from brainchip_utils.plot_utils import pretty_print_sparsity
    from <NAME>_data import get_samples
    samples = get_samples(DATA_PATH, input_shape=(<H>, <W>, <C>), num_samples=1000)
    sparsity_dict = compute_sparsity(akida_model, samples=samples)
    pretty_print_sparsity(sparsity_dict)
    ```
23. **Markdown** — "## Hardware Benchmark" — explain that this requires connected hardware.
24. **Code** — benchmark (guarded by `if device is not None`):
    ```python
    if device is not None:
        from brainchip_utils.hardware_utils import full_model_benchmark, per_layer_benchmark, get_mapping_stats
        from brainchip_utils.plot_utils import plot_full_model_results, plot_per_layer_results
        # full-model benchmark both modes, then per-layer — copy pattern from vww_notebook.ipynb
    else:
        print('Hardware not available — skipping benchmark')
    ```
25. **Markdown** — "## Summary" — print float/QAT/Akida accuracy side-by-side.
26. **Code** — summary print.

Write the notebook as proper JSON (nbformat 4.5). Each code cell has:
```json
{
  "cell_type": "code",
  "execution_count": null,
  "metadata": {},
  "outputs": [],
  "source": ["line 1\n", "line 2\n"]
}
```
Each markdown cell has:
```json
{
  "cell_type": "markdown",
  "metadata": {},
  "source": ["# Title\n", "\n", "Paragraph text.\n"]
}
```
The top-level structure:
```json
{
  "nbformat": 4,
  "nbformat_minor": 5,
  "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"}
  },
  "cells": [...]
}
```

### 3l. Directory stubs

Create `models/` and `data/` each containing a `.gitignore` that ignores everything except
itself (copy verbatim from the reference example — it is not a `.gitkeep`):
```
# Git to Ignore everything in this directory
*
# Except this .gitignore file
!.gitignore
```

---

## Akida 2 targets

When `AKIDA_VERSION == 2`, the canonical reference is `akida2/model_zoo/vww/`. Read those
files directly and mirror them. This section records how Akida 2 differs from the Akida 1
instructions above; where they conflict, this section wins. Everything not mentioned here
(data loading, eval's `evaluate_akida_model`, the DATA_ARG forwarding, the metrics→README
mechanism, the `get_samples` uint8 invariant) is the same as Akida 1.

> **Prerequisite:** this section assumes `akida2/model_zoo/vww/` exists in the repo as the
> reference example (the same role `akida1/model_zoo/vww/` plays for v1). If that directory
> is not present in the checkout, the v2 reference has not landed yet — stop and confirm with
> the user rather than generating a v2 example against a missing reference. The instructions
> below capture the v2 pattern in enough detail to review, but the committed reference is the
> source of truth.

Status note: as of this writing, Akida 2 reference hardware is an **FPGA at 25 MHz**;
AKD2500 production silicon does not exist yet. So the software pipeline is fully real, but
the hardware benchmark is FPGA-based and **latency-only** (the power path is still under
development). Do not fabricate power numbers or a production clock.

### A2.1 Model (`<NAME>_model.py`)

Akida 2 models are built **directly from the `akida_models` factory with `include_top=True`**,
not by grafting a custom head onto a pretrained backbone (that is the Akida 1 transfer-learning
pattern). Concretely, the VWW v2 reference is:
```python
from akida_models.imagenet import akidanet_imagenet
from cnn2snn import set_akida_version, AkidaVersion
from tf_keras.utils import set_random_seed

def build_<NAME>_model(seed=42):
    set_random_seed(seed)
    with set_akida_version(AkidaVersion.v2):
        model = akidanet_imagenet(
            input_shape=(<H>, <W>, <C>),
            alpha=<ALPHA>,
            classes=<NUM_CLASSES>,
            include_top=True,
            input_scaling=(255, 0),   # uint8 in; on-graph /255. The factory default
                                      # is (128, -1) and would be WRONG -- pass explicitly.
        )
    return model
```
Notes:
- `akidanet_imagenet` (and the other `akida_models` factories) are version-aware: under
  `set_akida_version(AkidaVersion.v2)` they select v2-compatible layer/activation variants
  automatically. Swapping the context manager is the whole version switch — there is no
  per-layer "verify v2 compatibility" work for standard factory models.
- `include_top=True` appends the factory's own `global_avg → dropout(1e-3) → classifier`
  (Dense) head. Do not hand-build a head or tap an intermediate layer.
- Verify the built model against any reference artifact you have (the VWW v2 model was
  confirmed layer-for-layer against the real `akidanet_vww.h5`). If the source example is a
  different architecture, use its corresponding factory but keep the same
  `include_top=True` + explicit `input_scaling` approach.
- Save path default: `./models/<short_model_name>_<NAME>_untrained.h5` (from-scratch, so the
  `_untrained` suffix applies — v2 does not fine-tune a pretrained backbone).

### A2.2 Quantization tool and variants

Akida 2 uses **`quantizeml`**, not `cnn2snn quantize`. The default bit-widths are 8/8/8
(`quantizeml.layers.QuantizationParams` defaults: `activation_bits=8, weight_bits=8,
input_weight_bits=8`). Standard Akida 2 practice for this class of model produces **three
quantized variants**:

| Variant | quantize command | QAT? |
|---|---|---|
| 8-bit | `quantizeml quantize -m <float>.h5 -i 8 -w 8 -a 8 -s <name>_i8_w8_a8.h5` | No — 8-bit PTQ is accurate enough |
| 4-bit weights (PTQ) | `quantizeml quantize -m <float>.h5 -i 8 -w 4 -a 8 -s <name>_i8_w4_a8.h5` | No |
| 4-bit weights (QAT) | fine-tune the 4-bit PTQ model with `<NAME>_train.py` | Yes |

- Activations stay 8-bit in every variant; only weight bit-width drops to 4 for the 4-bit
  variants.
- `quantizeml quantize` calibrates during quantization; with no `-sa samples.npz` it uses
  random calibration samples (`-ns` default 1024). Whether to pass explicit calibration
  samples is a per-example choice — default to the no-samples path and flag it.
- `quantizeml` has **no `convert` subcommand**. Conversion to `.fbz` is still
  `cnn2snn convert -m <model>.h5` for every variant — `cnn2snn.convert` accepts
  quantizeml-quantized models. Output filename is `<input_stem>.fbz`.

### A2.3 Pipeline (`<NAME>_train.sh`)

Structure (see `akida2/model_zoo/vww/vww_train.sh`): build untrained → float-train from
scratch → eval → then, for each of the three variants: `quantizeml quantize` → eval `.h5` →
`cnn2snn convert` → eval `.fbz` → `<NAME>_benchmark.py`. The 4-bit-QAT variant additionally
runs `<NAME>_train.py` on the 4-bit PTQ `.h5` to produce `<name>_i8_w4_a8_qat.h5` before its
convert/eval/benchmark.

Model naming (Akida 2):
- Float: `<short>_<NAME>_untrained.h5` → `<short>_<NAME>.h5`
- 8-bit: `<short>_<NAME>_i8_w8_a8.h5` → `<short>_<NAME>_i8_w8_a8.fbz`
- 4-bit PTQ: `<short>_<NAME>_i8_w4_a8.h5` → `<short>_<NAME>_i8_w4_a8.fbz`
- 4-bit QAT: `<short>_<NAME>_i8_w4_a8_qat.h5` → `<short>_<NAME>_i8_w4_a8_qat.fbz`

Float-training and QAT epoch/LR values are provisional (a weights file cannot reveal
training duration) — mark them as confirm-on-first-run. Since v2 trains from scratch, expect
more float epochs than a v1 fine-tuning example.

### A2.4 Benchmark (`<NAME>_benchmark.py`)

- `MEASURED_CLOCK = 25e6` (FPGA), replacing the AKD1500 `400e6`.
- Add a **projected latency** at a higher target clock: cycle count is clock-independent, so
  `projected_ms = mean_inf_clk / PROJECTED_CLOCK * 1000`. `PROJECTED_CLOCK` is a **provisional
  placeholder** (the reference uses 100 MHz with a `# TODO: confirm target clock`) — do not
  present it as final.
- **Latency-only**: do not request or surface power measurement (the FPGA power path is WIP).
  Drop the power columns/keys entirely.
- `--save-metrics` writes **variant-keyed** metrics (prefix inferred from the `.fbz`
  filename): `<variant>_sparsity`, and per map-mode `<variant>_<mode>_{nps,passes,cycles,
  latency_ms,projected_ms}`, where `<variant>` ∈ {`w8a8`, `w4a8_ptq`, `w4a8_qat`}. Watch the
  ordering trap: test for `i8_w4_a8_qat` (contains `qat`) before the plain `i8_w4_a8` case.

### A2.5 README template + metrics

- **Model card** is rows-per-variant: one row per quantized variant with columns
  `Variant | Weights/Acts | QAT | Quantized acc. | Akida acc. | Sparsity`. The two non-QAT
  variants show `-` in the QAT column. Float accuracy + params are stated once above the
  table (shared across variants).
- **Benchmark table** covers all three variants × {Minimal, AllNps} = 6 rows, latency-only,
  with `Latency @ 25 MHz` and `Projected @ <N> MHz (provisional)` columns. No power columns.
- `metrics.json` keys are variant-prefixed and must be in exact bijection with both the
  template placeholders and the eval/benchmark `--save-metrics` writes. (For reference, the
  VWW v2 example happens to have 41 such keys, but the count is example-specific — never
  hardcode it; always derive the set from the template you actually wrote.) Verify by
  regex-extracting template placeholders and comparing to the union of script-written keys —
  the sets must be equal.
- `--save-metrics` in `<NAME>_eval.py` writes `float_acc`/`params` for the float model and
  `<variant>_quant_acc` (from `.h5`) / `<variant>_akida_acc` (from `.fbz`) per variant, using
  the same filename-based variant inference (with the same `_qat`-before-`i8_w4_a8` ordering
  trap).

### A2.6 Directory + not-yet-existing paths

`TARGET_DIR` for v2 is `akida2/model_zoo/<NAME>/`. The `akida2/` tree may be new; create it.
Colab support (badge, `colab_setup.py`, notebook setup cell) is intentionally **out of scope
for now** — do not add it to v2 examples yet.

---

## Step 4 — Report

After creating all files, print a summary listing:
1. Every file created and its path.
2. Any TODOs left for the user (e.g. dataset URL, pretrained model URL, hash values,
   Akida2-specific constants).
3. The verification commands to run:
   ```bash
   cd TARGET_DIR
   python -c "import ast; [ast.parse(open(f).read()) for f in ['<NAME>_model.py','<NAME>_data.py','<NAME>_train.py','<NAME>_eval.py','<NAME>_benchmark.py']]"
   bash -n <NAME>_train.sh
   python update_readme.py   # must render with no KeyError
   ```
   For **Akida 2**, additionally confirm the `--save-metrics` key set matches the template
   exactly (extract `{...}` placeholders from the template and compare against the union of
   keys the eval + benchmark scripts write — the two sets must be equal, no extras on either
   side).

---

## Key invariants (do not break these)

- `update_readme.py` uses `str.format_map(metrics)` — every `{key}` in the template must
  have a matching entry in `docs/metrics.json`, including any new keys you add.
- The `--save-metrics` flag in `_eval.py` and `_benchmark.py` must write to
  `docs/metrics.json` relative to `__file__` (not CWD), matching the VWW pattern.
- `brainchip_utils` imports must be exactly:
  `from brainchip_utils.hardware_utils import get_akida_device, get_mapping_stats, per_layer_benchmark, full_model_benchmark`
  `from brainchip_utils.plot_utils import plot_full_model_results, plot_per_layer_results, pretty_print_sparsity`
- The shell script `DATA_ARG` forwarding pattern must be preserved verbatim.
- `get_samples()` must always return `np.ndarray` of `dtype=uint8` — this is required by
  `per_layer_benchmark` and `full_model_benchmark`.
- `AkidaVersion` has exactly two members, `v1` and `v2`. `akida_models` factories are
  version-aware inside a `set_akida_version(...)` context — swapping the context manager is
  the version switch; there is no per-layer v2 compatibility work for standard factory models.
- Akida 1 quantization is `cnn2snn quantize` (fixed `i8/w4/a4`, QAT required). Akida 2
  quantization is `quantizeml quantize` (default `i8/w8/a8`, no QAT needed at 8-bit).
  **Both** convert to `.fbz` with the same `cnn2snn convert` — there is no `quantizeml
  convert`, and no version flag on convert.
- Do not fabricate Akida 2 hardware numbers: AKD2500 production silicon does not exist yet;
  the reference platform is a 25 MHz FPGA and benchmarking is latency-only (no power) with a
  clearly-provisional projected clock. The software pipeline (train → quantize → convert →
  software-backend eval → sparsity) is fully real for Akida 2 regardless.
- For Akida 2, the metrics.json / template / `--save-metrics` key sets must be in exact
  three-way bijection (variant-prefixed keys). Verify programmatically, not by eye.
