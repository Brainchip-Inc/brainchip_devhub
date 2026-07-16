---
name: model-zoo-example
description: Generate a complete brainchip_devhub model zoo example from an akida_models source example
---

# Model Zoo Example Generator

Generate a self-contained model zoo example in `brainchip_devhub`, adapted from the source
scripts in `akida_models`. The new example follows the VWW example structure exactly.

## Usage

```
/model-zoo-example <name> [--akida-version 1|2]
```

- `<name>` — subdirectory name under `akida_models/scripts/` (e.g. `kws`, `mnist`, `face`)
- `--akida-version` — target hardware generation (default: `1`)

Parse these from `$ARGUMENTS` now. Set:
- `NAME` = the example name
- `AKIDA_VERSION` = `1` or `2` (default `1`)
- `SOURCE_DIR` = `$AKIDA_MODELS_DIR/scripts/<NAME>/`
- `TARGET_DIR` = `DEVHUB_DIR/akida<AKIDA_VERSION>/model_zoo/<NAME>/`

Resolve the two repo locations before doing anything else, in this order of preference:
1. Environment variables `BRAINCHIP_DEVHUB` and `AKIDA_MODELS`, if set.
2. Paths passed as `--devhub <path>` and `--akida-models <path>` in `$ARGUMENTS`.
3. Sibling directories of the current working directory (e.g. if invoked inside a
   `brainchip_devhub` checkout, look for `../akida_models`).
4. Search upward from the current directory for folders named `brainchip_devhub`
   and `akida_models`.

Set `DEVHUB_DIR` and `AKIDA_MODELS_DIR` from whichever resolves first. If neither
`brainchip_devhub` nor `akida_models` can be found, stop and ask the user for the
paths rather than guessing. Do not hardcode any absolute user path.

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

Read these files from the VWW example (they are the pattern every target file must follow):

- `akida1/model_zoo/vww/vww_model.py`
- `akida1/model_zoo/vww/vww_data.py`
- `akida1/model_zoo/vww/vww_train.py`
- `akida1/model_zoo/vww/vww_eval.py`
- `akida1/model_zoo/vww/vww_benchmark.py`
- `akida1/model_zoo/vww/vww_train.sh`
- `akida1/model_zoo/vww/update_readme.py`
- `akida1/model_zoo/vww/docs/README.md.template`
- `akida1/model_zoo/vww/docs/metrics.json`

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

For **Akida 2** targets, change `AkidaVersion.v1` → `AkidaVersion.v2` and add a comment:
```python
# TODO: Akida2 — verify architecture compatibility with v2 constraints
```

**Transfer learning models**: if Step 1 found that the source uses transfer learning,
generate `<NAME>_model.py` with the following structure instead of a plain factory call:

- Create the base model with `include_top=False, pooling='avg'` and
  `input_scaling=(255, 0)` (required for Akida 1 uint8 inputs)
- Load pretrained weights: `fetch_file(url, fname=..., cache_subdir='models')` then
  `base_model.load_weights(pretrained_weights, by_name=True)`
- Add the custom top layers from the source (e.g. `dense_block`, `Dropout`, `Activation`,
  `Reshape`) — copy the layer structure and names exactly from Step 1
- For Akida 1, obtain the relu activation via `get_params_by_version()` using the v1
  argument (not the v2 value from the source)
- Build with `tf_keras.Model(base_model.input, x, name='<name>')`
- Wrap the full construction in `with set_akida_version(AkidaVersion.v1):`
- Default save path: `./models/<short_model_name>_<NAME>.h5` (no `_untrained` suffix —
  the backbone is already pretrained; this file is the fine-tuning starting point)
- Add imports for `tf_keras`, `fetch_file`, `dense_block`, `get_params_by_version`, and
  any layer types used in the top (e.g. `Activation`, `Dropout`, `Reshape`)

### 3b. `<NAME>_data.py`

Follow `vww_data.py` exactly. Key rules:
- Expose exactly two public functions: `get_data(data_path, input_shape, batch_size,
  dtype=tf.uint8)` returning `(train_dataset, val_dataset)` and `get_samples(data_path,
  input_shape, num_samples=1024)` returning a `np.ndarray` of uint8.
- Adapt the internals to load the dataset discovered in Step 1:
  - If the dataset has `train/` + `val/` subdirectories: use `ImageDataGenerator` +
    `flow_from_directory`, exactly as in `vww_data.py`. Use `class_mode='sparse'`.
  - If the dataset is a `.npz` file (e.g. KWS): load with `np.load`, split into train/val,
    wrap in `tf.data.Dataset.from_tensor_slices`.
  - Preserve augmentation from the source where it applies (spatial augmentation for images;
    no augmentation for spectrograms/1D signals unless the source does it).
- The `get_samples()` function always returns uint8 numpy arrays — apply the same scaling
  logic the source uses for calibration samples.
- Default `data_path` should be `./data/<dataset_dir_name>`.

### 3c. `<NAME>_train.py`

Follow `vww_train.py` exactly. Key rules:
- The training function is named `train_<NAME>(model, train_ds, val_ds, epochs,
  learning_rate, regularization=None)`.
- Use `SparseCategoricalCrossentropy(from_logits=True)` for sparse integer labels (the
  standard case). Use `CategoricalCrossentropy` only if source uses one-hot labels.
- Apply the same step-decay LR scheduler from VWW (epochs 0-19: `lr`, 20-39: `lr*0.5`,
  40+: `lr*0.25`) unless the source uses a substantially different schedule, in which case
  adapt it but keep it inside `get_custom_scheduler()`.
- Always include `RestoreBest` from `akida_models.training`.
- CLI: `-l`, `-s`, `-d`, `-b`, `-e`, `-lr`, `-reg`, `--seed` with the same defaults as VWW
  except adapt epochs and LR defaults to match the source.

### 3d. `<NAME>_eval.py`

Copy `vww_eval.py` and make the following substitutions only:
- Replace all occurrences of `vww_data` → `<NAME>_data`.
- Replace the default `--data` path to match the dataset for this example.
- In the `--save-metrics` block, keep the same logic for inferring which key to write
  (`float_acc`, `qat_acc`, `akida_acc`, `params`) — this logic is generic and needs no
  change.
- The `evaluate_akida_model` function is identical; copy verbatim.

### 3e. `<NAME>_benchmark.py`

Copy `vww_benchmark.py` and make the following substitutions only:
- Replace `from vww_data import get_samples` → `from <NAME>_data import get_samples`.
- Replace the default `--data` path to match the dataset for this example.
- All `brainchip_utils` imports, benchmark calls, and plotting calls are identical; do not
  change them.
- For **Akida 2** targets, replace the clock frequency constant with:
  ```python
  CLOCK_FREQUENCY = 400e6  # TODO: Akida2 — confirm clock frequency for AKD2500
  ```
  and add a comment above the mapping block:
  ```python
  # TODO: Akida2 — confirm available MapMode values for v2
  map_modes = ['Minimal', 'AllNps']
  ```

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

   For **Akida 2**: same table structure with the same keys but change the header text
   to "AKD2500 hardware benchmark". Add a comment in the template source:
   `<!-- TODO: Akida2 — update once reference hardware benchmark is available -->`

   Then write a short description of the model architecture (name, key hyperparameters).

5. **`## Pipeline`** — copy the four-row pipeline table from VWW verbatim (the pipeline
   stages are the same for all Akida examples).

6. **`## Requirements`** — copy from VWW verbatim (versions are the same). If this example
   has additional dependencies (e.g. librosa for audio), add them.

7. **`## Dataset setup`** — describe how to obtain this dataset. If a URL is known from the
   source scripts, include it with the wget + extract commands. If not known, write
   `<!-- TODO: add dataset download instructions -->`.

8. **`## Usage`** — three subsections:
   - `### Notebook` — one paragraph linking `<NAME>_notebook.ipynb` and describing what
     it covers.
   - `### Script` — copy the VWW structure (the `bash <NAME>_train.sh [DATADIR]` intro,
     then the nine numbered steps), adapting filenames, epochs, and LR values.
   - Include the pretrained model shortcut as step "5b" if a BrainChip pretrained model
     URL is known from the source; otherwise omit.

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
will crash. Audit the template you just wrote and add any extra keys you introduced.

### 3j. `README.md`

Generate the initial README by running `update_readme.py` inline (read the template and
format it with the metrics dict). The result will have "TBD" everywhere metrics should be —
that is correct and expected until training runs are complete.

### 3k. `<NAME>_notebook.ipynb`

Write the notebook as a valid `.ipynb` JSON file. Use `nbformat` version 4.5. Follow the
VWW notebook cell structure (read `akida1/model_zoo/vww/vww_notebook.ipynb` for the exact
cell text and markdown content if needed, but it is large — focus on the structure).

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

Create these empty placeholder files:
- `models/.gitkeep`
- `data/.gitkeep`

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
   python -c "import json; json.load(open('<NAME>_notebook.ipynb'))"
   python update_readme.py
   ```

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
