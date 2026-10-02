#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""Regenerate README.md from README.md.template + metrics.json.

The template is filled from metrics.json plus a set of derived plot-path keys.
Each benchmarked model variant produces its own per-layer and full-model plots
(named after the model file by plant_village_benchmark.py), so the paths are
computed here from the variant -> model-stem map and exposed to the template as
`{<variant>_layers_plot}` / `{<variant>_full_plot}`. This keeps the plot names
in one place instead of hardcoding long filenames in the template.
"""
import json
import pathlib

here = pathlib.Path(__file__).parent

# Variant -> model filename stem. Must match the .fbz names produced by
# plant_village_train.sh and the variant keys written into metrics.json.
VARIANT_MODEL_STEMS = {
    "w8a8": "akidanet_plant_village_i8_w8_a8",
    "w4a4_qat": "akidanet_plant_village_i8_w4_a4_qat",
}

metrics = json.loads((here / "docs" / "metrics.json").read_text())
template = (here / "docs" / "README.md.template").read_text()

# Derived plot paths (relative to the README at the repo root, i.e. under docs/).
# These mirror the savepaths in plant_village_benchmark.py:
#   docs/ref_benchmark_results_layers_<stem>.png
#   docs/ref_benchmark_results_full_<stem>.png
context = dict(metrics)
for variant, stem in VARIANT_MODEL_STEMS.items():
    context[f"{variant}_layers_plot"] = f"docs/ref_benchmark_results_layers_{stem}.png"
    context[f"{variant}_full_plot"] = f"docs/ref_benchmark_results_full_{stem}.png"

(here / "README.md").write_text(template.format_map(context))
print("README.md updated.")
