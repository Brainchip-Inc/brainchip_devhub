#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""Regenerate README.md from README.md.template + metrics.json."""
import json
import pathlib
import runpy

here = pathlib.Path(__file__).parent
metrics = json.loads((here / "docs" / "metrics.json").read_text())
template = (here / "docs" / "README.md.template").read_text()
(here / "README.md").write_text(template.format_map(metrics))
print("README.md updated.")

# Refresh the Akida 1 landing README, whose model zoo table reads this metrics.json.
runpy.run_path(str(here.parents[1] / "update_readme.py"), run_name="__main__")
