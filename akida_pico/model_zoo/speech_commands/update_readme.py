#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""Regenerate README.md from docs/README.md.template + docs/metrics.json, then the
Akida Pico landing README, whose model zoo table reads the same metrics.json."""
import json
import pathlib
import runpy

here = pathlib.Path(__file__).parent

metrics = json.loads((here / "docs" / "metrics.json").read_text())
template = (here / "docs" / "README.md.template").read_text()
(here / "README.md").write_text(template.format_map(metrics))
print("README.md updated.")

runpy.run_path(str(here.parents[1] / "update_readme.py"), run_name="__main__")
