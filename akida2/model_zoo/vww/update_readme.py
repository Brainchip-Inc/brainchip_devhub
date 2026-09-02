#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""Regenerate README.md from README.md.template + metrics.json."""
import json
import pathlib

here = pathlib.Path(__file__).parent
metrics = json.loads((here / "docs" / "metrics.json").read_text())
template = (here / "docs" / "README.md.template").read_text()
(here / "README.md").write_text(template.format_map(metrics))
print("README.md updated.")
