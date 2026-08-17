#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""Regenerate README.md from README.md.template + metrics.json.

Also regenerates the benchmark summary figure, which is drawn from the same
metrics file and would otherwise drift away from the tables beneath it.
"""
import json
import pathlib

from imagenet_akidanet_summary_plot import plot_summary

here = pathlib.Path(__file__).parent
metrics = json.loads((here / "docs" / "metrics.json").read_text())
template = (here / "docs" / "README.md.template").read_text()
(here / "README.md").write_text(template.format_map(metrics))
print("README.md updated.")

summary_path = here / "docs" / "ref_benchmark_summary.png"
plot_summary(metrics, savepath=summary_path)
print(f"{summary_path.name} updated.")
