#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""Regenerate akida2/README.md from docs/README.md.template.

The model zoo summary table is built from each example's docs/zoo_card.json
(static row metadata, one row per quantization variant) and docs/metrics.json
(measured values), so it never drifts from the example READMEs. Examples without
a zoo_card.json are skipped.

Akida 2 is benchmarked on the FPGA, which has no power measurement, so the table
reports latency only: measured at the FPGA clock and projected to the AKD2500
target clock, for the mapping with the fewest cycles.
"""
import pathlib
import sys

here = pathlib.Path(__file__).parent
sys.path.insert(0, str(here.parent))  # brainchip_utils, without needing the package installed
from brainchip_utils.zoo_table import domain_sections, html_table, load_cards, value  # noqa: E402

DOMAINS = ["Image", "Audio", "Time series"]
MAPPINGS = {"minimal": "Minimal", "allnps": "AllNPs", "hwpr": "HwPr"}
HEADERS = ["Task", "Category", "Dataset", "Variant", "Performance", "Mapping",
           "Latency @ 25 MHz FPGA (ms)", "Projected @ 1 GHz AKD2500 (ms)", "Notes"]
RIGHT_ALIGNED = {6, 7}
MERGEABLE = 3   # Task, Category, Dataset: merged down a run of rows from the same example


def _fastest_mapping(metrics, prefix):
    """Mapping with the fewest cycles per inference, or None if unbenchmarked."""
    candidates = []
    for mapping in MAPPINGS:
        cycles = value(metrics, f"{prefix}{mapping}_cycles")
        if cycles is not None:
            candidates.append((float(cycles), mapping))
    return min(candidates)[1] if candidates else None


def _cells(example, row, metrics):
    prefix = row["bench_prefix"]
    perf = value(metrics, row["metric_key"])
    mapping = _fastest_mapping(metrics, prefix)
    latency = projected = None
    if mapping:
        latency = value(metrics, f"{prefix}{mapping}_latency_ms")
        projected = value(metrics, f"{prefix}{mapping}_projected_ms")
    return [
        f'<a href="model_zoo/{example}">{row["task"]}</a>',
        row["category"],
        row["dataset"],
        row["variant"],
        f"{perf} {row['metric_label']}" if perf else "—",
        MAPPINGS[mapping] if mapping else "—",
        latency or "—",
        projected or "—",
        row.get("notes", ""),
    ]


def build_table():
    sections = {domain: [] for domain in DOMAINS}
    for example, card, metrics in load_cards(here, DOMAINS):
        for i, row in enumerate(card["rows"]):
            key = (card.get("order", 0), example, i)
            sections[card["domain"]].append((key, (example, _cells(example, row, metrics))))
    return domain_sections({
        domain: html_table(HEADERS, [r for _, r in sorted(rows)], RIGHT_ALIGNED, MERGEABLE) if rows else ""
        for domain, rows in sections.items()})


if __name__ == "__main__":
    template = (here / "docs" / "README.md.template").read_text()
    (here / "README.md").write_text(template.replace("{model_zoo_table}", build_table()))
    print("akida2/README.md updated.")
