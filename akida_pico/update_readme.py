#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""Regenerate akida_pico/README.md from docs/README.md.template.

The model zoo summary table is built from each example's docs/zoo_card.json
(static row metadata) and docs/metrics.json (measured values), so it never drifts
from the example READMEs. Examples without a zoo_card.json are skipped.

Akida Pico examples are not benchmarked on hardware yet, so the table reports the
accuracy of the converted Akida model only.
"""
import pathlib
import sys

here = pathlib.Path(__file__).parent
sys.path.insert(0, str(here.parent))  # brainchip_utils, without needing the package installed
from brainchip_utils.zoo_table import domain_sections, html_table, load_cards, value  # noqa: E402

DOMAINS = ["Audio", "Time series"]
HEADERS = ["Task", "Category", "Dataset", "Variant", "Performance", "Parameters", "Notes"]
RIGHT_ALIGNED = {5}
MERGEABLE = 3   # Task, Category, Dataset: merged down a run of rows from the same example


def _cells(example, row, metrics):
    perf = value(metrics, row["metric_key"])
    return [
        f'<a href="model_zoo/{example}">{row["task"]}</a>',
        row["category"],
        row["dataset"],
        row["variant"],
        f"{perf} {row['metric_label']}" if perf else "—",
        value(metrics, "params") or "—",
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
    print("akida_pico/README.md updated.")
