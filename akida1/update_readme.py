#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""Regenerate akida1/README.md from docs/README.md.template.

The model zoo summary table is built from each example's docs/zoo_card.json
(static row metadata) and docs/metrics.json (measured values), so it never
drifts from the example READMEs. Examples without a zoo_card.json are skipped.
"""
import json
import pathlib

DOMAINS = ["Image", "Audio", "Time series"]
MAPPINGS = {"minimal": "Minimal", "allnps": "AllNPs", "hwpr": "HwPr"}
MISSING = ("", "TBD", None)

here = pathlib.Path(__file__).parent


def _value(metrics, key):
    value = metrics.get(key)
    return None if value in MISSING else value


def _best_mapping(metrics, prefix):
    """Mapping with the lowest total energy per inference, or None if unbenchmarked."""
    candidates = []
    for mapping in MAPPINGS:
        energy = _value(metrics, f"{prefix}{mapping}_total_E")
        if energy is not None:
            candidates.append((float(energy), mapping))
    return min(candidates)[1] if candidates else None


def _cells(example, row, metrics):
    prefix = row.get("bench_prefix", "")
    perf = _value(metrics, row["metric_key"])
    mapping = _best_mapping(metrics, prefix)
    energy = latency = None
    if mapping:
        energy = _value(metrics, f"{prefix}{mapping}_total_E")
        latency = _value(metrics, f"{prefix}{mapping}_latency_ms")
    return [
        f'<a href="model_zoo/{example}">{row['task']}</a>',
        row["category"],
        row["dataset"],
        f"{perf} {row['metric_label']}" if perf else "—",
        energy or "—",
        latency or "—",
        row.get("notes", ""),
    ]


HEADERS = ["Task", "Category", "Dataset", "Performance", "Energy (mJ/inf)", "Latency (ms)", "Notes"]
RIGHT_ALIGNED = {4, 5}
MERGEABLE = 3   # Task, Category, Dataset: merged down a run of rows from the same example


def _spans(rows):
    """rowspan per cell: >1 starts a merged run, 0 is covered by the run above."""
    spans = [[1] * len(HEADERS) for _ in rows]
    for col in range(MERGEABLE):
        run_start = 0
        for i in range(1, len(rows) + 1):
            same = (i < len(rows) and rows[i][0] == rows[run_start][0]
                    and rows[i][1][col] == rows[run_start][1][col])
            if not same:
                spans[run_start][col] = i - run_start
                for j in range(run_start + 1, i):
                    spans[j][col] = 0
                run_start = i
    return spans


def _html_table(rows):
    """rows: list of (example, cells). Returns an HTML table with repeated leading cells merged."""
    out = ["<table>", "  <thead>", "    <tr>"]
    out += [f"      <th>{h}</th>" for h in HEADERS]
    out += ["    </tr>", "  </thead>", "  <tbody>"]
    for (_, cells), row_spans in zip(rows, _spans(rows)):
        out.append("    <tr>")
        for col, (cell, span) in enumerate(zip(cells, row_spans)):
            if span == 0:
                continue
            attrs = f' rowspan="{span}"' if span > 1 else ""
            attrs += ' align="right"' if col in RIGHT_ALIGNED else ""
            out.append(f"      <td{attrs}>{cell}</td>")
        out.append("    </tr>")
    out += ["  </tbody>", "</table>"]
    return "\n".join(out)


def build_table():
    sections = {domain: [] for domain in DOMAINS}
    for card_path in sorted(here.glob("model_zoo/*/docs/zoo_card.json")):
        example_dir = card_path.parents[1]
        card = json.loads(card_path.read_text())
        metrics_path = example_dir / "docs" / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        domain = card["domain"]
        if domain not in sections:
            raise ValueError(f"{card_path}: unknown domain {domain!r}, expected one of {DOMAINS}")
        for i, row in enumerate(card["rows"]):
            key = (card.get("order", 0), example_dir.name, i)
            sections[domain].append((key, (example_dir.name, _cells(example_dir.name, row, metrics))))

    out = []
    for domain, rows in sections.items():
        if rows:
            out += [f"### {domain}", "", _html_table([r for _, r in sorted(rows)]), ""]
    return "\n".join(out).rstrip()


if __name__ == "__main__":
    template = (here / "docs" / "README.md.template").read_text()
    (here / "README.md").write_text(template.replace("{model_zoo_table}", build_table()))
    print("akida1/README.md updated.")
