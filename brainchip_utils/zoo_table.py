# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""Model zoo summary tables for the generated landing READMEs (akida1/, akida2/).

Each example's docs/zoo_card.json gives static row metadata and docs/metrics.json the
measured values, so the landing tables never drift from the example READMEs. Standard
library only: check_content.py imports this without the toolchain installed.
"""
import json

MISSING = ("", "TBD", None)


def value(metrics, key):
    """metrics[key], or None if absent or a placeholder."""
    v = metrics.get(key)
    return None if v in MISSING else v


def load_cards(platform_dir, domains):
    """Yield (example_name, card, metrics) for every example with a zoo_card.json."""
    for card_path in sorted(platform_dir.glob("model_zoo/*/docs/zoo_card.json")):
        example_dir = card_path.parents[1]
        card = json.loads(card_path.read_text())
        if card["domain"] not in domains:
            raise ValueError(f"{card_path}: unknown domain {card['domain']!r}, expected one of {domains}")
        metrics_path = example_dir / "docs" / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        yield example_dir.name, card, metrics


def _spans(rows, ncols, mergeable):
    """rowspan per cell: >1 starts a merged run, 0 is covered by the run above."""
    spans = [[1] * ncols for _ in rows]
    for col in range(mergeable):
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


def html_table(headers, rows, right_aligned=(), mergeable=0):
    """rows: list of (example, cells). The first `mergeable` columns are merged down a
    run of rows from the same example with the same value."""
    out = ["<table>", "  <thead>", "    <tr>"]
    out += [f"      <th>{h}</th>" for h in headers]
    out += ["    </tr>", "  </thead>", "  <tbody>"]
    for (_, cells), row_spans in zip(rows, _spans(rows, len(headers), mergeable)):
        out.append("    <tr>")
        for col, (cell, span) in enumerate(zip(cells, row_spans)):
            if span == 0:
                continue
            attrs = f' rowspan="{span}"' if span > 1 else ""
            attrs += ' align="right"' if col in right_aligned else ""
            out.append(f"      <td{attrs}>{cell}</td>")
        out.append("    </tr>")
    out += ["  </tbody>", "</table>"]
    return "\n".join(out)


def domain_sections(tables):
    """'### <domain>' heading + table for each domain; tables maps domain to HTML (or '')."""
    out = []
    for domain, table in tables.items():
        if table:
            out += [f"### {domain}", "", table, ""]
    return "\n".join(out).rstrip()
