#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Plot the joint activation + structured sparsity sweep: activation sparsity
(x) vs. parameter count (y), with each point colored by accuracy (sequential,
single-hue) and shaped by which family it belongs to -- baseline / activation
sparsity only / structured sparsity only / joint, determined from whether
reg and prune_target are each zero or nonzero rather than by exact tag name,
so any number of follow-up points at different reg/prune_target values still
group under the right shape -- identity is carried by shape, magnitude by
color, so nothing depends on color alone. Ported from ../vww/plot_joint_sparsity.py.

Example
-------
    python plot_joint_sparsity.py --csv joint_sweep_results/joint_sweep_summary.csv \\
        --out docs/joint_sparsity_sweep.png
"""
import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# Single-hue sequential blue ramp (dataviz skill default palette), light->dark,
# used for the accuracy color encoding.
BLUE_RAMP = ['#cde2fb', '#9ec5f4', '#5598e7', '#256abf', '#0d366b']
ACCURACY_CMAP = LinearSegmentedColormap.from_list('accuracy_blue', BLUE_RAMP)

FAMILY_STYLE = {
    (False, False): {'marker': 'o', 'label': 'Baseline (no regularization)'},
    (True, False): {'marker': '^', 'label': 'Activation sparsity only'},
    (False, True): {'marker': 's', 'label': 'Structured sparsity only'},
    (True, True): {'marker': 'D', 'label': 'Joint (activation + structured)'},
}


def family_of(row):
    return FAMILY_STYLE[(float(row['reg']) > 0, float(row['prune_target']) > 0)]


def load_rows(csv_path):
    with open(csv_path, newline='') as f:
        return list(csv.DictReader(f))


def plot(rows, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 6))

    accuracies = [float(r['accuracy']) for r in rows]
    vmin, vmax = min(accuracies), max(accuracies)
    # Keep a visible spread even if all accuracies land close together.
    if vmax - vmin < 1e-6:
        vmin, vmax = vmin - 0.01, vmax + 0.01

    xs = [float(r['sparsity']) * 100 for r in rows]
    ys = [int(r['params']) for r in rows]

    for row, x, y in zip(rows, xs, ys):
        style = family_of(row)
        acc = float(row['accuracy'])
        ax.scatter(x, y, marker=style['marker'], s=180, c=[acc], cmap=ACCURACY_CMAP,
                   vmin=vmin, vmax=vmax, edgecolors='#333333', linewidths=0.8, zorder=3)

    # Parameter count depends only on prune_target, so multiple points (e.g. an
    # activation-only variant and its joint counterpart at the same prune_target)
    # often share the exact same y -- a plain fixed-offset label would grow two
    # such labels sideways into each other. Group by y and stagger labels upward
    # in increasing steps (sorted by x) within each group instead, centered
    # directly over each marker, so no two labels in the same row overlap.
    by_y = defaultdict(list)
    for row, x, y in zip(rows, xs, ys):
        by_y[y].append((x, row))
    for y, items in by_y.items():
        for i, (x, row) in enumerate(sorted(items, key=lambda t: t[0])):
            acc = float(row['accuracy'])
            ax.annotate(f"{acc * 100:.1f}% ({row['tag']})", (x, y), textcoords='offset points',
                        xytext=(0, 10 + 15 * i), ha='center', va='bottom',
                        fontsize=8, color='#333333')

    # Pad the data range so edge points and their labels stay inside the axes.
    x_pad = (max(xs) - min(xs)) * 0.15 or 5
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) * 0.85, max(ys) * 1.35)

    ax.set_xlabel('Activation sparsity (%, float model, mean over ReLU layers)')
    ax.set_ylabel('Parameter count')
    ax.set_yscale('log')
    ax.set_title(f'{title}: activation sparsity vs. parameter count\n'
                  '(marker shape = family, color + label = accuracy)')
    ax.grid(True, which='both', linestyle='-', linewidth=0.5, color='#dddddd', zorder=0)
    ax.set_axisbelow(True)

    legend_handles = [
        Line2D([0], [0], marker=style['marker'], color='w', markerfacecolor='#5598e7',
               markeredgecolor='#333333', markersize=10, label=style['label'])
        for style in FAMILY_STYLE.values()
    ]
    # Below the axes, not overlapping data -- 'best' can and did land on a point.
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.14),
              ncol=2, fontsize=9, frameon=True)

    sm = plt.cm.ScalarMappable(cmap=ACCURACY_CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Accuracy')

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Plot written to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='joint_sweep_results/joint_sweep_summary.csv')
    parser.add_argument('--out', default='docs/joint_sparsity_sweep.png')
    parser.add_argument('--title', default='PlantVillage joint sparsity sweep')
    args = parser.parse_args()

    rows = load_rows(args.csv)
    plot(rows, args.out, args.title)
