#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
2x2 pilot sweep over VWW's two independent sparsity axes:
  - activation sparsity: HoyerSquare(normalize=True) activity regularization on
    every ReLU layer (see SPARSITY_EXPERIMENT.md -- reg=3.6 is the established
    best point, 70.4% sparsity for a 2.6pt accuracy cost).
  - structured sparsity: Network Slimming channel pruning of separable_4..
    separable_12's output channels (see vww_structured_sparsity.py).

Float-only end to end (no cnn2snn quantize/QAT/convert/Akida eval) -- see
JOINT_SPARSITY_EXPERIMENT.md for why. Per grid point:
  1. build model, attach BN-gamma L1 loss (if prune_target > 0)
  2. Phase 1 float train, with the activation regularizer attached if reg > 0
     (train_vww, imported from vww_train.py)
  3. prune (skipped if prune_target == 0)
  4. Phase 2 float fine-tune (skipped if prune_target == 0), re-attaching the
     activation regularizer since prune_model() rebuilds fresh layers that
     don't carry it over
  5. measure float accuracy, float activation sparsity, and parameter count

Example
-------
    python vww_joint_sparsity_sweep.py --data ./data/vw_coco2014_96

    # Follow-up: add lighter points to an existing summary CSV without re-running it
    python vww_joint_sparsity_sweep.py --data ./data/vw_coco2014_96 \\
        --points activation_light:1.5:0 structured_light:0:0.2 joint_light:1.5:0.2
"""
import argparse
import csv
from pathlib import Path

from vww_model import build_vww_model
from vww_data import get_data, get_samples
from vww_train import train_vww
from vww_structured_sparsity import add_gamma_l1_loss, prune_model
from vww_float_sparsity import compute_float_sparsity

PRUNABLE_LAYERS = [f'separable_{i}' for i in range(4, 13)]
REG_TYPE = 'hoyer_square_norm'


def run_point(reg, prune_target, args, tag_dir, train_ds, val_ds):
    tag_dir.mkdir(parents=True, exist_ok=True)
    reg_kwargs = dict(regularization=reg if reg > 0 else None, reg_type=REG_TYPE)

    model = build_vww_model(seed=args.seed)
    if prune_target > 0:
        add_gamma_l1_loss(model, args.gamma_l1_strength,
                           [f'{name}/BN' for name in PRUNABLE_LAYERS])

    # Phase 1: float train
    train_vww(model, train_ds, val_ds, epochs=args.phase1_epochs,
              learning_rate=args.phase1_lr, seed=args.seed, **reg_kwargs)

    final_model = model
    if prune_target > 0:
        final_model = prune_model(model, prune_target, PRUNABLE_LAYERS)
        # Phase 2: fine-tune post-surgery to recover accuracy lost to channel removal
        train_vww(final_model, train_ds, val_ds, epochs=args.phase2_epochs,
                  learning_rate=args.phase2_lr, seed=args.seed, **reg_kwargs)

    final_model.compile(metrics=['accuracy'])
    _, accuracy = final_model.evaluate(val_ds, verbose=0)

    samples = get_samples(args.data, final_model.input_shape[1:],
                           num_samples=args.num_sparsity_samples)
    _, sparsity = compute_float_sparsity(final_model, samples)
    params = final_model.count_params()

    final_model.save(tag_dir / 'model.h5', include_optimizer=False)
    print(f"reg={reg:g} prune_target={prune_target:g} -> "
          f"accuracy={accuracy:.4f} sparsity={sparsity:.4f} params={params}")
    return {'reg': reg, 'prune_target': prune_target, 'accuracy': accuracy,
            'sparsity': sparsity, 'params': params}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default='./data/vw_coco2014_96', help='VWW dataset root')
    p.add_argument('--reg', type=float, default=3.6,
                    help=f'{REG_TYPE} strength for the activation-sparsity corner '
                         '(established best point from SPARSITY_EXPERIMENT.md)')
    p.add_argument('--prune-target', type=float, default=0.4,
                    help='Fraction of output channels to prune per targeted layer, for '
                         'the structured-sparsity corner (starting guess, no prior data '
                         'point -- expect to revisit)')
    p.add_argument('--gamma-l1-strength', type=float, default=1e-5,
                    help='L1 strength on BN gamma during Phase 1, when prune_target > 0 '
                         '(starting guess -- expect to revisit)')
    p.add_argument('--phase1-epochs', type=int, default=40)
    p.add_argument('--phase1-lr', type=float, default=1e-3)
    p.add_argument('--phase2-epochs', type=int, default=15,
                    help='Post-prune fine-tune epochs (skipped when prune_target == 0)')
    p.add_argument('--phase2-lr', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-sparsity-samples', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-dir', default='joint_sweep_results')
    p.add_argument('--csv', default=None, help='Summary CSV path (default: <out-dir>/joint_sweep_summary.csv)')
    p.add_argument('--points', nargs='+', default=None,
                    help='Explicit follow-up grid points as "tag:reg:prune_target" (e.g. '
                         '"joint_light:1.5:0.2"), run in addition to whatever is already in '
                         '--csv. If omitted, runs the default 2x2 pilot grid instead.')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    csv_path = Path(args.csv) if args.csv else out_dir / 'joint_sweep_summary.csv'
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ['tag', 'reg', 'prune_target', 'accuracy', 'sparsity', 'params']
    rows = []
    if args.points:
        # Follow-up sweep: keep whatever's already in the CSV (e.g. the original
        # pilot's 4 points) and only run the new points requested here.
        if csv_path.exists():
            with open(csv_path, newline='') as f:
                rows = list(csv.DictReader(f))
        grid = [(tag, float(reg), float(prune)) for tag, reg, prune in
                (p.split(':') for p in args.points)]
    else:
        # Default 2x2 grid: baseline / activation-only / structured-only / joint
        grid = [
            ('baseline', 0.0, 0.0),
            ('activation_only', args.reg, 0.0),
            ('structured_only', 0.0, args.prune_target),
            ('joint', args.reg, args.prune_target),
        ]

    train_ds, val_ds = get_data(args.data, (96, 96, 3), args.batch_size, seed=args.seed)

    for tag, reg, prune_target in grid:
        print(f"\n==== {tag} (reg={reg:g}, prune_target={prune_target:g}) ====\n")
        row = run_point(reg, prune_target, args, out_dir / tag, train_ds, val_ds)
        row['tag'] = tag
        rows.append(row)

        # write incrementally so a crash mid-sweep doesn't lose earlier points
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSummary written to {csv_path}")


if __name__ == '__main__':
    main()
