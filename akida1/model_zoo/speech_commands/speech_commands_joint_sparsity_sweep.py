#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Joint activation + structured sparsity sweep for Speech Commands (DS-CNN),
mirroring ../vww/vww_joint_sparsity_sweep.py and
../plant_village/plant_village_joint_sparsity_sweep.py:
  - activation sparsity: activity regularization on every ReLU layer (see
    SPARSITY_EXPERIMENT.md -- unlike VWW/PlantVillage, this project's
    established best point on this axis is *raw* hoyer_square at reg=1e-4
    (71.3% Akida-measured sparsity, ~3.1pt accuracy cost); normalized
    Hoyer-Square has not been separately validated here.
  - structured sparsity: Network Slimming channel pruning of the 3 "plain"
    separable-conv blocks' output channels (see
    speech_commands_structured_sparsity.py).

Float-only end to end (no cnn2snn quantize/QAT/convert/Akida eval) -- see
JOINT_SPARSITY_EXPERIMENT.md for why. Unlike VWW/PlantVillage, this project is
config-driven rather than CLI-flag-driven, so model/training hyperparameters
(filters, dropout, epochs, learning rate, augmentation, etc.) come from
--base-config; only the regularizer strength/type and prune target are swept.

Per grid point:
  1. build model, attach BN-gamma L1 loss (if prune_target > 0)
  2. Phase 1 float train (train_speech_commands, imported from
     speech_commands_train.py), with the activation regularizer attached if
     reg > 0
  3. prune (skipped if prune_target == 0)
  4. Phase 2 float fine-tune (skipped if prune_target == 0), re-attaching the
     activation regularizer since prune_model() rebuilds fresh layers that
     don't carry it over
  5. measure float accuracy (on the validation split, matching
     speech_commands_eval.py's own convention), float activation sparsity,
     and parameter count

Example
-------
    python speech_commands_joint_sparsity_sweep.py --data ./data/sc10

    # Follow-up: add lighter points to an existing summary CSV without re-running it
    python speech_commands_joint_sparsity_sweep.py --data ./data/sc10 \\
        --points activation_light:5e-5:0 structured_light:0:0.2 joint_light:5e-5:0.2
"""
import argparse
import csv
from pathlib import Path

import tf_keras
import yaml

from speech_commands_model import build_ds_cnn
from speech_commands_data_loader import compute_mfcc_range, get_datasets, get_samples
from speech_commands_train import train_speech_commands
from speech_commands_structured_sparsity import add_gamma_l1_loss, prune_model
from speech_commands_float_sparsity import compute_float_sparsity

PRUNABLE_LAYERS = ['separable_conv2d', 'separable_conv2d_1', 'separable_conv2d_2']
REG_TYPE = 'hoyer_square'


def build_model(cfg):
    # build_ds_cnn's conv_block/separable_conv_block calls don't pass an
    # explicit `name=`, so Keras auto-names layers with a per-process global
    # counter (conv2d, conv2d_1, conv2d_2, ...) rather than resetting per
    # model -- unlike VWW/PlantVillage's pretrained-loader path, which names
    # every layer explicitly and so is stable across repeated builds in one
    # process. Clearing the session before each build resets that counter so
    # PRUNABLE_LAYERS' names stay correct regardless of how many models this
    # process has already built.
    tf_keras.backend.clear_session()
    return build_ds_cnn(
        filters=cfg['filters'],
        dropout_initial=cfg['dropout_initial'],
        dropout_final=cfg['dropout_final'],
        weight_decay=cfg['weight_decay'],
        num_sep_conv_blocks=cfg.get('num_sep_conv_blocks', 4),
        classifier_head=cfg.get('classifier_head', 'dense'),
        seed=cfg['seed'],
    )


def run_point(reg, prune_target, args, cfg, tag_dir, train_ds, val_ds, data_transform):
    tag_dir.mkdir(parents=True, exist_ok=True)
    warmup_fraction = cfg.get('warmup_fraction', 0.1)

    model = build_model(cfg)
    if prune_target > 0:
        add_gamma_l1_loss(model, args.gamma_l1_strength,
                           [f'{name}/BN' for name in PRUNABLE_LAYERS])

    # Phase 1: float train
    train_speech_commands(model, train_ds, val_ds, epochs=cfg['epochs_float'],
                           peak_lr=cfg['lr_float'], warmup_fraction=warmup_fraction,
                           act_reg_strength=reg, reg_type=args.reg_type, seed=cfg['seed'])

    final_model = model
    if prune_target > 0:
        final_model = prune_model(model, prune_target, PRUNABLE_LAYERS)
        # Phase 2: fine-tune post-surgery to recover accuracy lost to channel removal
        train_speech_commands(final_model, train_ds, val_ds, epochs=args.phase2_epochs,
                               peak_lr=args.phase2_lr, warmup_fraction=warmup_fraction,
                               act_reg_strength=reg, reg_type=args.reg_type, seed=cfg['seed'])

    final_model.compile(metrics=['accuracy'])
    _, accuracy = final_model.evaluate(val_ds, verbose=0)

    samples = get_samples(args.data, data_transform=data_transform,
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
    p.add_argument('--data', default='./data/sc10', help='Speech Commands tfds data directory')
    p.add_argument('--base-config', default='configs/training_cfg.yml',
                    help='Base config to read model/training hyperparameters from '
                         '(filters, dropout, epochs, lr, augmentation, etc.)')
    p.add_argument('--reg', type=float, default=1e-4,
                    help=f'{REG_TYPE} strength for the activation-sparsity corner '
                         '(established best point from SPARSITY_EXPERIMENT.md)')
    p.add_argument('--reg-type', choices=['l1l2', 'hoyer_square', 'hoyer_square_norm'],
                    default=REG_TYPE,
                    help='Regularizer type (raw hoyer_square is this project\'s own '
                         'established best -- normalized has not been separately validated)')
    p.add_argument('--prune-target', type=float, default=0.4,
                    help='Fraction of output channels to prune per targeted layer, for '
                         'the structured-sparsity corner (starting guess, no prior data '
                         'point -- expect to revisit)')
    p.add_argument('--gamma-l1-strength', type=float, default=1e-5,
                    help='L1 strength on BN gamma during Phase 1, when prune_target > 0 '
                         '(starting guess, reused from the VWW/PlantVillage experiments)')
    p.add_argument('--phase2-epochs', type=int, default=15,
                    help='Post-prune fine-tune epochs (skipped when prune_target == 0)')
    p.add_argument('--phase2-lr', type=float, default=4e-4,
                    help='Post-prune fine-tune learning rate (matches this config\'s lr_qat scale)')
    p.add_argument('--num-sparsity-samples', type=int, default=1000)
    p.add_argument('--out-dir', default='joint_sweep_results')
    p.add_argument('--csv', default=None, help='Summary CSV path (default: <out-dir>/joint_sweep_summary.csv)')
    p.add_argument('--points', nargs='+', default=None,
                    help='Explicit follow-up grid points as "tag:reg:prune_target" (e.g. '
                         '"joint_light:5e-5:0.2"), run in addition to whatever is already in '
                         '--csv. If omitted, runs the default 2x2 pilot grid instead.')
    args = p.parse_args()

    with open(args.base_config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(args.out_dir)
    csv_path = Path(args.csv) if args.csv else out_dir / 'joint_sweep_summary.csv'
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = ['tag', 'reg', 'prune_target', 'accuracy', 'sparsity', 'params']
    rows = []
    if args.points:
        if csv_path.exists():
            with open(csv_path, newline='') as f:
                rows = list(csv.DictReader(f))
        grid = [(tag, float(reg), float(prune)) for tag, reg, prune in
                (pt.split(':') for pt in args.points)]
    else:
        grid = [
            ('baseline', 0.0, 0.0),
            ('activation_only', args.reg, 0.0),
            ('structured_only', 0.0, args.prune_target),
            ('joint', args.reg, args.prune_target),
        ]

    data_transform = compute_mfcc_range(data_dir=args.data)
    train_ds, _test_ds, val_ds = get_datasets(
        data_dir=args.data, batch_size=cfg['batch_size'], data_transform=data_transform,
        aug_enabled=cfg.get('aug_enabled', False),
        aug_time_shift_max_ms=cfg.get('aug_time_shift_max_ms', 100),
        aug_freq_mask_param=cfg.get('aug_freq_mask_param', 2),
        aug_time_mask_param=cfg.get('aug_time_mask_param', 10),
        shuffle_seed=cfg.get('seed'), aug_seed=cfg.get('seed'),
    )

    for tag, reg, prune_target in grid:
        print(f"\n==== {tag} (reg={reg:g}, prune_target={prune_target:g}) ====\n")
        row = run_point(reg, prune_target, args, cfg, out_dir / tag, train_ds, val_ds,
                         data_transform)
        row['tag'] = tag
        rows.append(row)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSummary written to {csv_path}")


if __name__ == '__main__':
    main()
