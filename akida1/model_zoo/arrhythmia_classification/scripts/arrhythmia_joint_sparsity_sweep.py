#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Joint activation + structured sparsity sweep for ECG Arrhythmia
Classification, mirroring ../../vww/vww_joint_sparsity_sweep.py,
../../plant_village/plant_village_joint_sparsity_sweep.py, and
../../speech_commands/speech_commands_joint_sparsity_sweep.py:
  - activation sparsity: activity regularization on every ReLU layer (see
    SPARSITY_EXPERIMENT.md -- normalized Hoyer-Square at reg=3 is the
    established best point on this axis alone, 68.2% Akida-measured
    sparsity for only a ~0.9pt accuracy cost from the unregularized
    baseline -- the smallest cost of any of the four model-zoo examples).
  - structured sparsity: Network Slimming channel pruning of block1_sepconv
    and block2_sepconv's output channels (see
    arrhythmia_structured_sparsity.py). block3_sepconv (the final feature
    layer, feeding the classifier head via GlobalAveragePooling2D) is
    excluded, same reasoning as the other three examples.

Float-only end to end (no cnn2snn quantize/QAT/convert/Akida eval) -- see
JOINT_SPARSITY_EXPERIMENT.md for why. Unlike the other three examples, this
project's own train.py runs float training, quantization, and QAT all in one
process/script rather than exposing a reusable training function, so this
sweep calls build_akida_model/apply_activity_regularizer directly and
implements its own simple training loop (float-only, so no quantize/QAT
stage is needed regardless).

Per grid point:
  1. build model, attach BN-gamma L1 loss (if prune_target > 0)
  2. Phase 1 float train, with the activation regularizer attached if reg > 0
     (reusing apply_activity_regularizer from model.py), reloading the
     best-val-loss checkpoint after fit (matching train.py's own
     ModelCheckpoint(save_best_only=True) + EarlyStopping(restore_best_weights
     =False) pattern -- the live in-memory model after fit() is not
     necessarily the best one seen)
  3. prune (skipped if prune_target == 0)
  4. Phase 2 float fine-tune (skipped if prune_target == 0), re-attaching the
     activation regularizer since prune_model() rebuilds fresh layers that
     don't carry it over
  5. measure float test accuracy, float activation sparsity, and parameter
     count

Example
-------
    python arrhythmia_joint_sparsity_sweep.py --data-dir ../data/processed

    # Follow-up: add lighter points to an existing summary CSV without re-running it
    python arrhythmia_joint_sparsity_sweep.py --data-dir ../data/processed \\
        --points activation_light:1.5:0 structured_light:0:0.2 joint_light:1.5:0.2
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tf_keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tf_keras.losses import SparseCategoricalCrossentropy
from tf_keras.models import load_model
from tf_keras.optimizers import Adam

from model import build_akida_model, apply_activity_regularizer
from data import ECGDatasetLoader
from arrhythmia_structured_sparsity import add_gamma_l1_loss, prune_model
from arrhythmia_float_sparsity import compute_float_sparsity

INPUT_SHAPE = (36, 32, 1)
NUM_CLASSES = 3
KERNEL_L2_REG = 2e-5  # fixed kernel weight-decay strength, matches train.py's WEIGHT_L2_REG -- not swept
CLASS_WEIGHTS = {0: 1.0, 1: 6.0, 2: 3.0}  # matches train.py, pre-existing/untouched
PRUNABLE_LAYERS = ['block1_sepconv', 'block2_sepconv']
REG_TYPE = 'hoyer_square_norm'


def train_phase(model, X_train, y_train, X_val, y_val, epochs, lr, reg, ckpt_path):
    if reg > 0:
        apply_activity_regularizer(model, reg, REG_TYPE)
    model.compile(optimizer=Adam(lr), loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    callbacks = [
        ModelCheckpoint(filepath=str(ckpt_path), monitor='val_loss', save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=0),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=False),
    ]
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs,
              batch_size=64, class_weight=CLASS_WEIGHTS, callbacks=callbacks, verbose=1)
    # The live post-fit model isn't necessarily the best one seen (restore_best_weights=
    # False, matching train.py's own convention) -- reload the checkpointed best instead.
    return load_model(str(ckpt_path))


def run_point(reg, prune_target, args, tag_dir, X_train, y_train, X_val, y_val, X_test, y_test):
    tag_dir.mkdir(parents=True, exist_ok=True)

    model = build_akida_model(INPUT_SHAPE, NUM_CLASSES, KERNEL_L2_REG)
    if prune_target > 0:
        add_gamma_l1_loss(model, args.gamma_l1_strength, PRUNABLE_LAYERS)

    # Phase 1: float train
    model = train_phase(model, X_train, y_train, X_val, y_val, args.phase1_epochs,
                         args.phase1_lr, reg, tag_dir / 'phase1_best.h5')

    final_model = model
    if prune_target > 0:
        final_model = prune_model(model, prune_target, PRUNABLE_LAYERS)
        # Phase 2: fine-tune post-surgery to recover accuracy lost to channel removal
        final_model = train_phase(final_model, X_train, y_train, X_val, y_val,
                                   args.phase2_epochs, args.phase2_lr, reg,
                                   tag_dir / 'phase2_best.h5')

    preds = np.argmax(final_model.predict(X_test, verbose=0), axis=1)
    accuracy = accuracy_score(y_test, preds)

    sparsity_samples = X_test[:args.num_sparsity_samples].astype(np.float32)
    _, sparsity = compute_float_sparsity(final_model, sparsity_samples)
    params = final_model.count_params()

    final_model.save(tag_dir / 'model.h5')
    print(f"reg={reg:g} prune_target={prune_target:g} -> "
          f"accuracy={accuracy:.4f} sparsity={sparsity:.4f} params={params}")
    return {'reg': reg, 'prune_target': prune_target, 'accuracy': accuracy,
            'sparsity': sparsity, 'params': params}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', default='../data/processed',
                    help='Directory with preprocessed X_train.npy/y_train.npy/etc.')
    p.add_argument('--reg', type=float, default=3,
                    help=f'{REG_TYPE} strength for the activation-sparsity corner '
                         '(established best point from SPARSITY_EXPERIMENT.md)')
    p.add_argument('--prune-target', type=float, default=0.4,
                    help='Fraction of output channels to prune per targeted layer, for '
                         'the structured-sparsity corner (starting guess, no prior data '
                         'point -- expect to revisit)')
    p.add_argument('--gamma-l1-strength', type=float, default=1e-5,
                    help='L1 strength on BN gamma during Phase 1, when prune_target > 0 '
                         '(starting guess, reused from the other three experiments)')
    p.add_argument('--phase1-epochs', type=int, default=80,
                    help='Matches train.py\'s float-training default (early-stopping-bounded)')
    p.add_argument('--phase1-lr', type=float, default=3e-3, help='Matches train.py\'s default')
    p.add_argument('--phase2-epochs', type=int, default=40,
                    help='Post-prune fine-tune epochs (skipped when prune_target == 0)')
    p.add_argument('--phase2-lr', type=float, default=3e-4)
    p.add_argument('--num-sparsity-samples', type=int, default=1000)
    p.add_argument('--out-dir', default='../joint_sweep_results',
                    help='Run from inside scripts/ -- matches arrhythmia_sparsity_sweep.py\'s convention')
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

    loader = ECGDatasetLoader(data_dir=args.data_dir)
    X_train, y_train, X_val, y_val, X_test, y_test = loader.load_dataset()
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    SEED = 67004546
    for tag, reg, prune_target in grid:
        print(f"\n==== {tag} (reg={reg:g}, prune_target={prune_target:g}) ====\n")
        tf.keras.utils.set_random_seed(SEED)
        row = run_point(reg, prune_target, args, out_dir / tag, X_train, y_train,
                         X_val, y_val, X_test, y_test)
        row['tag'] = tag
        rows.append(row)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSummary written to {csv_path}")


if __name__ == '__main__':
    main()
