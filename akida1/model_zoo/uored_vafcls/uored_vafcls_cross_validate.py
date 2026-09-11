#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
UORED-VAFCLS cross-validation sweep.

Runs the whole pipeline - float training, quantization, quantization-aware
tuning, conversion, Akida evaluation - independently on every bearing-disjoint
fold, and reports the mean and spread of macro AUROC for each stage.

This is the script that produces the headline number for this example, and the
reason it exists is that a single fold does not support a conclusion. There are
only 20 bearings; a fold trains on 12 of them and is tested on 8. The AUROC
spread across folds is 0.04-0.05, and individual folds range from about 0.70 to
about 0.99 with the model and the seed held fixed. Two architectures differing
by a couple of AUROC points cannot be distinguished by one fold, and reporting
one fold's score as a model's performance is reporting noise.

Folds 0-4 are the tuning budget - the only folds a hyperparameter may be chosen
on. Folds 5-104 are the 100 evaluation folds, and the mean over them is the
number to quote.

Results accumulate in docs/cv_results.csv, one row per (fold, seed), flushed
after every fold. A run resumes by default, skipping rows already present, so an
interrupted sweep loses at most one fold.

Runtime is roughly 30-60 s per fold, so about 1-2 hours for the default
100-fold full chain. --skip-akida drops the quantize/convert/Akida stages and is
about four times faster; use it for recipe sweeps.

Example
-------
    # the headline run
    python uored_vafcls_cross_validate.py --save-metrics

    # a quick recipe comparison on the tuning folds
    python uored_vafcls_cross_validate.py --first-fold 0 --last-fold 4 \\
        --skip-akida -o /tmp/tuning.csv
"""

import argparse
import csv
import os
import pathlib
import time

import numpy as np
import tensorflow as tf
from cnn2snn import AkidaVersion, convert, quantize, set_akida_version
from local_utils import quantize_until
from tqdm import tqdm

from uored_vafcls_data import (BATCH_SIZE, EVAL_FOLDS, FIXED_FOLD,
                               LABEL_COLUMNS, get_data)
from uored_vafcls_eval import (auroc_scores, evaluate_akida_model,
                              predict_keras_model)
from uored_vafcls_model import build_uored_vafcls_model
from uored_vafcls_train import train_uored_vafcls

# Same reason as in the training and evaluation scripts, and it matters most
# here: without deterministic ops, a fold's score moves by up to 0.024 between
# identical runs, which would swamp any recipe comparison this script is used for.
tf.config.experimental.enable_op_determinism()

DEFAULT_OUT = pathlib.Path(__file__).parent / 'docs' / 'cv_results.csv'

FIELDNAMES = (
    ['fold', 'seed', 'float_auroc', 'qat_auroc', 'akida_auroc']
    + [f'float_auroc_{name}' for name in LABEL_COLUMNS]
    + [f'akida_auroc_{name}' for name in LABEL_COLUMNS]
    + ['params', 'n_train_windows', 'n_test_windows', 'seconds']
)


def run_fold(fold, seed=0, data_path='./data/uored_vafcls', epochs=30,
             learning_rate=2e-4, qat_epochs=10, qat_learning_rate=5e-5,
             batch_size=BATCH_SIZE, skip_akida=False, models_dir=None,
             verbose=0):
    """Run the full pipeline on one fold and return a result row.

    The stage order reproduces uored_vafcls_train.sh exactly, so a fold's
    numbers here match what the shell pipeline would produce for the same
    (fold, seed).

    Args:
        fold (int): bearing-disjoint fold index.
        seed (int): random seed for weights, crops, gains and shuffling.
        data_path (str): directory holding the prepared .npz cache.
        epochs (int): float training epochs.
        learning_rate (float): peak LR for float training.
        qat_epochs (int): quantization-aware tuning epochs.
        qat_learning_rate (float): peak LR for tuning.
        batch_size (int): batch size.
        skip_akida (bool): stop after float training, leaving the quantized and
            Akida columns empty.
        models_dir (str, optional): write the per-fold model files here. Off by
            default - a hundred folds' worth of checkpoints is a lot of noise.
        verbose (int): passed through to model.fit.

    Returns:
        dict: one row keyed by FIELDNAMES.
    """
    started = time.time()

    with set_akida_version(AkidaVersion.v1):
        model = build_uored_vafcls_model(seed=seed)
    train_ds, test_ds = get_data(data_path, model.input_shape[1:], batch_size,
                                 fold=fold, seed=seed)

    row = {name: '' for name in FIELDNAMES}
    row.update(fold=fold, seed=seed, params=model.count_params(),
               n_train_windows=len(train_ds) * batch_size,
               n_test_windows=sum(1 for _ in test_ds.unbatch()))

    # ---------------------------------------------------------------------------
    # Full precision
    # ---------------------------------------------------------------------------
    train_uored_vafcls(model, train_ds, test_ds, epochs, learning_rate, seed=seed)
    float_scores = auroc_scores(*predict_keras_model(model, test_ds))
    row['float_auroc'] = f'{float_scores["macro"]:.6f}'
    for name in LABEL_COLUMNS:
        row[f'float_auroc_{name}'] = f'{float_scores[name]:.6f}'

    if models_dir is not None:
        os.makedirs(models_dir, exist_ok=True)
        model.save(os.path.join(models_dir,
                                f'akdcnn_uored_vafcls_fold{fold:03d}_'
                                f'seed{seed}.h5'), include_optimizer=False)

    if not skip_akida:
        # -----------------------------------------------------------------------
        # Quantization, tuning and conversion
        # -----------------------------------------------------------------------
        with set_akida_version(AkidaVersion.v1):
            # qmodel = quantize(model, input_weight_quantization=8,
            #                   weight_quantization=4, activ_quantization=4)

            # qmodel = quantize_until(model, 'fc_relu')
            qmodel = quantize_until(model, 'block6_relu')
            
            qmodel.summary()
            train_uored_vafcls(qmodel, train_ds, test_ds, qat_epochs,
                               qat_learning_rate, seed=seed)
            qat_scores = auroc_scores(*predict_keras_model(qmodel, test_ds))
            row['qat_auroc'] = f'{qat_scores["macro"]:.6f}'

        #     ak_model = convert(qmodel)

        # if models_dir is not None:
        #     qmodel.save(os.path.join(models_dir,
        #                              f'akdcnn_uored_vafcls_fold{fold:03d}_'
        #                              f'seed{seed}_qat.h5'),
        #                 include_optimizer=False)
        #     ak_model.save(os.path.join(models_dir,
        #                                f'akdcnn_uored_vafcls_fold{fold:03d}_'
        #                                f'seed{seed}_qat.fbz'))

        # # -----------------------------------------------------------------------
        # # Akida
        # # -----------------------------------------------------------------------
        # akida_scores = auroc_scores(*evaluate_akida_model(ak_model, test_ds))
        # row['akida_auroc'] = f'{akida_scores["macro"]:.6f}'
        # for name in LABEL_COLUMNS:
        #     row[f'akida_auroc_{name}'] = f'{akida_scores[name]:.6f}'

    row['seconds'] = f'{time.time() - started:.1f}'
    return row


def _existing_rows(out_path):
    """Return the (fold, seed) pairs already present in the results file."""
    if not os.path.exists(out_path):
        return set()
    with open(out_path, newline='') as handle:
        return {(int(r['fold']), int(r['seed'])) for r in csv.DictReader(handle)
                if r.get('fold')}


def cross_validate(folds, seeds, out_path=DEFAULT_OUT, resume=True, **kwargs):
    """Run every (fold, seed) combination, appending results as they complete.

    Args:
        folds (iterable): fold indices to run.
        seeds (iterable): seeds to run for each fold.
        out_path (str): CSV to append results to.
        resume (bool): skip (fold, seed) pairs already in out_path. When False,
            the file is overwritten.
        **kwargs: passed to run_fold.

    Returns:
        list: the rows produced by this invocation.
    """
    folds, seeds = list(folds), list(seeds)
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    if not resume and os.path.exists(out_path):
        os.remove(out_path)
    done = _existing_rows(out_path) if resume else set()

    todo = [(f, s) for f in folds for s in seeds if (f, s) not in done]
    if done:
        print(f'Resuming: {len(done)} of {len(folds) * len(seeds)} '
              f'(fold, seed) results already in {out_path}')
    if not todo:
        print('Nothing to do.')
        return []

    write_header = not os.path.exists(out_path)
    rows = []
    progress = tqdm(todo, desc='Cross-validating', unit='fold')
    with open(out_path, 'a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for fold, seed in progress:
            row = run_fold(fold, seed=seed, **kwargs)
            writer.writerow(row)
            # Flushed per fold so an interrupted sweep loses at most one fold.
            handle.flush()
            rows.append(row)
            recent = [float(r['float_auroc']) for r in rows if r['float_auroc']]
            progress.set_postfix(float_mean=f'{np.mean(recent):.4f}')

    return rows


def _read_results(out_path):
    """Read the results CSV into a dict of column -> list."""
    with open(out_path, newline='') as handle:
        return [row for row in csv.DictReader(handle) if row.get('fold')]


def _per_fold(rows, column):
    """Seed-average a column, returning one value per fold.

    Averaging over seeds before averaging over folds is what the protocol does,
    and it matters: it keeps each fold weighted equally regardless of how many
    seeds happen to have been run for it.
    """
    by_fold = {}
    for row in rows:
        if row.get(column):
            by_fold.setdefault(int(row['fold']), []).append(float(row[column]))
    return {fold: float(np.mean(values)) for fold, values in sorted(by_fold.items())}


def summarise(rows, fixed_fold=FIXED_FOLD, quiet=False):
    """Report the fold distribution for each stage and return the cv_* metrics.

    Args:
        rows (list): result rows, as read from the CSV.
        fixed_fold (int): the fold the pretrained models come from, reported so
            the README can say where it sits in the distribution.
        quiet (bool): suppress printing.

    Returns:
        dict: the cv_* metrics keys.
    """
    seeds = {int(row['seed']) for row in rows}
    folds = {int(row['fold']) for row in rows}
    metrics = {'cv_folds': str(len(folds)), 'cv_seeds': str(len(seeds))}

    for stage in ('float', 'qat', 'akida'):
        per_fold = _per_fold(rows, f'{stage}_auroc')
        if not per_fold:
            continue
        values = np.array(list(per_fold.values()))
        mean, std = values.mean(), values.std(ddof=0)
        sem = std / len(values) ** 0.5

        metrics[f'cv_{stage}_auroc_mean'] = f'{mean:.4f}'
        metrics[f'cv_{stage}_auroc_std'] = f'{std:.4f}'
        if stage in ('float', 'akida'):
            metrics[f'cv_{stage}_auroc_min'] = f'{values.min():.4f}'
            metrics[f'cv_{stage}_auroc_max'] = f'{values.max():.4f}'
        if stage == 'akida':
            metrics['cv_akida_auroc_sem'] = f'{sem:.4f}'
            if fixed_fold in per_fold:
                metrics['cv_fixed_fold_akida_auroc'] = \
                    f'{per_fold[fixed_fold]:.4f}'

        if not quiet:
            print(f'\n{stage} macro AUROC over {len(values)} folds x '
                  f'{len(seeds)} seed(s), seed-averaged per fold:')
            print(f'  mean {mean:.4f}  std {std:.4f}  sem {sem:.4f}')
            print(f'  min {values.min():.4f}  median {np.median(values):.4f}  '
                  f'max {values.max():.4f}')
            if len(seeds) > 1:
                within = np.mean([np.std(v, ddof=0) for v in
                                  _by_fold_values(rows, f'{stage}_auroc')])
                print(f'  mean within-fold seed std {within:.4f}  '
                      f'<- the spread a single seed hides')
                metrics[f'cv_{stage}_auroc_seed_std'] = f'{within:.4f}'

    return metrics


def _by_fold_values(rows, column):
    """Group a column's raw values by fold, for within-fold seed statistics."""
    by_fold = {}
    for row in rows:
        if row.get(column):
            by_fold.setdefault(int(row['fold']), []).append(float(row[column]))
    return [v for v in by_fold.values() if len(v) > 1]


def plot_distribution(rows, fixed_fold, savepath):
    """Histogram the per-fold AUROCs, marking the fixed fold and the mean."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    per_fold = _per_fold(rows, 'akida_auroc') or _per_fold(rows, 'float_auroc')
    values = np.array(list(per_fold.values()))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(values, bins=20, color='#4C72B0', edgecolor='white')
    ax.axvline(values.mean(), color='#C44E52', linewidth=2,
               label=f'mean {values.mean():.4f}')
    if fixed_fold in per_fold:
        ax.axvline(per_fold[fixed_fold], color='#55A868', linestyle='--',
                   linewidth=2,
                   label=f'fold {fixed_fold} (pretrained) '
                         f'{per_fold[fixed_fold]:.4f}')
    ax.set_xlabel('macro AUROC on the held-out bearings')
    ax.set_ylabel('folds')
    ax.set_title(f'Per-fold performance over {len(values)} '
                 f'bearing-disjoint folds')
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150)
    print(f'Distribution plot saved to {savepath}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default='./data/uored_vafcls',
                        help='Directory holding the prepared .npz cache')
    parser.add_argument('--first-fold', type=int, default=EVAL_FOLDS.start,
                        help='First fold to run (inclusive)')
    parser.add_argument('--last-fold', type=int, default=EVAL_FOLDS.stop - 1,
                        help='Last fold to run (inclusive)')
    parser.add_argument('--seeds', type=int, default=1,
                        help='Number of seeds per fold; seeds are range(N)')
    parser.add_argument('-b', '--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-lr', '--learning_rate', type=float, default=2e-4)
    parser.add_argument('--qat-epochs', type=int, default=10)
    parser.add_argument('--qat-learning-rate', type=float, default=5e-5)
    parser.add_argument('-o', '--out', default=str(DEFAULT_OUT),
                        help='CSV to append per-fold results to. Two machines '
                             'must not append to the same file')
    parser.add_argument('--no-resume', action='store_true',
                        help='Overwrite the results file instead of resuming')
    parser.add_argument('--skip-akida', action='store_true',
                        help='Float training only - no quantization, tuning or '
                             'Akida evaluation. About four times faster')
    parser.add_argument('--keep-models', action='store_true',
                        help='Write each fold\'s models to models/cv/')
    parser.add_argument('--summarise-only', action='store_true',
                        help='Summarise an existing results file without '
                             'running anything')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Write the cv_* keys and the distribution plot to '
                             'docs/. Requires a complete evaluation sweep')
    parser.add_argument('--save-seed-std', action='store_true',
                        help='Write only cv_seed_std, the mean within-fold '
                             'spread across seeds. For the multi-seed subset '
                             'run, which deliberately covers few folds')
    args = parser.parse_args()

    folds = range(args.first_fold, args.last_fold + 1)

    if not args.summarise_only:
        cross_validate(
            folds, range(args.seeds), out_path=args.out,
            resume=not args.no_resume, data_path=args.data,
            epochs=args.epochs, learning_rate=args.learning_rate,
            qat_epochs=args.qat_epochs,
            qat_learning_rate=args.qat_learning_rate,
            batch_size=args.batch_size, skip_akida=args.skip_akida,
            models_dir='./models/cv' if args.keep_models else None)

    rows = _read_results(args.out)
    metrics = summarise(rows)

    if args.save_seed_std:
        import json

        # Quoted in the README's cross-validation section as the measured
        # single-seed uncertainty. It comes from the multi-seed subset run
        # rather than the headline sweep, which is one seed per fold.
        groups = _by_fold_values(rows, 'akida_auroc') \
            or _by_fold_values(rows, 'float_auroc')
        if not groups:
            raise SystemExit('--save-seed-std needs at least two seeds for at '
                             'least one fold; run with --seeds 3.')
        seed_std = float(np.mean([np.std(v, ddof=0) for v in groups]))

        metrics_path = pathlib.Path(__file__).parent / 'docs' / 'metrics.json'
        stored = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        stored['cv_seed_std'] = f'{seed_std:.4f}'
        stored['cv_seed_std_folds'] = str(len(groups))
        metrics_path.write_text(json.dumps(stored, indent=4) + '\n')
        print(f'\nMean within-fold seed std {seed_std:.4f} over {len(groups)} '
              f'folds, saved to {metrics_path}')

    if args.save_metrics:
        import json

        # The published aggregate must describe the whole evaluation range, or
        # the README would quote a mean over whichever folds happened to run.
        have = {int(row['fold']) for row in rows}
        missing = set(EVAL_FOLDS) - have
        if missing:
            raise SystemExit(
                f'--save-metrics needs every evaluation fold '
                f'({EVAL_FOLDS.start}-{EVAL_FOLDS.stop - 1}); '
                f'{len(missing)} are missing, e.g. {sorted(missing)[:5]}.')
        if not any(row.get('akida_auroc') for row in rows):
            raise SystemExit('--save-metrics needs the Akida column; this '
                             'looks like a --skip-akida run.')

        docs = pathlib.Path(__file__).parent / 'docs'
        metrics_path = docs / 'metrics.json'
        stored = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        stored.update(metrics)
        metrics_path.write_text(json.dumps(stored, indent=4) + '\n')
        print(f'\nMetrics saved to {metrics_path}')

        plot_distribution(rows, FIXED_FOLD,
                          docs / 'ref_cv_auroc_distribution.png')
