#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
UORED-VAFCLS evaluation for tf_keras or akida models.

The reported metric is macro AUROC over the four fault labels, not accuracy.
Healthy bearings are encoded as the all-zero label vector rather than a fifth
class, so there is no argmax prediction to be right or wrong about: any accuracy
or F1 figure would depend on a decision threshold nobody has chosen. AUROC is
threshold-free and is what the protocol this example follows reports.

Scores from a single fold are of limited use on their own - the spread across
folds is far wider than the difference between models worth comparing. Use
uored_vafcls_cross_validate.py for a number that means something.

Example
-------
    python uored_vafcls_eval.py -d ./data/uored_vafcls \\
        -l models/akdcnn_uored_vafcls.h5
"""
import argparse
import json
import pathlib

import akida
import numpy as np
import tensorflow as tf
from cnn2snn import load_quantized_model
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from akida_models.sparsity import compute_sparsity
from brainchip_utils.hardware_utils import get_akida_device
from brainchip_utils.plot_utils import pretty_print_sparsity
from uored_vafcls_data import (FIXED_FOLD, LABEL_COLUMNS, BATCH_SIZE,
                               get_samples, get_test_data)

tf.config.experimental.enable_op_determinism()

# Windows used to measure activation sparsity. Held under the 360 deterministic
# training windows a fold provides, so the measurement is repeatable.
NUM_SPARSITY_SAMPLES = 240


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def auroc_scores(logits, labels):
    """Macro and per-label AUROC.

    Uses the same scikit-learn function the reference protocol uses, so the
    metric can never be the source of a discrepancy between this port and the
    published numbers.

    Args:
        logits (np.ndarray): raw model outputs, shape (N, len(LABEL_COLUMNS)).
        labels (np.ndarray): binary indicator labels, same shape.

    Returns:
        dict: 'macro' plus one entry per label in LABEL_COLUMNS.
    """
    per_label = roc_auc_score(labels, logits, average=None)
    scores = {'macro': float(roc_auc_score(labels, logits, average='macro'))}
    scores.update({name: float(v) for name, v in zip(LABEL_COLUMNS, per_label)})
    return scores


def report(scores, split_label):
    """Print the macro and per-label AUROC."""
    print(f'\n{split_label} macro AUROC: {scores["macro"]:.4f}')
    print('Per-label AUROC:')
    for name in LABEL_COLUMNS:
        print(f'  {name:<6} {scores[name]:.4f}')


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_keras_model(model, dataset):
    """Run inference with a tf_keras model and return (logits, labels)."""
    logits_all, labels_all = [], []
    for batch, label_batch in dataset:
        logits_all.append(model.predict(batch, verbose=0))
        labels_all.append(label_batch.numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


def evaluate_akida_model(akida_model, val_dataset):
    """Run inference with an Akida model and return (logits, labels).

    Returns logits rather than predictions: AUROC is rank-based and multi-label,
    so thresholding here would throw away exactly the information it needs.
    """
    device = get_akida_device(target_version=akida_model.ip_version)
    if device is not None:
        akida_model.map(device, mode=akida.MapMode.Minimal)
        print('Running inference on Akida hardware device')
        akida_model.summary()

    labels_all = None
    logits_all = None

    # Akida can't directly digest the tensorflow dataset, we need to
    # manually iterate over the dataset to deliver inputs as numpy arrays
    for batch, label_batch in tqdm(val_dataset, desc="Evaluating on Akida"):
        if not isinstance(batch, np.ndarray):
            batch = batch.numpy()

        # Inference on Akida
        logits_batch = akida_model.predict(batch)
        logits_batch = logits_batch.squeeze(axis=(1, 2))  # (B, 1, 1, C) -> (B, C)

        if labels_all is None:
            labels_all = label_batch
            logits_all = logits_batch
        else:
            labels_all = np.concatenate([labels_all, label_batch])
            logits_all = np.concatenate([logits_all, logits_batch])

    return logits_all, np.asarray(labels_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras or .fbz akida model)')
    parser.add_argument('-d', '--data', default='./data/uored_vafcls',
                        help='Directory holding the prepared .npz cache')
    parser.add_argument('-b', '--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--fold', type=int, default=FIXED_FOLD,
                        help='Bearing-disjoint fold to evaluate on')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for the sparsity sample draw')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Write AUROC (and param count for .h5) to '
                             'metrics.json. Only valid on the fixed fold')
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    if args.loadmodel.endswith('.h5'):
        model = load_quantized_model(args.loadmodel)
        isakida = False
        imsize = model.input_shape[1:]
    elif args.loadmodel.endswith('.fbz'):
        model = akida.Model(args.loadmodel)
        isakida = True
        imsize = tuple(model.input_shape)
    else:
        raise ValueError(f'Unsupported model file: {args.loadmodel}')

    # ---------------------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------------------
    test_ds = get_test_data(args.data, imsize, batch_size=args.batch_size,
                            fold=args.fold)

    # ---------------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------------
    if isakida:
        logits, labels = evaluate_akida_model(model, test_ds)
    else:
        logits, labels = predict_keras_model(model, test_ds)

    scores = auroc_scores(logits, labels)
    report(scores, f'Fold {args.fold} held-out')

    # ---------------------------------------------------------------------------
    # Activation sparsity
    # ---------------------------------------------------------------------------
    sparsity = None
    if isakida:
        samples = get_samples(args.data, imsize,
                              num_samples=NUM_SPARSITY_SAMPLES,
                              fold=args.fold, seed=args.seed)
        sparsity_dict = compute_sparsity(model, samples=samples)
        pretty_print_sparsity(sparsity_dict)
        sparsity = float(np.mean(list(sparsity_dict.values())))
        print(f'Mean activation sparsity over {NUM_SPARSITY_SAMPLES} windows: '
              f'{sparsity * 100:.2f}%')

    # ---------------------------------------------------------------------------
    # Persist metrics
    # ---------------------------------------------------------------------------
    if args.save_metrics:
        # The is used to update the stored metrics that are used to generate the
        # performance tables in the README of this folder.
        # This should only be used for code maintenance, when the model or training
        # pipeline is updated and a new trained model integrated.
        #
        # Guarded on the fold: the README's reference table is specifically the
        # fixed fold, and a stray --fold would otherwise silently replace it
        # with a number from a different held-out set.
        if args.fold != FIXED_FOLD:
            raise SystemExit(
                f'--save-metrics writes the README reference table, which is '
                f'fold {FIXED_FOLD}; refusing to write results from fold '
                f'{args.fold}.')

        metrics_path = pathlib.Path(__file__).parent / 'docs' / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        auroc_str = f'{scores["macro"]:.4f}'
        metrics['fixed_fold'] = str(args.fold)
        if isakida:
            metrics['akida_auroc'] = auroc_str
            metrics['sparsity'] = f'{sparsity * 100:.2f}%'
            # The per-label breakdown is only reported for the deployed model.
            for name in LABEL_COLUMNS:
                metrics[f'akida_auroc_{name}'] = f'{scores[name]:.4f}'
        elif 'qat' in pathlib.Path(args.loadmodel).stem:
            metrics['qat_auroc'] = auroc_str
        else:
            metrics['float_auroc'] = auroc_str
            metrics['params'] = f'{model.count_params():,}'
        metrics_path.write_text(json.dumps(metrics, indent=4) + '\n')
        print(f'Metrics saved to {metrics_path}')
