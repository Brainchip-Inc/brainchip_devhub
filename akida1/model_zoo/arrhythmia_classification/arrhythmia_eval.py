#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Arrhythmia classification evaluation for tf_keras or akida models.

Defaults to the inter-patient test split (DS2 records), which is the number
worth quoting: none of those patients appear in the training data.

Accuracy on its own is a poor summary here, because 90% of the test beats are
normal - a model that predicted "normal" for everything would score 0.90. The
per-class report is printed alongside it, and the supraventricular (S) row is
the one that separates a useful model from a lazy one.

Example
-------
    python arrhythmia_eval.py -d ./data/mitdb -l models/arrhythmia_classification.h5
"""
import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf

from tqdm import tqdm

import akida

from cnn2snn import load_quantized_model
from sklearn.metrics import classification_report, confusion_matrix

from arrhythmia_data import TARGET_NAMES, get_data, get_test_data
from brainchip_utils.hardware_utils import get_akida_device

tf.config.experimental.enable_op_determinism()


# ---------------------------------------------------------------------------
# Evaluation on Akida
# ---------------------------------------------------------------------------
def evaluate_akida_model(akida_model, dataset):
    """Run inference with an Akida model and return (predictions, labels)."""
    device = get_akida_device(target_version=akida_model.ip_version)
    if device is not None:
        akida_model.map(device, mode=akida.MapMode.Minimal)
        print('Running inference on Akida hardware device')
        akida_model.summary()

    labels_all = None
    logits_all = None

    # Akida can't directly digest the tensorflow dataset, we need to
    # manually iterate over the dataset to deliver inputs as numpy arrays.
    for batch, label_batch in tqdm(dataset, desc='Evaluating on Akida'):
        batch = batch.numpy()
        label_batch = label_batch.numpy()

        # Inference on Akida
        logits_batch = akida_model.predict(batch.astype(np.uint8))
        logits_batch = logits_batch.squeeze(axis=(1, 2))  # (B, 1, 1, C) -> (B, C)

        if labels_all is None:
            labels_all = label_batch
            logits_all = logits_batch
        else:
            labels_all = np.concatenate([labels_all, label_batch])
            logits_all = np.concatenate([logits_all, logits_batch])

    preds = np.argmax(logits_all, axis=1)
    accuracy = np.mean(np.equal(preds, labels_all))
    print(f'Akida accuracy: {accuracy:.4f}')
    return preds, labels_all


def predict_keras_model(model, dataset):
    """Run inference with a tf_keras model and return (predictions, labels)."""
    logits = model.predict(dataset, verbose=0)
    labels = np.concatenate([label_batch.numpy() for _, label_batch in dataset])
    return np.argmax(logits, axis=1), labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras or .fbz akida model)')
    parser.add_argument('-d', '--data', default='./data/mitdb',
                        help='MIT-BIH record directory')
    parser.add_argument('--split', choices=['test', 'val'], default='test',
                        help='Evaluate on the inter-patient test records '
                             '(default) or the training hold-out')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Write accuracy (and param count for .h5) to metrics.json')
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    if args.loadmodel.endswith('.h5'):
        model = load_quantized_model(args.loadmodel)
        model.compile(metrics=['accuracy'])
        isakida = False
        imsize = model.input_shape[1:]
    elif args.loadmodel.endswith('.fbz'):
        model = akida.Model(args.loadmodel)
        isakida = True
        imsize = tuple(model.input_shape)
    else:
        raise ValueError(f'Unsupported model format: {args.loadmodel}')

    # ---------------------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------------------
    if args.split == 'test':
        dataset = get_test_data(args.data, imsize, batch_size=64)
    else:
        _, dataset = get_data(args.data, imsize, batch_size=64)

    # ---------------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------------
    if isakida:
        preds, labels = evaluate_akida_model(model, dataset)
    else:
        preds, labels = predict_keras_model(model, dataset)

    accuracy = float(np.mean(np.equal(preds, labels)))
    print(f'\n{args.split.capitalize()} accuracy: {accuracy:.4f}')

    report = classification_report(labels, preds, target_names=list(TARGET_NAMES),
                                   digits=4, zero_division=0)
    print(f'\n--- Per-class report ({args.split}) ---')
    print(report)
    print('Confusion matrix (rows = true, columns = predicted):')
    print(f'      {"  ".join(f"{n:>7}" for n in TARGET_NAMES)}')
    for name, row in zip(TARGET_NAMES, confusion_matrix(labels, preds)):
        print(f'  {name}   {"  ".join(f"{v:>7}" for v in row)}')

    scores = classification_report(labels, preds, target_names=list(TARGET_NAMES),
                                   output_dict=True, zero_division=0)
    macro_f1 = scores['macro avg']['f1-score']
    print(f'\nMacro F1: {macro_f1:.4f}')

    # ---------------------------------------------------------------------------
    # Persist metrics
    # ---------------------------------------------------------------------------
    if args.save_metrics:
        # The is used to update the stored metrics that are used to generate the
        # performance tables in the README of this folder.
        # This should only be used for code maintenance, when the model or training
        # pipeline is updated and a new trained model integrated.
        metrics_path = pathlib.Path(__file__).parent / 'docs' / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        acc_str = f'{accuracy * 100:.2f}%'
        f1_str = f'{macro_f1:.3f}'
        if isakida:
            metrics['akida_acc'] = acc_str
            metrics['akida_f1'] = f1_str
            # The per-class table is only reported for the deployed Akida model.
            for name in TARGET_NAMES:
                key = name.lower()
                metrics[f'akida_{key}_precision'] = f'{scores[name]["precision"]:.3f}'
                metrics[f'akida_{key}_recall'] = f'{scores[name]["recall"]:.3f}'
                metrics[f'akida_{key}_f1'] = f'{scores[name]["f1-score"]:.3f}'
                metrics[f'akida_{key}_support'] = f'{int(scores[name]["support"]):,}'
        elif 'qat' in pathlib.Path(args.loadmodel).stem:
            metrics['qat_acc'] = acc_str
            metrics['qat_f1'] = f1_str
        else:
            metrics['float_acc'] = acc_str
            metrics['float_f1'] = f1_str
            metrics['params'] = f'{model.count_params():,}'
        metrics_path.write_text(json.dumps(metrics, indent=4) + '\n')
        print(f'Metrics saved to {metrics_path}')
