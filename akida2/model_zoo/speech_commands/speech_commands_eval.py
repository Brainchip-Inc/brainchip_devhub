#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Speech Commands Keyword Spotting (KWS) evaluation for tf_keras or akida models.

Example
-------
    python speech_commands_eval.py -l models/ds_cnn_speech_commands.h5
"""
import argparse
import json
import pathlib
import numpy as np
import tensorflow as tf

from tqdm import tqdm

import akida

from cnn2snn import load_quantized_model

from speech_commands_data import get_data, compute_mfcc_range
from brainchip_utils.hardware_utils import get_akida_device

tf.config.experimental.enable_op_determinism()


# ---------------------------------------------------------------------------
# Evaluation on Akida
# ---------------------------------------------------------------------------
def evaluate_akida_model(akida_model, val_dataset):
    """Run inference with an Akida model and return (predictions, labels)."""
    device = get_akida_device(target_version=akida_model.ip_version)
    if device is not None:
        akida_model.map(device, mode=akida.MapMode.Minimal)
        print('Running inference on Akida hardware device')
        akida_model.summary()

    labels_all = None
    logits_all = None

    # Akida can't directly digest the tf.data dataset; iterate manually and
    # deliver inputs as uint8 numpy arrays.
    for batch, label_batch in tqdm(val_dataset, desc="Evaluating on Akida"):
        if not isinstance(batch, np.ndarray):
            batch = batch.numpy()
        if not isinstance(label_batch, np.ndarray):
            label_batch = label_batch.numpy()

        logits_batch = akida_model.predict(batch.astype(np.uint8))
        logits_batch = logits_batch.squeeze(axis=(1, 2))  # (B, 1, 1, C) -> (B, C)

        if labels_all is None:
            labels_all = label_batch
            logits_all = logits_batch
        else:
            labels_all = np.concatenate([labels_all, label_batch])
            logits_all = np.concatenate([logits_all, logits_batch])

    preds = np.argmax(logits_all, axis=1)
    accuracy = np.mean(np.equal(np.array(preds), np.array(labels_all)))
    print(f'Akida accuracy: {accuracy:.4f}')
    return preds, labels_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras or .fbz akida model)')
    parser.add_argument('-d', '--data', default='./data/speech_commands',
                        help='tfds data_dir for the speech_commands dataset')
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
    elif args.loadmodel.endswith('.fbz'):
        model = akida.Model(args.loadmodel)
        isakida = True

    # ---------------------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------------------
    data_transform = compute_mfcc_range(data_dir=args.data)
    _, _, val_ds = get_data(
        args.data, batch_size=100, data_transform=data_transform)

    # ---------------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------------
    if isakida:
        preds, labels = evaluate_akida_model(model, val_ds)
        accuracy = float(np.mean(np.equal(preds, labels)))
    else:
        _, accuracy = model.evaluate(val_ds, verbose=0)
        print(f'Validation accuracy: {accuracy:.4f}')

    # ---------------------------------------------------------------------------
    # Persist metrics
    # ---------------------------------------------------------------------------
    if args.save_metrics:
        # Updates the stored metrics used to generate the README performance
        # tables. For code maintenance only, run against the pretrained models.
        #
        # Two quantized variants, disambiguated by filename:
        #   ds_cnn_speech_commands.h5                    -> float_acc, params
        #   ds_cnn_speech_commands_i8_w8_a8.{h5,fbz}     -> w8a8_quant_acc / w8a8_akida_acc
        #   ds_cnn_speech_commands_i8_w4_a4_qat.{h5,fbz} -> w4a4_qat_quant_acc / w4a4_qat_akida_acc
        # (The 4-bit PTQ model is a throwaway on the way to QAT -- never stored.)
        metrics_path = pathlib.Path(__file__).parent / 'docs' / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        acc_str = f'{accuracy * 100:.2f}%'
        stem = pathlib.Path(args.loadmodel).stem

        if 'i8_w8_a8' in stem:
            variant = 'w8a8'
        elif 'i8_w4_a4' in stem:
            variant = 'w4a4_qat'
        else:
            variant = None  # float model

        if variant is None:
            metrics['float_acc'] = acc_str
            metrics['params'] = f'{model.count_params():,}'
        elif isakida:
            metrics[f'{variant}_akida_acc'] = acc_str
        else:
            metrics[f'{variant}_quant_acc'] = acc_str
        metrics_path.write_text(json.dumps(metrics, indent=4) + '\n')
        print(f'Metrics saved to {metrics_path}')
