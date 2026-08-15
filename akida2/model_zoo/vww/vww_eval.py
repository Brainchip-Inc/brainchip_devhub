#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
VWW evaluation for tf_keras or akida models.
Example
-------
    python eval.py -d /data/vww_coco2014_96/ -l akidanet_vww.h5
"""
import argparse
import json
import pathlib
import numpy as np
import tensorflow as tf

from tqdm import tqdm

import akida

from cnn2snn import load_quantized_model

from vww_data import get_data
from brainchip_utils.hardware_utils import get_akida_device

tf.config.experimental.enable_op_determinism()

# ---------------------------------------------------------------------------
# Evaluation on Akida
# ---------------------------------------------------------------------------
def evaluate_akida_model(akida_model, val_dataset):
    """Run inference with an Akida model and return (predictions, labels)."""
    device = get_akida_device(target_version = akida_model.ip_version)
    if device is not None:
        akida_model.map(device, mode=akida.MapMode.Minimal)
        print('Running inference on Akida hardware device')
        akida_model.summary()

    labels_all = None
    logits_all = None

    # Akida can't directly digest the tensorflow dataset, we need to
    # manually iterate over the dataset to deliver inputs as numpy arrays.
    # val_dataset is a Keras DirectoryIterator, which cycles indefinitely,
    # so we must limit iteration to a single epoch (len(val_dataset) batches).
    num_batches = len(val_dataset)
    for _ in tqdm(range(num_batches), desc="Evaluating on Akida"):
        batch, label_batch = next(val_dataset)
        if not isinstance(batch, np.ndarray):
            batch = batch.numpy()

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
    accuracy = np.mean(np.equal(np.array(preds), np.array(labels_all)))
    print(f'Akida accuracy: {accuracy:.4f}')
    return preds, labels_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras or .fbz akida model)')
    parser.add_argument('-d', '--data', default='./data/vw_coco2014_96',
                        help='VWW dataset root (contains train/ and val/ subdirs)')
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

    # ---------------------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------------------
    train_ds, val_ds = get_data(args.data, imsize, batch_size=32)

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
        # The Akida 2 VWW example has three quantized variants, disambiguated by
        # filename. Each variant contributes a "quantized" accuracy (from its
        # .h5) and an "akida" accuracy (from its .fbz):
        #   akidanet_vww.h5                 -> float_acc, params
        #   akidanet_vww_i8_w8_a8.{h5,fbz}  -> w8a8_quant_acc / w8a8_akida_acc
        #   akidanet_vww_i8_w4_a8.{h5,fbz}  -> w4a8_ptq_quant_acc / w4a8_ptq_akida_acc
        #   akidanet_vww_i8_w4_a8_qat.{h5,fbz} -> w4a8_qat_quant_acc / w4a8_qat_akida_acc
        metrics_path = pathlib.Path(__file__).parent / 'docs' / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        acc_str = f'{accuracy * 100:.2f}%'
        stem = pathlib.Path(args.loadmodel).stem

        # Determine the variant prefix from the filename. Order matters: the QAT
        # 4-bit filename also contains 'i8_w4_a8', so test for '_qat' first.
        if 'i8_w8_a8' in stem:
            variant = 'w8a8'
        elif 'i8_w4_a8_qat' in stem or ('i8_w4_a8' in stem and 'qat' in stem):
            variant = 'w4a8_qat'
        elif 'i8_w4_a8' in stem:
            variant = 'w4a8_ptq'
        else:
            variant = None  # float model

        if variant is None:
            # Float model: record float accuracy and parameter count.
            metrics['float_acc'] = acc_str
            metrics['params'] = f'{model.count_params():,}'
        elif isakida:
            metrics[f'{variant}_akida_acc'] = acc_str
        else:
            metrics[f'{variant}_quant_acc'] = acc_str
        metrics_path.write_text(json.dumps(metrics, indent=4) + '\n')
        print(f'Metrics saved to {metrics_path}')
