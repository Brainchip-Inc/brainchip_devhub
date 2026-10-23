#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Portrait128 evaluation for tf_keras or akida models.

Reports the binary IoU (mean of the person and background IoUs) and the pixel
accuracy on the validation split, both thresholded at a logit of 0.

Example
-------
    python portrait128_eval.py -d ./data/portrait128 -l models/akida_unet_portrait128.h5
"""
import argparse
import json
import pathlib
import numpy as np
import tensorflow as tf

from tqdm import tqdm

import akida

from cnn2snn import load_quantized_model

from portrait128_data import get_data
from portrait128_train import get_metrics
from brainchip_utils.hardware_utils import get_akida_device

tf.config.experimental.enable_op_determinism()


# ---------------------------------------------------------------------------
# Evaluation on Akida
# ---------------------------------------------------------------------------
def evaluate_akida_model(akida_model, val_dataset):
    """Run inference with an Akida model and return (binary IoU, pixel accuracy)."""
    device = get_akida_device(target_version=akida_model.ip_version)
    if device is not None:
        akida_model.map(device, mode=akida.MapMode.Minimal)
        print('Running inference on Akida hardware device')
        akida_model.summary()

    iou_metric, acc_metric = get_metrics()

    # Akida can't directly digest the tensorflow dataset, we need to
    # manually iterate over the dataset to deliver inputs as numpy arrays.
    # The Akida output already has the mask's (B, H, W, 1) shape, so it can be
    # compared with the labels directly: no squeeze is needed.
    for batch, label_batch in tqdm(val_dataset, desc="Evaluating on Akida"):
        logits_batch = akida_model.predict(batch.numpy().astype(np.uint8))
        iou_metric.update_state(label_batch, logits_batch)
        acc_metric.update_state(label_batch, logits_batch)

    binary_iou = float(iou_metric.result())
    accuracy = float(acc_metric.result())
    print(f'Akida binary IoU: {binary_iou:.4f}, pixel accuracy: {accuracy:.4f}')
    return binary_iou, accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras or .fbz akida model)')
    parser.add_argument('-d', '--data', default='./data/portrait128',
                        help='Portrait128 folder (contains img_uint8.npy and msk_uint8.npy)')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Write IoU and accuracy (and param count for .h5) to metrics.json')
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    if args.loadmodel.endswith('.h5'):
        model = load_quantized_model(args.loadmodel)
        model.compile(metrics=get_metrics())
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
        binary_iou, accuracy = evaluate_akida_model(model, val_ds)
    else:
        results = model.evaluate(val_ds, verbose=0, return_dict=True)
        binary_iou, accuracy = results['binary_iou'], results['accuracy']
        print(f'Validation binary IoU: {binary_iou:.4f}, pixel accuracy: {accuracy:.4f}')

    # ---------------------------------------------------------------------------
    # Persist metrics
    # ---------------------------------------------------------------------------
    if args.save_metrics:
        # Updates the stored metrics used to generate the README performance
        # tables. For code maintenance only, run against the pretrained models.
        #
        # Models are told apart by filename. The 8-bit model contributes
        # "quantized" metrics (from its .h5) and "akida" metrics (from its .fbz):
        #   akida_unet_portrait128.h5                    -> float_iou, float_acc, params
        #   akida_unet_portrait128_i8_w8_a8_qat.{h5,fbz} -> w8a8_quant_* / w8a8_akida_*
        # (The PTQ model is a throwaway on the way to QAT: it is never evaluated
        #  or stored, so it has no metrics.)
        metrics_path = pathlib.Path(__file__).parent / 'docs' / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        iou_str = f'{binary_iou:.4f}'
        acc_str = f'{accuracy * 100:.2f}%'
        stem = pathlib.Path(args.loadmodel).stem

        # Determine the variant prefix from the filename.
        if 'i8_w8_a8' in stem:
            variant = 'w8a8'
        else:
            variant = None  # float model

        if variant is None:
            # Float model: record float metrics and parameter count.
            metrics['float_iou'] = iou_str
            metrics['float_acc'] = acc_str
            metrics['params'] = f'{model.count_params():,}'
        elif isakida:
            metrics[f'{variant}_akida_iou'] = iou_str
            metrics[f'{variant}_akida_acc'] = acc_str
        else:
            metrics[f'{variant}_quant_iou'] = iou_str
            metrics[f'{variant}_quant_acc'] = acc_str
        metrics_path.write_text(json.dumps(metrics, indent=4) + '\n')
        print(f'Metrics saved to {metrics_path}')
