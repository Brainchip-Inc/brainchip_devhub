#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Detection evaluation (mAP) for tf_keras or akida YOLOv2 models.

Example
-------
    python detection_eval.py -d /data/voc/ -l yolo_akidanet_detection.h5
"""
import argparse
import json
import pathlib

import akida
import tensorflow as tf

from tf_keras import Model
from tf_keras.layers import Reshape
# Must be called before any TF ops to make GPU ops (conv backward passes,
# bilinear resize, etc.) deterministic. Has a small throughput cost.
tf.config.experimental.enable_op_determinism()

from cnn2snn import load_quantized_model

from akida_models.detection.map_evaluation import MapEvaluation

from detection_data import LABELS, get_anchors, get_voc_dataset
from brainchip_utils.hardware_utils import get_akida_device

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras or .fbz akida model)')
    parser.add_argument('-d', '--data', default='./data/voc',
                        help='VOC dataset root (directory containing the VOC tar archives)')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Write mAP@0.5 (and param count for .h5) to metrics.json')
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    anchors = get_anchors()

    if args.loadmodel.endswith('.h5'):
        model = load_quantized_model(args.loadmodel)
        is_keras_model = True

        # MapEvaluation expects a Keras model whose output is already shaped
        # (grid_h, grid_w, num_anchors, 5+classes); it only reshapes the flat
        # conv output internally for Akida models. Wrap with the same
        # YOLO_output reshape used during training, for evaluation only.
        grid_size = model.output_shape[1:3]
        num_classes = model.output_shape[-1] // len(anchors) - 5
        eval_output = Reshape((grid_size[0], grid_size[1], len(anchors), 5 + num_classes),
                              name='YOLO_output')(model.output)
        eval_model = Model(model.input, eval_output)
    elif args.loadmodel.endswith('.fbz'):
        model = akida.Model(args.loadmodel)
        is_keras_model = False
        eval_model = model

        device = get_akida_device(target_version=model.ip_version)
        if device is not None:
            model.map(device, mode=akida.MapMode.HwPr)
            print('Running inference on Akida hardware device')
            model.summary()

    # ---------------------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------------------
    val_data, labels, num_valid = get_voc_dataset(args.data, labels=LABELS, training=False)

    # ---------------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------------
    map_evaluator = MapEvaluation(eval_model, val_data, num_valid, labels, anchors,
                                  is_keras_model=is_keras_model)
    # map_dict holds mAP per IoU threshold (0.5 to 0.95). Following the
    # PASCAL VOC convention, only mAP at IoU 0.5 is reported. The per-class
    # APs also returned are averaged over all thresholds, so are not shown.
    map_dict, _ = map_evaluator.evaluate_map()
    map_50 = map_dict[0.5]
    print(f'mAP (IoU 0.5): {map_50:.4f}')

    # ---------------------------------------------------------------------------
    # Persist metrics
    # ---------------------------------------------------------------------------
    if args.save_metrics:
        # This is used to update the stored metrics that are used to generate the
        # performance tables in the README of this folder.
        # This should only be used for code maintenance, when the model or training
        # pipeline is updated and a new trained model integrated.
        metrics_path = pathlib.Path(__file__).parent / 'docs' / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        map_str = f'{map_50 * 100:.2f}%'
        if not is_keras_model:
            metrics['akida_map'] = map_str
        elif 'qat' in pathlib.Path(args.loadmodel).stem:
            metrics['qat_map'] = map_str
        else:
            metrics['float_map'] = map_str
            metrics['params'] = f'{model.count_params():,}'
        metrics_path.write_text(json.dumps(metrics, indent=4) + '\n')
        print(f'Metrics saved to {metrics_path}')
