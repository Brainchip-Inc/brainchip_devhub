#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Detection training

Example
-------
    python detection_train.py -d /data/voc/ -e 70  \\
        -l yolo_akidanet_detection.h5 -s yolo_akidanet_detection.h5
"""
import argparse
import numpy as np
import tensorflow as tf

from tf_keras import Model
from tf_keras.layers import Reshape, ReLU
from tf_keras.optimizers.legacy import Adam
from tf_keras import regularizers
from tf_keras.utils import set_random_seed

from cnn2snn import load_quantized_model

from akida_models.detection.yolo_loss import YoloLoss

from detection_data import get_data, get_anchors

# Must be called before any TF ops to make GPU ops (conv backward passes,
# bilinear resize, etc.) deterministic. Has a small throughput cost.
tf.config.experimental.enable_op_determinism()


def train_detection(model, train_ds, val_ds, num_train, epochs, learning_rate, anchors,
                    batch_size, regularization=None, seed=42):
    set_random_seed(seed)

    grid_size = model.output_shape[1:3]
    num_classes = model.output_shape[-1] // len(anchors) - 5
    steps_per_epoch = int(np.ceil(num_train / batch_size))

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    if regularization is not None:
        print('Adding Activity Regularization to ReLU layers')
        regularizer = regularizers.L1L2(regularization, regularization)
        for layer in model.layers:
            if isinstance(layer, ReLU):
                layer.activity_regularizer = regularizer

    # YoloLoss expects targets shaped (grid_h, grid_w, num_anchors, 5+classes),
    # while the model's raw output is the flat conv output (grid_h, grid_w,
    # num_anchors*(5+classes)). Wrap with a reshape for training only; `model`
    # (returned to the caller) shares the same underlying layers, so its
    # weights are updated in place and its own (unreshaped) output is
    # unaffected.
    output = Reshape((grid_size[0], grid_size[1], len(anchors), 5 + num_classes),
                     name='YOLO_output')(model.output)
    train_model = Model(model.input, output)
    train_model.compile(optimizer=Adam(learning_rate=learning_rate),
                        loss=YoloLoss(anchors, grid_size, batch_size))

    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------
    train_model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_ds,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras or .fbz akida model)')
    parser.add_argument('-s', '--savemodel', required=True,
                        help='Model save path')

    parser.add_argument('-d', '--data', default='./data/voc',
                        help='VOC dataset root (directory containing the VOC tar archives)')

    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=70)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-4,
                        help='Initial learning rate')
    parser.add_argument('-reg', '--regularization', type=float, default=None,
                        help='Activity Regularization to increase sparsity')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    model = load_quantized_model(args.loadmodel)

    # ---------------------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------------------
    anchors = get_anchors()
    train_ds, val_ds, num_train = get_data(args.data, model.input_shape[1:], args.batch_size,
                                           seed=args.seed)

    train_detection(model=model,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    num_train=num_train,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    anchors=anchors,
                    batch_size=args.batch_size,
                    regularization=args.regularization,
                    seed=args.seed)

    model.save(args.savemodel, include_optimizer=False)
    print(f'Model saved as {args.savemodel}.')
