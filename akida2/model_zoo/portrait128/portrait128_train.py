#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Portrait128 training, for float training and quantization-aware training (QAT).

The model outputs one logit per pixel, so the loss is binary cross-entropy on
logits. Training reports the binary IoU (mean of the person and background
IoUs) and pixel accuracy. For both, a pixel counts as "person" when its logit
is above 0, i.e. its sigmoid probability is above 0.5.

The learning rate follows a one-cycle cosine schedule, stepped every batch: it
rises from learning_rate / 25 to learning_rate over the first 30 % of training,
then decays to learning_rate / 250000.

Example
-------
    python portrait128_train.py -d ./data/portrait128 -e 50 -lr 3e-5 \\
        -l models/akida_unet_portrait128.h5 -s models/akida_unet_portrait128.h5
"""
import argparse
import os

import tensorflow as tf
from tf_keras.losses import BinaryCrossentropy
from tf_keras.metrics import BinaryIoU, BinaryAccuracy
from tf_keras.optimizers.legacy import Adam
from tf_keras import regularizers
from tf_keras.layers import ReLU
from tf_keras.utils import set_random_seed

from akida_models.param_scheduler import get_cosine_lr_scheduler
from cnn2snn import load_quantized_model

from portrait128_data import get_data

# Must be called before any TF ops to make GPU ops (conv backward passes,
# bilinear resize, etc.) deterministic. Has a small throughput cost.
tf.config.experimental.enable_op_determinism()

# Disable quantizeml's run-time assertions (checks on quantized tensor values
# inside every quantized op) for training, QAT and evaluation. Set before any
# quantized model runs.
os.environ["ASSERT_ENABLED"] = "0"


class LogitBinaryIoU(BinaryIoU):
    """Binary IoU over person and background, on logits.

    A pixel is "person" when its logit is strictly positive, the same rule as
    BinaryAccuracy(threshold=0). BinaryIoU on its own counts a logit of exactly 0
    as "person", and quantized and Akida models can output exact zeros, so
    without this the two metrics would disagree on the same mask.
    """

    def __init__(self, name='binary_iou', **kwargs):
        super().__init__(target_class_ids=[0, 1], threshold=0.5, name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(tf.cast(y_pred, self._dtype) > 0, self._dtype)
        return super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        return {'name': self.name, 'dtype': self.dtype}


def get_metrics():
    """Binary IoU and pixel accuracy on logits (a pixel is "person" if its logit > 0)."""
    return [LogitBinaryIoU(), BinaryAccuracy(threshold=0.0, name='accuracy')]


def train_portrait128(model, train_ds, val_ds, epochs, learning_rate,
                      regularization=None, seed=42):
    set_random_seed(seed)

    if regularization is not None:
        print('Adding Activity Regularization to ReLU layers')
        regularizer = regularizers.L1L2(regularization, regularization)
        for layer in model.layers:
            if isinstance(layer, ReLU):
                layer.activity_regularizer = regularizer

    # The model outputs logits (no sigmoid), so the loss applies the sigmoid
    # itself (from_logits=True).
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=BinaryCrossentropy(from_logits=True),
                  metrics=get_metrics())

    # One-cycle cosine schedule over every training step: warm-up for the
    # first 30 % of steps, then decay.
    total_steps = epochs * len(train_ds)
    callbacks = [get_cosine_lr_scheduler(learning_rate, total_steps, pct_start=0.3)]

    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras or quantized model)')
    parser.add_argument('-s', '--savemodel', required=True,
                        help='Model save path')
    parser.add_argument('-d', '--data', default='./data/portrait128',
                        help='Portrait128 folder (contains img_uint8.npy and msk_uint8.npy)')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-5,
                        help='Peak learning rate')
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
    train_ds, val_ds = get_data(args.data, model.input_shape[1:], args.batch_size,
                                seed=args.seed)

    train_portrait128(model=model,
                      train_ds=train_ds,
                      val_ds=val_ds,
                      epochs=args.epochs,
                      learning_rate=args.learning_rate,
                      regularization=args.regularization,
                      seed=args.seed)

    model.save(args.savemodel, include_optimizer=False)
    print(f'Model saved as {args.savemodel}.')
