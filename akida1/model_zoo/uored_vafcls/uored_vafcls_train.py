#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
UORED-VAFCLS bearing fault training.

The loss is binary cross-entropy over four independent fault labels, not a
softmax over classes: a machine can in principle show more than one fault, and
a healthy bearing is the all-zero label vector rather than a fifth class. The
model emits raw logits, so `from_logits=True` throughout.

There is no validation set, and consequently no early stopping, no
learning-rate-on-plateau and no best-weight restore. This is deliberate. The
only data not used for training is the held-out fold, and selecting weights on
that would be exactly the leakage the bearing-disjoint split exists to prevent.
The recipe is instead fixed - a set epoch count, evaluated at the final epoch -
and folds 0-4 are reserved for tuning that pipeline. Do not "fix"
this by adding a callback that monitors the held-out fold.

Example
-------
    python uored_vafcls_train.py -l models/akdcnn_uored_vafcls_untrained.h5 \\
        -s models/akdcnn_uored_vafcls.h5 -e 30 -lr 2e-4 --fold 5
"""

import argparse

import tensorflow as tf
from cnn2snn import load_quantized_model
from tf_keras import regularizers
from tf_keras.layers import ReLU
from tf_keras.losses import BinaryCrossentropy
from tf_keras.metrics import AUC
from tf_keras.optimizers.legacy import Adam
from tf_keras.optimizers.schedules import CosineDecay
from tf_keras.utils import set_random_seed

from uored_vafcls_data import get_data

# Must be called before any TF ops to make GPU ops deterministic.
tf.config.experimental.enable_op_determinism()


def train_uored_vafcls(model, train_ds, epochs, learning_rate,
                       regularization=None, seed=0):
    """Train or quantization-aware tune the model on one fold.

    Args:
        model (tf_keras.Model): the model to train, float or quantized.
        train_ds (tf.data.Dataset): training dataset.
        epochs (int): number of epochs.
        learning_rate (float): peak learning rate for the cosine schedule.
        regularization (float, optional): activity regularization on the ReLU
            layers, to increase activation sparsity.
        seed (int): random seed.

    Returns:
        tf_keras.callbacks.History: the fit history.
    """
    set_random_seed(seed)

    # Optionaly add activity regularization for additional sparsity
    if regularization is not None:
        print('Adding Activity Regularization to ReLU layers')
        regularizer = regularizers.L1L2(regularization, regularization)
        for layer in model.layers:
            if isinstance(layer, ReLU):
                layer.activity_regularizer = regularizer

    # Learning Rate Scheduler
    steps_per_epoch = len(train_ds)
    total_steps = steps_per_epoch*epochs
    warmup_fraction = 0.05
    warmup_steps = max(1, int(warmup_fraction * total_steps))
    lr_schedule = CosineDecay(initial_learning_rate=learning_rate * 0.01,
                           warmup_target=learning_rate,
                           warmup_steps=warmup_steps,
                           decay_steps=max(1, total_steps - warmup_steps))

    num_labels = model.output_shape[-1]
    model.compile(optimizer=Adam(learning_rate=lr_schedule),
                  loss=BinaryCrossentropy(from_logits=True),
                  metrics=[AUC(name='auroc', multi_label=True,
                               num_labels=num_labels)])

    # Training
    return model.fit(train_ds, epochs=epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras or quantized model)')
    parser.add_argument('-s', '--savemodel', required=True,
                        help='Model save path')
    parser.add_argument('-d', '--data', default='./data/uored_vafcls',
                        help='Directory holding the prepared .npz cache')
    parser.add_argument('-b', '--batch_size', type=int, default=120,
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-lr', '--learning_rate', type=float, default=2e-4,
                        help='Peak learning rate for the cosine schedule')
    parser.add_argument('-reg', '--regularization', type=float, default=None,
                        help='Activity Regularization to increase sparsity')
    parser.add_argument('--fold', type=int, default=5,
                        help='Bearing-disjoint fold index.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Model
    model = load_quantized_model(args.loadmodel)

    # Dataset
    train_ds, test_ds = get_data(args.data, model.input_shape[1:],
                                 args.batch_size, fold=args.fold,
                                 seed=args.seed)

    # Run Training
    train_uored_vafcls(model=model, train_ds=train_ds,
                       epochs=args.epochs, learning_rate=args.learning_rate,
                       regularization=args.regularization, seed=args.seed)

    # Save Trained Model
    model.save(args.savemodel, include_optimizer=False)
    print(f'Model saved as {args.savemodel}.')
