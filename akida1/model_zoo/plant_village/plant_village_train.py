#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
PlantVillage training

Example
-------
    python plant_village_train.py -d ./data/plant_village -e 50  \\
        -l akidanet_plant_village_untrained.h5 -s akidanet_plant_village.h5
"""
import argparse

import tensorflow as tf
from tf_keras.losses import SparseCategoricalCrossentropy
from tf_keras.optimizers.legacy import Adam
from tf_keras.callbacks import LearningRateScheduler
from tf_keras import regularizers
from tf_keras.layers import ReLU
from tf_keras.utils import set_random_seed

from cnn2snn import load_quantized_model
from akida_models.training import RestoreBest

from plant_village_data import get_data

# Must be called before any TF ops to make GPU ops (conv backward passes,
# bilinear resize, etc.) deterministic. Has a small throughput cost.
tf.config.experimental.enable_op_determinism()


class HoyerSquare(regularizers.Regularizer):
    """Hoyer-Square activity regularizer: factor * (sum|x|)^2 / sum(x^2).

    Scale-invariant sparsity-inducing penalty (Yang, Wen & Li, "DeepHoyer",
    ICLR 2020). Unlike L1/L2, minimizing this ratio pushes activations toward a
    sparse (few large, rest ~zero) configuration rather than uniformly shrinking
    every value -- a large-but-sparse activation and a small-but-dense one can
    have the same L1 norm, but the sparse one has a lower Hoyer-Square value.

    Raw form is unbounded: for a tensor of N elements the ratio ranges up to N
    itself (dense/uniform case), so the same `factor` exerts more pressure on
    layers with more elements. With `normalize=True`, divides by N so the
    penalty is bounded in (0, 1] regardless of tensor size -- see the VWW
    sparsity experiment (SPARSITY_EXPERIMENT.md in ../vww/) for why this
    matters in practice.
    """

    def __init__(self, factor, normalize=False):
        self.factor = float(factor)
        self.normalize = bool(normalize)

    def __call__(self, x):
        l1 = tf.reduce_sum(tf.abs(x))
        l2_sq = tf.reduce_sum(tf.square(x))
        # eps avoids 0/0 on an all-zero activation (e.g. early in training)
        hoyer_sq = tf.square(l1) / (l2_sq + 1e-12)
        if self.normalize:
            n = tf.cast(tf.size(x), tf.float32)
            hoyer_sq = hoyer_sq / n
        return self.factor * hoyer_sq

    def get_config(self):
        return {'factor': float(self.factor), 'normalize': self.normalize}


def get_custom_scheduler(initial_lr: float, n_epochs: int):
    """
    Exponential LR scheduler matching the source training schedule:
    decays continuously from initial_lr to initial_lr * 0.01 over n_epochs.
    """
    lr_decay = (initial_lr * 0.01 / initial_lr) ** (1.0 / max(n_epochs, 1))

    def lr_schedule(epoch: int, lr: float) -> float:
        return initial_lr * lr_decay ** epoch

    return LearningRateScheduler(lr_schedule)


def train_plant_village(model, train_ds, val_ds, epochs, learning_rate, regularization=None,
                         reg_type='l1l2', seed=42):
    set_random_seed(seed)
    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    if regularization is not None:
        print(f'Adding {reg_type} Activity Regularization to ReLU layers')
        if reg_type == 'hoyer_square':
            regularizer = HoyerSquare(regularization)
        elif reg_type == 'hoyer_square_norm':
            regularizer = HoyerSquare(regularization, normalize=True)
        else:
            regularizer = regularizers.L1L2(regularization, regularization)
        for layer in model.layers:
            if isinstance(layer, ReLU):
                layer.activity_regularizer = regularizer

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------
    callbacks = []

    lr_scheduler = get_custom_scheduler(initial_lr=learning_rate, n_epochs=epochs)
    callbacks.append(lr_scheduler)

    model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras or .fbz akida model)')
    parser.add_argument('-s', '--savemodel', required=True,
                        help='Model save path')

    parser.add_argument('-d', '--data', default='./data/plant_village',
                        help='PlantVillage tfds data directory')

    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2,
                        help='Initial learning rate')
    parser.add_argument('-reg', '--regularization', type=float, default=None,
                        help='Activity Regularization to increase sparsity')
    parser.add_argument('--reg-type', choices=['l1l2', 'hoyer_square', 'hoyer_square_norm'],
                        default='l1l2',
                        help='Type of activity regularizer to use with -reg')
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
    train_ds, val_ds, test_ds = get_data(args.data, model.input_shape[1:], args.batch_size)

    train_plant_village(model=model,
                        train_ds=train_ds,
                        val_ds=val_ds,
                        epochs=args.epochs,
                        learning_rate=args.learning_rate,
                        regularization=args.regularization,
                        reg_type=args.reg_type,
                        seed=args.seed)

    model.save(args.savemodel, include_optimizer=False)
    print(f'Model saved as {args.savemodel}.')
