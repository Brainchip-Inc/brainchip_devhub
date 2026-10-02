#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
PlantVillage training

Example
-------
    python plant_village_train.py -d ./data/plant_village -e 10 \\
        -l models/akidanet_plant_village.h5 -s models/akidanet_plant_village.h5
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

from plant_village_data import get_data

# Must be called before any TF ops to make GPU ops (conv backward passes,
# bilinear resize, etc.) deterministic. Has a small throughput cost.
tf.config.experimental.enable_op_determinism()


def get_custom_scheduler(initial_lr, n_epochs):
    """Exponential LR scheduler: decays continuously from initial_lr to
    initial_lr * 0.01 over n_epochs."""
    lr_decay = (initial_lr * 0.01 / initial_lr) ** (1.0 / max(n_epochs, 1))

    def lr_schedule(epoch, lr):
        return initial_lr * lr_decay ** epoch

    return LearningRateScheduler(lr_schedule)


def train_plant_village(model, train_ds, val_ds, epochs, learning_rate,
                        regularization=None, seed=42):
    set_random_seed(seed)

    if regularization is not None:
        print('Adding Activity Regularization to ReLU layers')
        regularizer = regularizers.L1L2(regularization, regularization)
        for layer in model.layers:
            if isinstance(layer, ReLU):
                layer.activity_regularizer = regularizer

    # The model ends in a softmax activation, so the loss is computed on
    # probabilities (from_logits=False).
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    callbacks = [get_custom_scheduler(initial_lr=learning_rate, n_epochs=epochs)]

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
    parser.add_argument('-d', '--data', default='./data/plant_village',
                        help='PlantVillage tfds data directory')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2,
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
    train_ds, val_ds, test_ds = get_data(args.data, model.input_shape[1:], args.batch_size)

    train_plant_village(model=model,
                        train_ds=train_ds,
                        val_ds=val_ds,
                        epochs=args.epochs,
                        learning_rate=args.learning_rate,
                        regularization=args.regularization,
                        seed=args.seed)

    model.save(args.savemodel, include_optimizer=False)
    print(f'Model saved as {args.savemodel}.')
