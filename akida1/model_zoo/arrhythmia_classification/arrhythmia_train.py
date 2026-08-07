#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Arrhythmia classification training.

Used for both stages of the pipeline: full-precision training from scratch, and
quantization-aware fine-tuning of the quantized model. The only difference is
which model is loaded and for how many epochs.

Class weights are on by default and matter a great deal here. The training set
is 91% normal beats, 2% supraventricular and 7% ventricular; without
reweighting, the model reaches high accuracy by essentially ignoring the
supraventricular class.

Example
-------
    python arrhythmia_train.py -d ./data/mitdb -e 80 -lr 3e-3 \\
        -l models/arrhythmia_classification_untrained.h5 \\
        -s models/arrhythmia_classification.h5
"""
import argparse

import tensorflow as tf

from tf_keras import regularizers
from tf_keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tf_keras.layers import ReLU
from tf_keras.losses import SparseCategoricalCrossentropy
from tf_keras.optimizers.legacy import Adam
from tf_keras.utils import set_random_seed

from akida_models.training import RestoreBest
from cnn2snn import load_quantized_model

from arrhythmia_data import get_data

# Must be called before any TF ops to make GPU ops (conv backward passes,
# bilinear resize, etc.) deterministic. Has a small throughput cost.
tf.config.experimental.enable_op_determinism()

# Inverse-frequency-flavoured weights, tuned by hand: enough to make the
# minority classes count without destabilising training.
CLASS_WEIGHTS = {0: 1.0, 1: 6.0, 2: 3.0}


def train_arrhythmia(model, train_ds, val_ds, epochs, learning_rate,
                     regularization=None, class_weight=None, seed=42):
    """Train or fine-tune the arrhythmia model.

    The learning rate is reduced on plateau rather than following a fixed decay
    schedule: beat-level validation loss is noisy, and the number of epochs
    actually needed varies between the float and quantized stages. Training
    stops early once validation loss stops improving, and the best weights are
    restored at the end.

    Args:
        model (tf_keras.Model): model to train, float or quantized.
        train_ds (tf.data.Dataset): training dataset.
        val_ds (tf.data.Dataset): validation dataset used for scheduling,
            early stopping and best-weight selection.
        epochs (int): maximum number of epochs; early stopping usually cuts
            this short.
        learning_rate (float): initial learning rate.
        regularization (float, optional): L1L2 activity regularization applied
            to ReLU layers to encourage activation sparsity. Defaults to None.
        class_weight (dict, optional): per-class loss weights. Defaults to None.
        seed (int): random seed for reproducibility.
    """
    set_random_seed(seed)

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    if regularization is not None:
        print('Adding Activity Regularization to ReLU layers')
        regularizer = regularizers.L1L2(regularization, regularization)
        for layer in model.layers:
            if isinstance(layer, ReLU):
                layer.activity_regularizer = regularizer

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, verbose=1),
        # EarlyStopping(monitor='val_loss', patience=8, verbose=1),
        # RestoreBest(model, monitor='val_loss', mode='min'),
    ]

    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------
    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        class_weight=class_weight,
        callbacks=callbacks,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras or quantized model)')
    parser.add_argument('-s', '--savemodel', required=True,
                        help='Model save path')
    parser.add_argument('-d', '--data', default='./data/mitdb',
                        help='MIT-BIH record directory')
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-e', '--epochs', type=int, default=80)
    parser.add_argument('-lr', '--learning_rate', type=float, default=3e-3,
                        help='Initial learning rate')
    parser.add_argument('-reg', '--regularization', type=float, default=None,
                        help='Activity Regularization to increase sparsity')
    parser.add_argument('--no-class-weights', action='store_true',
                        help='Train without per-class loss weighting')
    parser.add_argument('--seed', type=int, default=67004546,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    model = load_quantized_model(args.loadmodel)

    # ---------------------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------------------
    train_ds, val_ds = get_data(args.data, model.input_shape[1:],
                                args.batch_size, seed=args.seed)

    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------
    class_weight = None if args.no_class_weights else CLASS_WEIGHTS
    print(f'Class weights: {class_weight}')

    train_arrhythmia(model, train_ds, val_ds, args.epochs, args.learning_rate,
                     regularization=args.regularization,
                     class_weight=class_weight, seed=args.seed)

    model.save(args.savemodel, include_optimizer=False)
    print(f'Model saved as {args.savemodel}.')
