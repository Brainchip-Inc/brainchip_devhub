#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Eyetracking training.

This is a regression task (continuous (x, y) gaze angle in degrees), not
classification, so the loss/metric choices and y-normalization differ from
the other examples in this zoo:

- MSE loss on z-normalized targets (mean/std computed from the training
  split), not SparseCategoricalCrossentropy. The normalization stats are
  saved alongside the model (as `<savepath>.yscaler.npz`) since
  eyetracking_eval.py / eyetracking_benchmark.py need them to report error
  in real degrees, not normalized units.
- The model predicts gaze at EVERY timestep (causal/streaming -- see
  eyetracking_model.py), so labels are per-timestep too: training targets
  the whole (window, 1, 1, 2) trajectory, not a single end-of-window value.
  Early timesteps have little causal context ("warm-up") and are expected
  to be less accurate; this is visible in the per-position error curve
  plotted by eyetracking_eval.py.

Example
-------
    python eyetracking_train.py -d ./data/processed -e 30 \\
        -l models/tinytemporalcnn_eyetracking_untrained.h5 -s models/tinytemporalcnn_eyetracking.h5
"""
import argparse

import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras.utils import set_random_seed

from cnn2snn import load_quantized_model

from eyetracking_data import get_data

# Must be called before any TF ops to make GPU ops deterministic. Has a
# small throughput cost.
tf.config.experimental.enable_op_determinism()


def train_eyetracking(model, train_data, val_data, epochs, learning_rate, seed=42):
    """Trains `model` in place.

    Args:
        model: a compiled-or-not tf_keras Model with the (window, 1, 1, 2)
            output shape of build_eyetracking_model().
        train_data, val_data: (X, y) tuples as returned by
            eyetracking_data.get_data(), y in real (not normalized) degrees.
        epochs (int): number of training epochs.
        learning_rate (float): initial Adam learning rate.
        seed (int): random seed for reproducibility.

    Returns:
        (history, y_mean, y_std): the Keras History object and the target
        normalization stats (needed by callers to de-normalize predictions).
    """
    set_random_seed(seed)

    X_train, y_train = train_data
    X_val, y_val = val_data

    # Normalize using ALL timesteps' statistics (not just window-end).
    y_mean = y_train.reshape(-1, 2).mean(axis=0)
    y_std = y_train.reshape(-1, 2).std(axis=0)

    def to_norm(y):
        return (y - y_mean) / y_std

    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='mse', metrics=['mae'])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    ]

    history = model.fit(
        X_train, to_norm(y_train),
        validation_data=(X_val, to_norm(y_val)),
        epochs=epochs,
        batch_size=512,
        callbacks=callbacks,
        verbose=2,
    )
    return history, y_mean, y_std


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras or quantized model)')
    parser.add_argument('-s', '--savemodel', required=True,
                        help='Model save path')

    parser.add_argument('-d', '--data', default='./data/processed',
                        help='Path to the processed dataset directory (train/val/test _seq.npz)')

    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    try:
        model = load_quantized_model(args.loadmodel)
    except Exception:
        model = keras.models.load_model(args.loadmodel)

    # ---------------------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------------------
    train_data, val_data = get_data(args.data)

    history, y_mean, y_std = train_eyetracking(
        model=model,
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    model.save(args.savemodel, include_optimizer=False)
    np.savez(args.savemodel + '.yscaler.npz', mean=y_mean, std=y_std)
    print(f'Model saved as {args.savemodel}.')
    print(f'Target scaler saved as {args.savemodel}.yscaler.npz.')
