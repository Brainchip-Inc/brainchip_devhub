#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Data loading for the eyetracking example.

Unlike the image-classification examples in this zoo, there is no
"download a tarball of JPEGs" step: the dataset is built by simulating an
8-channel photodiode array response from real GazeBase gaze trajectories
(see docs/dataset_report.md and README.md "Dataset setup" for the full
pipeline and why). `get_data`/`get_samples` below read the already-built
per-timestep windows from data/processed/{split}_seq.npz, produced by
data/build_dataset_seq.py.

Labels are continuous (x, y) gaze angle in degrees, not class indices --
this is a regression task, so there is no `class_mode`/one-hot handling
here, and `get_data` returns raw (X, y) arrays rather than a Keras
directory iterator.

Input is SIGNED int8, not the usual uint8: Akida's BufferTempConv path (the
hardware primitive this model's Conv3D layers quantize into -- see
eyetracking_model.py) only accepts signed input, so the [0, 255] ADC-style
reading is recentered to int8's [-128, 127] range here.
"""

import os

import numpy as np

WINDOW = 64
CHANNELS = 8


def to_signed_int8(X):
    """(N, window, channels) float32 in [0, 1] -> (N, window, 1, 1, channels) int8."""
    X_u8 = np.clip(np.round(X * 255.0), 0, 255).astype(np.int16)
    X_i8 = (X_u8 - 128).astype(np.int8)
    return X_i8[:, :, None, None, :]


def _load_split(data_path, split):
    d = np.load(os.path.join(data_path, f'{split}_seq.npz'))
    X = to_signed_int8(d['X'])  # (N, window, 1, 1, channels) int8
    y = d['y'][:, :, None, None, :].astype(np.float32)  # (N, window, 1, 1, 2), matches model output
    return X, y


def get_data(data_path, input_shape=(WINDOW, 1, 1, CHANNELS), batch_size=None):
    """Loads the eyetracking train/val splits.

    Args:
        data_path (str): path to the processed dataset directory (contains
            train_seq.npz, val_seq.npz, test_seq.npz).
        input_shape (tuple): unused, kept for signature parity with the
            other examples in this zoo; the window/channel shape is fixed
            by how data/build_dataset_seq.py built the .npz files.
        batch_size (int, optional): unused -- this dataset is small enough
            to train directly on in-memory arrays via model.fit(batch_size=...).

    Returns:
        (X_train, y_train), (X_val, y_val): int8 inputs, float32 (x, y) gaze
        targets, both shaped (N, window, 1, 1, C) / (N, window, 1, 1, 2).
    """
    del input_shape, batch_size  # unused, see docstring
    train = _load_split(data_path, 'train')
    val = _load_split(data_path, 'val')
    return train, val


def get_samples(data_path, input_shape=(WINDOW, 1, 1, CHANNELS), num_samples=1024, seed=0):
    """Loads calibration samples for quantization.

    quantizeml calibrates on real data. The deployed model consumes ONE
    timestep at a time after bufferization (BufferTempConv has no window
    axis), so calibration samples must be individual frames, not whole
    (window, 1, 1, C) windows -- the window axis is flattened away here.

    Args:
        data_path (str): path to the processed dataset directory.
        input_shape (tuple): unused, see get_data().
        num_samples (int): number of calibration frames to return.
        seed (int): RNG seed for sample selection.

    Returns:
        np.ndarray: (num_samples, 1, 1, channels), dtype int8 (signed --
        see module docstring; this is NOT the uint8 convention used
        elsewhere in this zoo).
    """
    del input_shape  # unused, see docstring
    X_train, _ = _load_split(data_path, 'train')
    frames = X_train.reshape(-1, 1, 1, X_train.shape[-1])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(frames), size=num_samples, replace=False)
    return frames[idx]
