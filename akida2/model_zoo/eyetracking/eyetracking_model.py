#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Create the Tiny Temporal CNN model for the eyetracking example, targeting
the Akida 2 platform.

This is a causal, streaming 1D temporal CNN: instead of predicting one gaze
value from a whole symmetric window (the common "GlobalAveragePooling1D"
design), it predicts gaze continuously at EVERY timestep from past-only
context. That shape is required by Akida's BufferTempConv primitive, the
only supported hardware path for genuine 1D temporal convolution on Akida 2
(the same primitive used by BrainChip's own eye-tracking reference model,
akida_models/tenn_spatiotemporal/eye_train.py). A symmetric Conv1D-as-Conv2D
reshape was tried first and trains/quantizes fine, but cnn2snn.convert()
rejects it: Akida's 2D conv hardware only supports SQUARE kernels, and a
(5, 1)-style "height-only" kernel is not one. See docs/benchmark_report.md
for the full account of both attempts.

quantizeml's replace_conv3d transform turns the Conv3D layers below into
BufferTempConv automatically at quantization time (see
quantizeml.models.transforms.replace_conv3d); no manual transform call is
needed in this file.

Input is (window=64, 1, 1, channels=8): a 5D (batch, T, H, W, C) shape with
H=W=1, since there is no spatial structure, just 8 photodiode channels over
time. Akida's BufferTempConv path only accepts SIGNED int8 input, so the
8-channel [0, 255] ADC-style reading is recentered to int8's [-128, 127]
range by the data pipeline (eyetracking_data.py); the Rescaling layer's
offset undoes that shift for the float math.

Usage:
    python eyetracking_model.py [-s OUTPUT_PATH]
"""

import argparse

import tf_keras as keras
from tf_keras import layers
from tf_keras.utils import set_random_seed

WINDOW = 64
CHANNELS = 8


def build_eyetracking_model(window=WINDOW, channels=CHANNELS, seed=42):
    set_random_seed(seed)

    inputs = keras.Input(shape=(window, 1, 1, channels), name='photodiode_signal', dtype='int8')
    x = layers.Rescaling(1.0 / 255.0, offset=128.0 / 255.0, name='rescaling')(inputs)

    x = layers.ZeroPadding3D(padding=((4, 0), (0, 0), (0, 0)))(x)
    x = layers.Conv3D(16, kernel_size=(5, 1, 1), padding='valid', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.ZeroPadding3D(padding=((2, 0), (0, 0), (0, 0)))(x)
    x = layers.Conv3D(32, kernel_size=(3, 1, 1), padding='valid', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Dense(32)(x)
    x = layers.ReLU()(x)
    # outputs shape: (batch, window, 1, 1, 2). A trailing Reshape to squeeze the
    # trivial (1, 1) spatial dims trips a bug in quantizeml's replace_conv3d
    # (_verify_reshape assumes every Reshape layer's config has
    # 'batch_input_shape', true only for a model's first layer) -- so the (1, 1)
    # dims are left in place here and squeezed in numpy post-processing instead
    # (see eyetracking_train.py / eyetracking_eval.py).
    outputs = layers.Dense(2, name='gaze_xy')(x)

    return keras.Model(inputs, outputs, name='tinytemporalcnn_eyetracking')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build the Tiny Temporal CNN eyetracking model for Akida 2')
    parser.add_argument('-s', '--savepath', type=str,
                        default='./models/tinytemporalcnn_eyetracking_untrained.h5',
                        help='Save model with the specified path + name')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    model = build_eyetracking_model(seed=args.seed)
    model.summary()
    model.save(args.savepath, include_optimizer=False)
    print(f'Model saved to {args.savepath}')
