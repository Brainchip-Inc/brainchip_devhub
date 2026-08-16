#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Create the arrhythmia classification model for Akida 1.

A small depthwise-separable convolutional network over 36x32 ECG scalograms.
Every layer maps to Akida 1: a dense stem convolution, three separable
convolution blocks with stride-2 max pooling between them, then global average
pooling and two dense layers. 5x5 kernels are used throughout - wider than the
3x3 typical of image models - because the informative structure in a wavelet
scalogram is spread over a broader time-frequency neighbourhood.

Input scaling is part of the model: a Rescaling layer maps the uint8 inputs the
data pipeline produces into the [0, 1] range the network trains on. Keeping it
in the graph means float training, quantized training and Akida inference all
see identical values, and cnn2snn folds the layer into the first convolution at
conversion time.

Usage:
    python arrhythmia_model.py [-s OUTPUT_PATH]
"""

import argparse

from tf_keras import Model, regularizers
from tf_keras.layers import (BatchNormalization, Conv2D, Dense,
                             GlobalAveragePooling2D, Input, MaxPooling2D, ReLU,
                             Rescaling, SeparableConv2D)
from tf_keras.utils import set_random_seed

from cnn2snn import AkidaVersion, set_akida_version

from arrhythmia_data import INPUT_SHAPE, NUM_CLASSES

# L2 penalty on the separable blocks' pointwise kernels. Small: it is there to
# discourage a handful of dominant channels, not to constrain capacity.
POINTWISE_L2 = 2e-7


def _ds_block(x, filters, regularizer, name):
    """Depthwise-separable convolution, batch norm, capped ReLU."""
    x = SeparableConv2D(filters=filters, kernel_size=(5, 5),
                        pointwise_regularizer=regularizer, padding='same',
                        use_bias=False, name=f'{name}_sepconv')(x)
    x = BatchNormalization(name=f'{name}_bn')(x)
    x = ReLU(max_value=6, name=f'{name}_relu6')(x)
    return x


def _final_ds_block(x, filters, regularizer, name):
    """As _ds_block, without the activation - global pooling follows."""
    x = SeparableConv2D(filters=filters, kernel_size=(5, 5),
                        pointwise_regularizer=regularizer, padding='same',
                        use_bias=False, name=f'{name}_sepconv')(x)
    x = BatchNormalization(name=f'{name}_bn')(x)
    return x


def build_arrhythmia_model(seed=42, pointwise_l2=POINTWISE_L2):
    """Build the untrained arrhythmia classification model.

    Args:
        seed (int): random seed for weight initialisation.
        pointwise_l2 (float): L2 penalty on the separable pointwise kernels.

    Returns:
        tf_keras.Model: the model, expecting uint8 inputs of shape INPUT_SHAPE
        and returning NUM_CLASSES logits.
    """
    set_random_seed(seed)
    regularizer = regularizers.L2(pointwise_l2)

    with set_akida_version(AkidaVersion.v1):
        inputs = Input(shape=INPUT_SHAPE, name='input')

        # uint8 [0, 255] -> [0, 1]. Folded into stem_conv by cnn2snn.
        x = Rescaling(1. / 255, 0., name='rescaling')(inputs)

        # Stem: a dense convolution, so the first layer sees the raw scalogram.
        x = Conv2D(16, (5, 5), padding='same', use_bias=False,
                   name='stem_conv')(x)
        x = BatchNormalization(name='stem_bn')(x)
        x = ReLU(max_value=6, name='stem_relu6')(x)

        # Feature blocks, halving the spatial resolution between each.
        x = _ds_block(x, 32, regularizer, 'block1')
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                         name='block1_pool')(x)

        x = _ds_block(x, 64, regularizer, 'block2')
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same',
                         name='block2_pool')(x)

        x = _final_ds_block(x, 128, regularizer, 'block3')

        # Classification head.
        x = GlobalAveragePooling2D(name='global_avg_pool')(x)
        x = ReLU(max_value=6, name='head_relu6')(x)
        x = Dense(64, activation='linear', name='dense')(x)
        x = ReLU(max_value=6, name='dense_relu6')(x)
        outputs = Dense(NUM_CLASSES, activation='linear',
                        name='predictions')(x)

        model = Model(inputs, outputs, name='arrhythmia_classification')

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build the arrhythmia classification model for Akida 1')
    parser.add_argument('-s', '--savepath',
                        default='./models/arrhythmia_classification_untrained.h5',
                        help='Save model with the specified path + name')
    parser.add_argument('--seed', type=int, default=7,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    model = build_arrhythmia_model(seed=args.seed)
    model.summary()
    model.save(args.savepath, include_optimizer=False)
    print(f'Model saved to {args.savepath}')
