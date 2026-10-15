#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Create a DS-CNN model for the Speech Commands Keyword Spotting (KWS) task,
targeting the Akida 2 platform.

This model is the DS-CNN (depthwise-separable CNN) architecture from
akida_models, created for the "Speech Commands" / "Keyword Spotting" task. It
operates on MFCC features of shape (49, 10, 1) and classifies into 12 keyword
classes (SC12: Down, Go, Left, No, Off, On, Right, Stop, Up, Yes, Silence,
Unknown).

The architecture follows akida_models' `ds_cnn_kws` factory, with one change:
every depthwise layer is followed by its own BatchNormalization and ReLU, as in
MobileNet. `ds_cnn_kws` keeps the Akida 1 block design, where a separable
convolution is a single fused layer that can't have an activation between its
depthwise and pointwise parts. On Akida 2 the depthwise convolution is a
distinct layer, so it can have a ReLU. That makes the depthwise outputs sparse,
and the pointwise layers that consume them process fewer events.

Input preprocessing (rescaling) is included as part of the model via a
Rescaling layer: the model expects uint8 inputs, and the data pipeline delivers
uint8 MFCC features accordingly.

Usage:
    python speech_commands_model.py [-s OUTPUT_PATH]
"""

import argparse

import tensorflow as tf
from tf_keras import Model
from tf_keras.layers import (Activation, BatchNormalization, DepthwiseConv2D, Input,
                             Rescaling)
from tf_keras.utils import set_random_seed
from akida_models.layer_blocks import conv_block, dense_block
from akida_models.custom_layers import act_to_layer
from akida_models.utils import get_params_by_version
from cnn2snn import set_akida_version, AkidaVersion


def _separable_block(x, index, relu_activation, last=False):
    """Depthwise (BN, ReLU) then pointwise (BN, ReLU), as two distinct layers.

    Layer names follow akida_models' `separable_conv_block` (`dw_separable_N`,
    `pw_separable_N`), with `dw_separable_N/BN` and `dw_separable_N/relu` added.
    The last block ends with global average pooling after its ReLU, as
    `ds_cnn_kws` does on Akida 2.
    """
    name = f'separable_{index}'
    x = DepthwiseConv2D((3, 3), padding='same', use_bias=False, name=f'dw_{name}')(x)
    x = BatchNormalization(name=f'dw_{name}/BN')(x)
    x = act_to_layer(relu_activation, name=f'dw_{name}/relu')(x)
    x = conv_block(x,
                   filters=64,
                   kernel_size=(1, 1),
                   padding='same',
                   use_bias=False,
                   name=f'pw_{name}',
                   pooling='global_avg' if last else None,
                   post_relu_gap=True,
                   add_batchnorm=True,
                   relu_activation=relu_activation)
    return x


def build_speech_commands_model(seed=42):
    set_random_seed(seed)

    classes = 12

    with set_akida_version(AkidaVersion.v2):
        # ReLU3.75 on Akida 2
        _, _, relu_activation = get_params_by_version()

        # uint8 MFCC features; the /255 rescaling is part of the model
        inputs = Input(shape=(49, 10, 1), name='input', dtype=tf.uint8)
        x = Rescaling(1. / 255, 0, name='rescaling')(inputs)

        x = conv_block(x,
                       filters=64,
                       kernel_size=(5, 5),
                       padding='same',
                       strides=(2, 2),
                       use_bias=False,
                       name='conv_0',
                       add_batchnorm=True,
                       relu_activation=relu_activation)

        for index in range(1, 5):
            x = _separable_block(x, index, relu_activation, last=(index == 4))

        x = dense_block(x,
                        units=classes,
                        name='dense_5',
                        use_bias=True,
                        relu_activation=False)
        x = Activation('softmax', name='act_softmax')(x)

    return Model(inputs, x, name='ds_cnn_kws')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Build the DS-CNN KWS model for Akida 2')
    parser.add_argument("-s",
                        "--savepath",
                        type=str,
                        default='./models/ds_cnn_speech_commands_untrained.h5',
                        help="Save model with the specified path + name")
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    model = build_speech_commands_model(seed=args.seed)
    model.summary()
    model.save(args.savepath, include_optimizer=False)
    print(f'Model saved to {args.savepath}')
