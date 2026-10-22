#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Create an Akida U-Net model for Portrait128 person segmentation, targeting the
Akida 2 platform.

The model is a binary segmentation network on 128 x 128 RGB input. Its output is
a 128 x 128 x 1 map of logits: one value per pixel, positive for "person" and
negative for "background".

Architecture:
  * Encoder: AkidaNet (`akida_models.akidanet_imagenet`) with width multiplier
    alpha=0.5 and no classification top. Five stride-2 stages take the 128 x 128
    input down to a 4 x 4 x 512 feature map.
  * Decoder: five separable transposed-convolution blocks
    (`sepconv_transpose_block`), each doubling the spatial size, back up to
    128 x 128, with Dropout(0.5) after each block. Pointwise weights in the
    decoder carry an L2 penalty of 4e-5.
  * Head: a 1 x 1 convolution to one channel, with no activation (logits).

There are no skip connections between encoder and decoder. Upsampling with
transposed convolutions is an Akida 2 feature: this model cannot be built for
Akida 1.

Inputs are uint8 in [0, 255]. The encoder's `input_scaling=(128, -1)` adds a
Rescaling layer that maps them to [-1, 1) inside the model, so the same uint8
data pipeline feeds the float, quantized and Akida models. At conversion the
rescaling is folded into the Akida input layer.

Encoder initialization: with `base_weights='imagenet'` the encoder starts from
BrainChip's ImageNet-trained AkidaNet 0.5 (Akida 2) float weights, loaded by
layer name; the decoder and head start from random weights. That is how the
training pipeline builds the model. Convolution weights don't depend on the
input size, so weights trained at 224 x 224 load directly into this
128 x 128 model. Without `base_weights` the whole model starts from random
weights.

Usage:
    python portrait128_model.py [-bw imagenet] [-s OUTPUT_PATH]
"""

import argparse

from tf_keras import Model, regularizers
from tf_keras.layers import Dropout
from tf_keras.utils import set_random_seed

from cnn2snn import set_akida_version, AkidaVersion
from akida_models import akidanet_imagenet
from akida_models.layer_blocks import sepconv_transpose_block, conv_block
from akida_models.model_io import get_model_path
from akida_models.utils import fetch_file

# ImageNet-trained AkidaNet 0.5 float weights (Akida 2) from the akida_models model store
IMAGENET_WEIGHTS = 'akidanet_imagenet_224_alpha_0.5.h5'
IMAGENET_WEIGHTS_HASH = 'd818dec7c924757de2e6970b996da9c807ccdf9a9044262091ce526678c0640c'


def _fetch_imagenet_weights():
    """Download (or reuse the cached copy of) the ImageNet AkidaNet 0.5 weights."""
    with set_akida_version(AkidaVersion.v2):
        url, fname, file_hash = get_model_path('akidanet', model_name_v2=IMAGENET_WEIGHTS,
                                               file_hash_v2=IMAGENET_WEIGHTS_HASH)
    return fetch_file(url, fname=fname, file_hash=file_hash, cache_subdir='models')


def build_portrait128_model(input_shape=(128, 128, 3),
                            alpha=0.5,
                            input_scaling=(128, -1),
                            base_weights=None,
                            seed=42):
    """Instantiates the Akida U-Net architecture for Portrait128.

    Args:
        input_shape (tuple, optional): input shape tuple. Defaults to (128, 128, 3).
        alpha (float, optional): controls the width (number of filters) of the model.
            Defaults to 0.5.
        input_scaling (tuple, optional): scale factor and offset to apply to inputs.
            Defaults to (128, -1). Following Akida convention, the scale factor is
            used as a divisor.
        base_weights (str, optional): weights loaded by layer name after the model
            is built: 'imagenet' for the ImageNet AkidaNet 0.5 encoder weights, or the
            path to an .h5 file. Defaults to None (random initialization).
        seed (int, optional): random seed for weight initialization. Defaults to 42.

    Returns:
        tf_keras.Model: a Keras Model instance.
    """
    set_random_seed(seed)

    # Transposed convolutions are only available on Akida 2
    with set_akida_version(AkidaVersion.v2):
        # Define weight regularization, will apply to pointwise weights of sepconv transposed layers
        weight_regularizer = regularizers.l2(4e-5)

        # Create an AkidaNet network without top layers
        encoder = akidanet_imagenet(input_shape=input_shape,
                                    alpha=alpha,
                                    include_top=False,
                                    input_scaling=input_scaling)

        # Add the decoder layers
        x = encoder.layers[-1].output
        x = sepconv_transpose_block(x,
                                    filters=int(512 * alpha),
                                    kernel_size=(3, 3),
                                    strides=2,
                                    padding='same',
                                    kernel_initializer='he_normal',
                                    add_batchnorm=True,
                                    relu_activation='ReLU7.5',
                                    name='sepconv_t_0',
                                    pointwise_regularizer=weight_regularizer)
        x = Dropout(0.5)(x)
        x = sepconv_transpose_block(x,
                                    filters=int(256 * alpha),
                                    kernel_size=(3, 3),
                                    strides=2,
                                    padding='same',
                                    kernel_initializer='he_normal',
                                    add_batchnorm=True,
                                    relu_activation='ReLU7.5',
                                    name='sepconv_t_1',
                                    pointwise_regularizer=weight_regularizer)
        x = Dropout(0.5)(x)
        x = sepconv_transpose_block(x,
                                    filters=int(128 * alpha),
                                    kernel_size=(3, 3),
                                    strides=2,
                                    padding='same',
                                    kernel_initializer='he_normal',
                                    add_batchnorm=True,
                                    relu_activation='ReLU7.5',
                                    name='sepconv_t_2',
                                    pointwise_regularizer=weight_regularizer)
        x = Dropout(0.5)(x)
        x = sepconv_transpose_block(x,
                                    filters=int(64 * alpha),
                                    kernel_size=(3, 3),
                                    strides=2,
                                    padding='same',
                                    kernel_initializer='he_normal',
                                    add_batchnorm=True,
                                    relu_activation='ReLU7.5',
                                    name='sepconv_t_3',
                                    pointwise_regularizer=weight_regularizer)
        x = Dropout(0.5)(x)
        x = sepconv_transpose_block(x,
                                    filters=int(32 * alpha),
                                    kernel_size=(3, 3),
                                    strides=2,
                                    padding='same',
                                    kernel_initializer='he_normal',
                                    add_batchnorm=True,
                                    relu_activation='ReLU7.5',
                                    name='sepconv_t_4',
                                    pointwise_regularizer=weight_regularizer)
        x = Dropout(0.5)(x)
        x = conv_block(x, filters=1, kernel_size=(1, 1), relu_activation=False, name='head')

    # Build the whole model: encoder followed by decoder
    model = Model(inputs=encoder.input, outputs=x, name='akida_unet')

    if base_weights is not None:
        if base_weights == 'imagenet':
            base_weights = _fetch_imagenet_weights()
        model.load_weights(base_weights, by_name=True)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Build the Akida U-Net Portrait128 model for Akida 2')
    parser.add_argument("-s",
                        "--savepath",
                        type=str,
                        default='./models/akida_unet_portrait128.h5',
                        help="Save model with the specified path + name")
    parser.add_argument("-bw",
                        "--base_weights",
                        type=str,
                        default=None,
                        help="Weights to load by layer name: 'imagenet' or an .h5 path")
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    model = build_portrait128_model(base_weights=args.base_weights, seed=args.seed)
    model.summary()
    model.save(args.savepath, include_optimizer=False)
    print(f'Model saved to {args.savepath}')
