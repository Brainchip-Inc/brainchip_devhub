#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
UORED-VAFCLS bearing fault model (akdcnn).

A 2D convolutional network over a framed 1 s vibration window, descended from
WDCNN but rebuilt to map wholly onto Akida 1 hardware.

Input representation
--------------------
The model takes the 42000-sample window already framed to (1200, 35, 1) by the
data pipeline, not the raw waveform. The framing is outside the graph because an
Akida model cannot contain a reshape - the deployed network receives the framed
tensor directly. Row j is samples [35j, 35j+35), so the row axis is coarse time
(1200 frames, 0.833 ms apart) and the column axis is fine phase within a frame,
much like the framing of a speech signal. The 7x7 stem therefore correlates 7
consecutive samples across 7 consecutive frames: a lag/phase view.

(The source repo's docstring claimed this stem was equivalent to a (49, 1)
kernel at stride 7. That was true only of an earlier variant which transposed
the two axes before framing; that transpose is not applied here, and the
equivalence does not hold for this layout.)

Input scaling is part of the model: a Rescaling layer inverts the data
pipeline's uint8 encoding, mapping [0, 255] back to the roughly [-1, 1] range
the network trains on. Keeping it in the graph means float training, quantized
training and Akida inference all see identical values, and cnn2snn folds the
layer into the first convolution at conversion time.

Hardware constraints that fixed the geometry
--------------------------------------------
Two AKD1500 constraints drove the design, and both are load-bearing:

* The long dimension has to come first, hence (1200, 35) rather than (35, 1200).
* Every layer's second spatial dimension must stay above 1, which is why the
  last two blocks omit their max-pool: the height runs 35 -> 29 -> 14 -> 7 -> 4
  -> 2 -> 1, and a sixth pool would take it to 0.

Initializers
------------
The source repo used custom initializers that reproduced PyTorch's defaults, to
keep a Keras port numerically comparable with a PyTorch reference. That
reference is not part of this example, and the custom bias initializer is what
forced a custom_object_scope workaround around every quantize and convert call
(Keras resolves initializers by name when it clones a model, including inside
the `cnn2snn convert` CLI where no scope can be installed). Keras defaults are
used here instead, which removes that whole class of problem.

Example
-------
    python uored_vafcls_model.py -s models/akdcnn_uored_vafcls_untrained.h5
"""

import argparse

from cnn2snn import AkidaVersion, set_akida_version
from tf_keras import Model
from tf_keras.layers import (BatchNormalization, Conv2D, Dense, Flatten, Input,
                             MaxPooling2D, ReLU, Rescaling)
from tf_keras.utils import set_random_seed

from uored_vafcls_data import INPUT_SHAPE, NUM_LABELS, ZERO_POINT

# BatchNorm settings carried over from the source model.
BN_MOMENTUM = 0.9
BN_EPSILON = 1e-5

# Filters per block, and whether the block pools. The last two do not: the
# height is already 1 by then (see the module docstring).
BLOCK_FILTERS = (32, 64, 64, 64, 64, 64)
BLOCK_POOLING = (True, True, True, True, False, False)

STEM_FILTERS = 16
STEM_KERNEL = (7, 7)
DENSE_UNITS = 100
RELU_MAX = 6.0


def _conv_block(x, filters, name, kernel=(3, 3), padding='same', pool=True,
                pool_padding='same'):
    """Conv -> BatchNorm -> ReLU6 [-> MaxPool], with no convolution bias.

    The bias is omitted because the BatchNorm that follows immediately
    reintroduces a per-channel offset, so a conv bias would be redundant, and
    cnn2snn folds the BatchNorm into the convolution at conversion.
    """
    x = Conv2D(filters, kernel, padding=padding, use_bias=False,
               name=f'{name}_conv')(x)
    x = BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPSILON,
                           name=f'{name}_bn')(x)
    x = ReLU(max_value=RELU_MAX, name=f'{name}_relu')(x)
    if pool:
        x = MaxPooling2D((2, 2), strides=(2, 2), padding=pool_padding,
                         name=f'{name}_pool')(x)
    return x


def build_uored_vafcls_model(seed=0):
    """Build the untrained akdcnn model.

    Args:
        seed (int): random seed for weight initialization.

    Returns:
        tf_keras.Model: the model, expecting uint8 inputs of shape INPUT_SHAPE
        and returning NUM_LABELS raw logits (one per fault mode; a healthy
        bearing is all four logits low, not a fifth class).
    """
    set_random_seed(seed)

    with set_akida_version(AkidaVersion.v1):
        inputs = Input(shape=INPUT_SHAPE, name='input')

        # Inverts the data pipeline's uint8 encoding. Folded into stem_conv at
        # conversion, giving input_scaling = (127, ZERO_POINT).
        x = Rescaling(1.0 / 127.0, -ZERO_POINT / 127.0, name='rescaling')(inputs)

        # Stem: a dense convolution with no padding, so the first layer sees the
        # raw framed signal.
        x = _conv_block(x, STEM_FILTERS, 'stem', kernel=STEM_KERNEL,
                        padding='valid', pool=True, pool_padding='valid')

        for i, (filters, pool) in enumerate(zip(BLOCK_FILTERS, BLOCK_POOLING),
                                            start=1):
            x = _conv_block(x, filters, f'block{i}', pool=pool)

        x = Flatten(name='flatten')(x)
        x = Dense(DENSE_UNITS, name='fc')(x)
        x = BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPSILON,
                               name='fc_bn')(x)
        x = ReLU(name='fc_relu')(x)
        outputs = Dense(NUM_LABELS, name='predictions')(x)

        model = Model(inputs, outputs, name='akdcnn_uored_vafcls')

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--savepath',
                        default='./models/akdcnn_uored_vafcls_untrained.h5',
                        help='Model save path')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    model = build_uored_vafcls_model(seed=args.seed)
    model.summary()
    model.save(args.savepath, include_optimizer=False)
    print(f'Model saved to {args.savepath}')
