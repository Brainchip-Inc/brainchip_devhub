#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
UORED-VAFCLS bearing fault model (akdcnn).

A 2D convolutional network over a framed 1 s vibration window, descended from
WDCNN but rebuilt to map wholly onto Akida 1 hardware.

Input representation
--------------------
The model takes the 42000-sample window already framed to (300, 140, 1) by the
data pipeline, not the raw waveform. The framing is outside the graph because an
Akida model cannot contain a reshape - the deployed network receives the framed
tensor directly. Row j is samples [140j, 140j+140), so the row axis is coarse time
(300 frames, 3.33 ms apart) and the column axis is fine phase within a frame,
much like the framing of a speech signal. The 7x7 stem therefore correlates 7
consecutive samples across 7 consecutive frames: a lag/phase view.

Input scaling is part of the model: a Rescaling layer normalizes the data from 
the uint8 range furnished by the data preprocessing pipeline down to the roughly
[-1, 1] range the network trains on. Defining this Rescaling within the model 
graph means float training, quantized training and Akida inference all see 
identical values input values. The actual rescaling op itself is folded into the
weights of the first convolution layer by the cnn2snn package at conversion time.

Hardware constraints that fixed the geometry
--------------------------------------------
Some AKD1500 constraints drove the design:
* Most layers on Akida 1 only accept 4-bit inputs; only the specialised Input 
  Convolutional layer takes uint8 inputs, and we definitely want to use that
  here. However, second input dimension on that layer has a hard limit (256 
  on Akida 1). Thus the long input dimension has to go first, (300, 140) rather 
  than (140, 300)
* We found it helpful to use Global Average Pooling at the end of the 
  convolutional backbone. However, on Akida 1, that operation requires at least
  3 columns. This imposed constraints earlier in the model, limiting the 
  number of pooling steps, and forcing a relatively wide input.

Example
-------
    python uored_vafcls_model.py -s models/akdcnn_uored_vafcls_untrained.h5
"""

import argparse

from tf_keras import Model
from tf_keras.layers import (BatchNormalization, Conv2D, Dense, Input,
                             MaxPooling2D, ReLU, Rescaling, GlobalAveragePooling2D, Flatten)
from tf_keras.utils import set_random_seed

import akida
from cnn2snn import quantize, convert

from uored_vafcls_data import INPUT_SHAPE

def build_uored_vafcls_model(input_shape=INPUT_SHAPE, fault_classes=4, seed=0):
    """Build the untrained akdcnn model.

    Args:
        input_shape (tuple): input shape (H, W, C)
        fault_classes (int): number of different fault classes
        seed (int): random seed for weight initialization.

    Returns:
        tf_keras.Model: the model, expecting uint8 inputs of shape INPUT_SHAPE
        and returning fault_classes raw logits (one per fault mode; a healthy
        bearing is all four logits low, not a fifth class).
    """
    set_random_seed(seed)

    # Model definition parameters
    STEM_FILTERS = 16
    STEM_KERNEL_SIZE = 7

    BLOCK_FILTERS = [32, 64, 64, 64, 96, 128]
    BLOCK_POOLING = [True, True, True, True, True, False]

    DENSE_UNITS = 512

    # BatchNorm settings
    # Important in this case - unusually, we find that the tf_keras 
    # default values (0.99, 0.001) cause problems when the model is quantized
    BN_MOMENTUM = 0.9
    BN_EPSILON = 1e-5

    def _conv_block(x, filters, name, kernel=(3, 3), padding='same', pool=True,
                pool_padding='same'):
        """Re-usable block with
        Conv -> BatchNorm -> ReLU6 [-> MaxPool]
        """
        x = Conv2D(filters, kernel, padding=padding, use_bias=False,
                name=f'{name}_conv')(x)
        x = BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPSILON,
                            name=f'{name}_bn')(x)
        x = ReLU(max_value=6.0, name=f'{name}_relu')(x)
        if pool:
            x = MaxPooling2D((2, 2), strides=(2, 2), padding=pool_padding,
                            name=f'{name}_pool')(x)
        return x

    inputs = Input(shape=input_shape, name='input')

    # Normalize incoming data from the uint8 range to [-1, 1]
    # Folded into stem_conv weights at conversion
    x = Rescaling(1.0 / 128.0, -1.0, name='rescaling')(inputs)

    # Stem: a dense convolution with no padding
    x = Conv2D(STEM_FILTERS, STEM_KERNEL_SIZE, padding='valid', use_bias=False,
                    name='stem_conv')(x)
    x = BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPSILON,
                        name='stem_bn')(x)
    x = ReLU(max_value=6.0, name='stem_relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid',
                        name='stem_pool')(x)

    # Blocks
    for i, (filters, pool) in enumerate(zip(BLOCK_FILTERS[:-1], BLOCK_POOLING[:-1]),
                                        start=1):
        x = _conv_block(x, filters, f'block{i}', pool=pool)

    # Final block with Global Average Pooling
    # Note that for Akida 1, it needs to be placed before the neighbouring ReLU
    i = len(BLOCK_FILTERS)
    x = Conv2D(BLOCK_FILTERS[-1], 3, padding='same', use_bias=False,
               name=f'block{i}_conv')(x)
    x = BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPSILON,
                            name=f'block{i}_bn')(x)
    x = GlobalAveragePooling2D(name='gap')(x) # Akida 1: GAP must be placed before the ReLU
    x = ReLU(max_value=6.0, name=f'block{i}_relu')(x)

    # Dense Classifier Ending
    x = Dense(DENSE_UNITS, name='fc')(x)
    x = BatchNormalization(momentum=BN_MOMENTUM, epsilon=BN_EPSILON,
                            name='fc_bn')(x)
    x = ReLU(max_value=6.0, name='fc_relu')(x)
    outputs = Dense(fault_classes, name='predictions')(x)

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

    # Check that the model is hardware-compatible
    # This is a very useful check before doing any training if working on model re-design
    qmodel = quantize(model, weight_quantization=4, activ_quantization=4, input_weight_quantization=8)
    ak_model = convert(qmodel)
    device = akida.AKD1500()

    ak_model.map(device, mode=akida.MapMode.Minimal, hw_only=True)
    ak_model.summary()
