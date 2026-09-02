#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Create a model for the VWW dataset targeting the Akida 2 platform.

This model is the MobileNet (V1) architecture with a small width
multiplier (alpha=0.25, adequate for this binary task) at 96x96 RGB input.

The source model expects inputs scaled to the [-1, 1] range.
The final Akida model will require uint8 inputs. 
To enable a single data preprocessing pipeline across model
versions, rescaling for the tf_keras model version is implemented
within the model itself, as a Rescaling layer. That is added to the
model using a helper from quantizeml. The data pipeline thus delivers
inputs in the uint8 range.

Usage:
    python vww_model.py [-s OUTPUT_PATH]
"""

import argparse

from tf_keras.utils import set_random_seed
from tf_keras import Model
from tf_keras.applications.mobilenet import MobileNet
from akida_models.layer_blocks import dense_block
from quantizeml.models.transforms import insert_rescaling


def build_vww_model(seed = 42):
    set_random_seed(seed)

    classes = 2

    base_model = MobileNet(input_shape=(96, 96, 3),
                           alpha=0.25,
                           include_top=False,
                           weights='imagenet',
                           pooling='avg')

    # Pretrained mobilenet expects inputs in the [-1, 1] range
    # Include the relevant preprocessing (scale and shift) 
    # as a layer within the model
    base_model = insert_rescaling(base_model, scale=1/127.5, offset=-1)

    x = base_model.output
    # 2 class block
    x = dense_block(x,
                    units = classes,
                    name = 'predictions',
                    add_batchnorm = False,
                    relu_activation = False
                    )
    
    model = Model(base_model.input, x, name = 'mobilenet_vww')
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Build the MobileNet-VWW model for Akida 2')
    parser.add_argument("-s",
                        "--savepath",
                        type=str,
                        default='./models/mobilenet_vww_untrained.h5',
                        help="Save model with the specified path + name")
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    model = build_vww_model(seed=args.seed)
    model.summary()
    model.save(args.savepath, include_optimizer=False)
    print(f'Model saved to {args.savepath}')
