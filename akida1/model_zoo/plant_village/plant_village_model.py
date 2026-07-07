#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Create a model for PlantVillage dataset. This model optimally targets the
Akida 1 platform, and is based on the AkidaNet architecture, with
width multiplier alpha=0.5 (adequate for this 38-class task) and weights
pre-trained on ImageNet.

The model is set to 224x224 RGB input and the head modified to 38 classes
for plant disease classification (returning output logits for training without
a softmax function).

Note that the model itself includes input scaling to divide the inputs
by a factor of 255 - thus the preprocessing pipeline should NOT include
any normalization of the data, but rather deliver inputs in the uint8
range.

Usage:
    python plant_village_model.py [-s OUTPUT_PATH]
"""

import argparse

from tf_keras.utils import set_random_seed
from tf_keras import Model
from tf_keras.layers import Dropout

from akida_models import akidanet_imagenet_pretrained
from akida_models.layer_blocks import dense_block
from cnn2snn import set_akida_version, AkidaVersion


def build_plant_village_model(seed=42):
    set_random_seed(seed)

    # Create a base model with 38 classes
    classes = 38
    with set_akida_version(AkidaVersion.v1):
        base_model = akidanet_imagenet_pretrained(
                                  alpha=0.5,
                                  quantized=False)
        
    x = base_model.get_layer('separable_13/relu').output

    x = dense_block(x,
                    units=512,
                    name='fc_1',
                    add_batchnorm=True,
                    relu_activation='ReLU6.0')
    x = Dropout(0.5, name='dropout_1')(x)
    x = dense_block(x,
                    units=classes,
                    name='predictions',
                    add_batchnorm=False,
                    relu_activation=False)

    # Build the model
    model = Model(base_model.input, x, name='akidanet_plantvillage')
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Build the Akidanet-PlantVillage model for Akida 1')
    parser.add_argument("-s",
                        "--savepath",
                        type=str,
                        default='./models/akidanet_plant_village_untrained.h5',
                        help="Save model with the specified path + name")
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    model = build_plant_village_model(seed=args.seed)
    model.summary()
    model.save(args.savepath, include_optimizer=False)
    print(f'Model saved to {args.savepath}')
