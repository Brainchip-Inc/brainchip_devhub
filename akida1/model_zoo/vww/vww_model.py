#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Create a model for VWW dataset. This model optimally targtes the
Akida 1 platform, and is based on the AkidaNet architecture, with
small width, alpha=0.25 (adequate for this task) and weights pre-trained
on ImageNet.

The model is adjusted to 96x96 RGB input (down from 224x224 for the
base model) and the head modified to two classes for Visual Wake Words
(person / non-person, returning output logits for training without a 
softmax function).

Note that the model itself includes input scaling to divide the inputs
by a factor of 255 - thus the preprocessing pipeline should NOT include
any normalization of the data, but rather deliver inputs in the uint8
range.

Usage:
    python vww_model.py [-s OUTPUT_PATH]
"""

import argparse

from tf_keras.utils import set_random_seed
from tf_keras import Model
from akida_models.layer_blocks import dense_block
from akida_models.imagenet.imagenet_train import rescale
from akida_models import akidanet_imagenet_pretrained
from cnn2snn import set_akida_version, AkidaVersion

def build_vww_model(seed = 42):
    set_random_seed(seed)

    classes = 2
    with set_akida_version(AkidaVersion.v1):
        base_model = akidanet_imagenet_pretrained(
                                           
                                            alpha=0.25,
                                            quantized=False
        )

    x = base_model.get_layer('separable_13/relu').output
    # 2 class block
    x = dense_block(x,
                    units = classes,
                    name = 'predictions',
                    add_batchnorm = False,
                    relu_activation = False
                    )
    
    model = Model(base_model.input, x, name = 'akidanet_vww')
    model = rescale(model, (96, 96))
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build the Akidanet-VWW model for Akida 1')
    parser.add_argument("-s",
                        "--savepath",
                        type=str,
                        default='./models/akidanet_vww_untrained.h5',
                        help="Save model with the specified path + name")
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    set_random_seed(args.seed)
    
    model = build_vww_model()
    model.summary()
    model.save(args.savepath, include_optimizer=False)
    print(f'Model saved to {args.savepath}')