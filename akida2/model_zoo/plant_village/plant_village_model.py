#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Create a model for the PlantVillage dataset targeting the Akida 2 platform.

The model is based on the AkidaNet architecture with width multiplier alpha=0.5
(adequate for this 38-class task) and weights pre-trained on ImageNet. The
ImageNet-pretrained backbone is used as a feature extractor, with the top layers
replaced by a 38-class classification head for plant disease classification.

The model is set to 224x224 RGB input. Input scaling (divide by 255) is included
in the model via a Rescaling layer, so the preprocessing pipeline should NOT
apply any normalization but rather deliver inputs in the uint8 range.

Usage:
    python plant_village_model.py [-s OUTPUT_PATH]
"""

import argparse

from tf_keras import Model
from tf_keras.layers import Activation, Dropout, Reshape
from tf_keras.utils import set_random_seed

from akida_models import akidanet_imagenet, fetch_file
from akida_models.layer_blocks import dense_block
from akida_models.utils import get_params_by_version
from cnn2snn import set_akida_version, AkidaVersion


def build_plant_village_model(seed=42):
    set_random_seed(seed)

    # Number of classes in the PlantVillage dataset
    classes = 38

    with set_akida_version(AkidaVersion.v2):
        # Create a base model without top layers, with global average pooling
        base_model = akidanet_imagenet(input_shape=(224, 224, 3),
                                       classes=classes,
                                       alpha=0.5,
                                       include_top=False,
                                       pooling='avg')

        # Get ImageNet-pretrained weights and load them into the base model
        pretrained_weights = fetch_file(
            "https://data.brainchip.com/models/AkidaV2/akidanet/"
            "akidanet_imagenet_224_alpha_0.5.h5",
            fname="akidanet_imagenet_224_alpha_0.5.h5",
            cache_subdir='models')
        base_model.load_weights(pretrained_weights, by_name=True)

        # Version-appropriate ReLU activation for the head
        _, _, relu_activation = get_params_by_version(relu_v2='ReLU7.5')

        # Replace the classification head with one sized for PlantVillage
        x = base_model.output
        x = dense_block(x,
                        units=512,
                        name='fc_1',
                        add_batchnorm=True,
                        relu_activation=relu_activation)
        x = Dropout(0.5, name='dropout_1')(x)
        x = dense_block(x,
                        units=classes,
                        name='predictions',
                        add_batchnorm=False,
                        relu_activation=False)
        x = Activation('softmax', name='act_softmax')(x)
        x = Reshape((classes,), name='reshape')(x)

    model = Model(base_model.input, x, name='akidanet_plant_village')

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Build the AkidaNet-PlantVillage model for Akida 2')
    parser.add_argument("-s",
                        "--savepath",
                        type=str,
                        default='./models/akidanet_plant_village.h5',
                        help="Save model with the specified path + name")
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    model = build_plant_village_model(seed=args.seed)
    model.summary()
    model.save(args.savepath, include_optimizer=False)
    print(f'Model saved to {args.savepath}')
