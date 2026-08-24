#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Create a model for the VWW dataset targeting the Akida 2 platform.

This model is based on the AkidaNet architecture with a small width
multiplier (alpha=0.25, adequate for this binary task) at 96x96 RGB input.
Unlike the Akida 1 VWW example -- which fine-tunes an ImageNet-pretrained
backbone -- this Akida 2 model is built directly from the `akidanet_imagenet`
factory with `include_top=True` and a 2-class head (person / non-person),
returning output logits for training without a softmax.

The model expects uint8 inputs in the [0, 255] range: input scaling
(divide by 255) is included in the model via the Rescaling layer, so the
preprocessing pipeline should NOT apply any additional normalization.

Structure (matches the reference float model akidanet_vww.h5):
    input (uint8, 96x96x3)
      -> rescaling (scale=1/255, offset=0)
      -> conv_0..conv_3
      -> dw/pw_separable_4 .. dw/pw_separable_13
      -> global_avg pooling
      -> dropout (1e-3)
      -> classifier (Dense, 2 units, linear)

Usage:
    python vww_model.py [-s OUTPUT_PATH]
"""

import argparse

from tf_keras.utils import set_random_seed
from akida_models.imagenet import akidanet_imagenet
from cnn2snn import set_akida_version, AkidaVersion


def build_vww_model(seed=42):
    set_random_seed(seed)

    classes = 2
    # akidanet_imagenet is version-aware: under AkidaVersion.v2 it builds a
    # model using v2-compatible layer/activation variants. include_top=True
    # appends the standard global_avg -> dropout(1e-3) -> Dense(classes) head.
    # input_scaling=(255, 0) bakes the /255 rescaling into the model so that
    # uint8 inputs can be fed directly (the factory default is (128, -1),
    # which would be WRONG here -- it must be passed explicitly).
    with set_akida_version(AkidaVersion.v2):
        model = akidanet_imagenet(
            input_shape=(96, 96, 3),
            alpha=0.25,
            classes=classes,
            include_top=True,
            input_scaling=(255, 0),
        )

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Build the AkidaNet-VWW model for Akida 2')
    parser.add_argument("-s",
                        "--savepath",
                        type=str,
                        default='./models/akidanet_vww_untrained.h5',
                        help="Save model with the specified path + name")
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    model = build_vww_model(seed=args.seed)
    model.summary()
    model.save(args.savepath, include_optimizer=False)
    print(f'Model saved to {args.savepath}')
