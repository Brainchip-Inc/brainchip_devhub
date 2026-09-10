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

The model is built from akida_models' `ds_cnn_kws` factory. Input
preprocessing (rescaling) is included as part of the model via a Rescaling
layer: the model expects uint8 inputs, and the data pipeline delivers uint8
MFCC features accordingly.

Usage:
    python speech_commands_model.py [-s OUTPUT_PATH]
"""

import argparse

from tf_keras.utils import set_random_seed
from akida_models import ds_cnn_kws
from cnn2snn import set_akida_version, AkidaVersion


def build_speech_commands_model(seed=42):
    set_random_seed(seed)

    classes = 12

    # ds_cnn_kws is version-aware: under AkidaVersion.v2 it builds using
    # v2-compatible layer/activation variants. input_scaling=(255, 0) bakes the
    # /255 rescaling into the model so uint8 MFCC features can be fed directly.
    with set_akida_version(AkidaVersion.v2):
        model = ds_cnn_kws(
            input_shape=(49, 10, 1),
            classes=classes,
            include_top=True,
            input_scaling=(255, 0),
        )

    return model


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
