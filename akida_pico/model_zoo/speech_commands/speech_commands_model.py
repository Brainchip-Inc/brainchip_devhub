#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Create the recurrent TENN keyword spotting model for Akida Pico, and convert a
trained one to its stateful (streaming) form.

The model is `tenn_recurrent_sc12` from akida_models: six recurrent TENN blocks
(8 to 128 channels, temporal subsampling 4-4-2-2-2-2) and a 12-class recurrent head,
reading raw int16 audio. A Rescaling layer (divide by 2^15) is part of the model, so
the data pipeline delivers the raw int16 values.

The same weights run in two forms:
    - Training form: each recurrent layer is `Kernelized`, i.e. computed as a long
      temporal convolution over the whole clip (16384 samples), which trains
      efficiently on a GPU.
    - Stateful form: each layer becomes a `StatefulRecurrent` layer that processes the
      clip in chunks of `timesteps` samples and carries its internal state from one
      chunk to the next. This is the form that is quantized and runs on Akida Pico.

Usage:
    python speech_commands_model.py [-s OUTPUT_PATH]
    python speech_commands_model.py --stateful -l TRAINED.h5 -s STATEFUL.h5 [-ts 256]
"""
import argparse

from tf_keras.models import load_model
from tf_keras.utils import set_random_seed

from akida_models import tenn_recurrent_sc12
from akida_models.tenn_recurrent.convert_recurrent import convert_to_stateful

from speech_commands_data import NUM_CLASSES, SIGNAL_LENGTH

TIMESTEPS = 256   # samples per chunk in the stateful model (16 ms at 16 kHz)


def build_speech_commands_model(seed=42):
    set_random_seed(seed)
    return tenn_recurrent_sc12(input_shape=(SIGNAL_LENGTH, 1), num_classes=NUM_CLASSES)


def to_stateful(model, timesteps=TIMESTEPS):
    """Converts a trained Kernelized model to its StatefulRecurrent form."""
    return convert_to_stateful(model, timesteps=timesteps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Build the recurrent TENN Speech Commands model for Akida Pico, '
                    'or convert a trained one to stateful form')
    parser.add_argument('-s', '--savepath', type=str,
                        default='./models/tenn_recurrent_sc12_untrained.h5',
                        help='Save model with the specified path + name')
    parser.add_argument('--stateful', action='store_true',
                        help='Convert the model given with -l to stateful form')
    parser.add_argument('-l', '--loadmodel', type=str, default=None,
                        help='Trained model to convert (with --stateful)')
    parser.add_argument('-ts', '--timesteps', type=int, default=TIMESTEPS,
                        help='Samples per chunk in the stateful model')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    if args.stateful:
        if args.loadmodel is None:
            parser.error('--stateful needs -l/--loadmodel')
        model = to_stateful(load_model(args.loadmodel), timesteps=args.timesteps)
    else:
        model = build_speech_commands_model(seed=args.seed)
    model.summary()
    model.save(args.savepath, include_optimizer=False)
    print(f'Model saved to {args.savepath}')
