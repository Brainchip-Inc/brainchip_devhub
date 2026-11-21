#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Float training of the Pico KWS model (kernelized form) on Speech Commands, 12 classes.

Training uses the kernelized model, which takes a whole 16,384-sample clip. The recipe is
the one the pretrained weights were produced with:

  * AdamW (weight decay 0.05), peak learning rate 0.01, cosine decay with 2.5% linear warm-up
  * 200 epochs at batch size 128 for the reference model
  * class-balanced epochs: 3,100 clips per class each epoch (``unknown`` is downsampled,
    ``silence`` is upsampled)
  * no augmentation

The best epoch is restored at the end. Note that, as in the reference recipe, the best epoch
is chosen on the test split. For your own data keep a separate validation set (see the README).

Usage:
    python kws_train.py -l models/tenn_recurrent_sc12_untrained.h5 \
                        -s models/tenn_recurrent_sc12.h5 -e 200 -b 128
"""
import argparse

import tf_keras as keras
from tf_keras.utils import set_random_seed
from akida_models.tenn_recurrent.sc12_train import compile_model, train_model

from kws_data import get_train_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Pico KWS model on Speech Commands (12 classes)')
    parser.add_argument('-l', '--loadmodel', required=True, help='Untrained or partly trained model (.h5)')
    parser.add_argument('-s', '--savemodel', required=True, help='Where to save the trained model (.h5)')
    parser.add_argument('-e', '--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-d', '--data', default=None,
                        help='TFDS data directory (default: ~/tensorflow_datasets)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_random_seed(args.seed)
    model = keras.models.load_model(args.loadmodel, compile=False)

    train_ds, val_ds, train_steps, _, _ = get_train_data(args.batch_size, args.data)
    compile_model(model, train_steps * args.epochs)
    train_model(model, train_ds, val_ds, train_steps, args.epochs)

    model.save(args.savemodel, include_optimizer=False)
    print(f'Saved {args.savemodel}')
