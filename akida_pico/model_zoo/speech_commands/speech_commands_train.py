#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Train the recurrent TENN keyword spotting model (training form, full-length clips).

The recipe is the one akida_models uses for `tenn_recurrent_sc12`: AdamW (weight decay
0.05), learning rate 0.01 with a linear warm-up over the first 2.5% of steps then cosine
decay, batch size 128, class-balanced epochs. One difference: the best epoch is chosen
on the validation split (macro-F1, since validation is dominated by `_unknown_`), and
the test split is kept for reporting.

Example
-------
    python speech_commands_train.py -e 200 \\
        -l models/tenn_recurrent_sc12_untrained.h5 -s models/tenn_recurrent_sc12.h5
"""
import argparse
import os

from tf_keras.losses import SparseCategoricalCrossentropy
from tf_keras.metrics import SparseCategoricalAccuracy
from tf_keras.models import load_model
from tf_keras.optimizers.experimental import AdamW
from tf_keras.utils import set_random_seed

from akida_models.param_scheduler import CosineDecayWithLinearWarmup
from akida_models.training import RestoreBest
from akida_models.tenn_recurrent.sc12_training_utils import MacroF1

from speech_commands_data import NUM_CLASSES, get_data

LEARNING_RATE = 0.01
WARMUP_RATIO = 0.025
WEIGHT_DECAY = 0.05


def train_speech_commands(model, train_ds, val_ds, train_steps, val_steps, epochs):
    scheduler = CosineDecayWithLinearWarmup(LEARNING_RATE, WARMUP_RATIO, train_steps * epochs)
    model.compile(optimizer=AdamW(learning_rate=scheduler, weight_decay=WEIGHT_DECAY),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[SparseCategoricalAccuracy(name='accuracy'),
                           MacroF1(num_classes=NUM_CLASSES)])
    # Keep the weights of the epoch with the best validation macro-F1
    callbacks = [RestoreBest(model, monitor='val_macro_f1')]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, steps_per_epoch=train_steps,
              validation_steps=val_steps, callbacks=callbacks, verbose=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True, help='Model to load (.h5)')
    parser.add_argument('-s', '--savemodel', required=True, help='Model save path')
    parser.add_argument('-d', '--data', default='./data/speech_commands',
                        help='TFDS data_dir for the speech_commands dataset')
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-e', '--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # quantizeml's internal assertions slow training down and aren't needed here
    os.environ['ASSERT_ENABLED'] = '0'
    set_random_seed(args.seed)

    model = load_model(args.loadmodel)
    train_ds, val_ds, train_steps, val_steps = get_data(
        args.data, batch_size=args.batch_size, timestep=model.input_shape[1], seed=args.seed)

    train_speech_commands(model, train_ds, val_ds, train_steps, val_steps, args.epochs)

    model.save(args.savemodel, include_optimizer=False)
    print(f'Model saved as {args.savemodel}.')
