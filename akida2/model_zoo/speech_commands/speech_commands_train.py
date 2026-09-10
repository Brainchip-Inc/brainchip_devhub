#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Speech Commands Keyword Spotting (KWS) training.

Example
-------
    python speech_commands_train.py -e 16 \\
        -l models/ds_cnn_speech_commands_untrained.h5 -s models/ds_cnn_speech_commands.h5
"""
import argparse
import tensorflow as tf

from tf_keras.losses import SparseCategoricalCrossentropy
from tf_keras.optimizers.legacy import Adam
from tf_keras.optimizers.schedules import CosineDecay
from tf_keras import regularizers
from tf_keras.layers import ReLU
from tf_keras.utils import set_random_seed

from cnn2snn import load_quantized_model

from speech_commands_data import get_data, compute_mfcc_range

# Must be called before any TF ops to make GPU ops deterministic. Small
# throughput cost.
tf.config.experimental.enable_op_determinism()


def train_speech_commands(model, train_ds, val_ds, epochs, learning_rate, regularization=None, seed=42):
    set_random_seed(seed)

    steps_per_epoch = len(train_ds)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup

    lr_scheduler = CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=total_steps - warmup_steps,
        warmup_target=learning_rate,
        warmup_steps=warmup_steps,
    )

    if regularization is not None:
        print('Adding Activity Regularization to ReLU layers')
        regularizer = regularizers.L1L2(regularization, regularization)
        for layer in model.layers:
            if isinstance(layer, ReLU):
                layer.activity_regularizer = regularizer

    model.compile(optimizer=Adam(learning_rate=lr_scheduler),
                  loss=SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras or quantized model)')
    parser.add_argument('-s', '--savemodel', required=True,
                        help='Model save path')
    parser.add_argument('-d', '--data', default='./data/speech_commands',
                        help='tfds data_dir for the speech_commands dataset')
    parser.add_argument('-b', '--batch_size', type=int, default=100)
    parser.add_argument('-e', '--epochs', type=int, default=16)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('-reg', '--regularization', type=float, default=None,
                        help='Activity Regularization to increase sparsity')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    model = load_quantized_model(args.loadmodel)

    # ---------------------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------------------
    # MFCC features are scaled to uint8 using percentile bounds from the train
    # split (matches the model's uint8 input expectation).
    data_transform = compute_mfcc_range(data_dir=args.data)
    train_ds, test_ds, val_ds = get_data(
        args.data, batch_size=args.batch_size, data_transform=data_transform,
        seed=args.seed)

    train_speech_commands(model=model,
              train_ds=train_ds,
              val_ds=val_ds,
              epochs=args.epochs,
              learning_rate=args.learning_rate,
              regularization=args.regularization,
              seed=args.seed)

    model.save(args.savemodel, include_optimizer=False)
    print(f'Model saved as {args.savemodel}.')
