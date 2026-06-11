#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
VWW training

Example
-------
    python vww_train.py -d /data/vww_coco2014_96/ -e 50  \\
        -l akidanet_vww_untrained.h5 -s akidanet_vww.h5
"""
import argparse

from tf_keras.losses import SparseCategoricalCrossentropy
from tf_keras.optimizers.legacy import Adam
from tf_keras.callbacks import LearningRateScheduler
from tf_keras import regularizers
from tf_keras.layers import ReLU
from tf_keras.utils import set_random_seed

from cnn2snn import load_quantized_model
from akida_models.training import RestoreBest

from vww_data import get_data


def get_custom_scheduler(initial_lr: float):
    """
    Custom LR scheduler:
    - Epochs  0-19:  initial_lr
    - Epochs 20-39:  initial_lr x 0.5
    - Epochs 40+:    initial_lr x 0.25
    """
    def lr_schedule(epoch: int, lr: float) -> float:
        if epoch < 20:
            return initial_lr
        elif epoch < 40:
            return initial_lr * 0.5
        else:
            return initial_lr * 0.25

    return LearningRateScheduler(lr_schedule)


def train_vww(model, train_ds, val_ds, epochs, learning_rate, regularization=None):
    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    if regularization is not None:
        print('Adding Activity Regularization to ReLU layers')
        regularizer = regularizers.L1L2(regularization, regularization)
        for layer in model.layers:
            if isinstance(layer, ReLU):
                layer.activity_regularizer = regularizer


    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------
    callbacks = []

    lr_scheduler = get_custom_scheduler(initial_lr=learning_rate)
    callbacks.append(lr_scheduler)

    # Model checkpoints (save best model and retrieve it when training is complete)
    restore_model = RestoreBest(model)
    callbacks.append(restore_model)

    history = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=val_ds,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras or .fbz akida model)')
    parser.add_argument('-s', '--savemodel', required=True,
                        help='Model save path')

    parser.add_argument('-d', '--data', default='./data/vw_coco2014_96',
                        help='VWW dataset root (contains train/ and val/ subdirs)')

    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('-reg', '--regularization', type=float, default=None,
                        help='Activity Regularization to increase sparsity')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    set_random_seed(args.seed)

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    model = load_quantized_model(args.loadmodel)

    # ---------------------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------------------
    train_ds, val_ds = get_data(args.data, model.input_shape[1:], args.batch_size)

    train_vww(model=model,
              train_ds=train_ds,
              val_ds=val_ds,
              epochs=args.epochs,
              learning_rate=args.learning_rate,
              regularization=args.regularization)
    
    model.save(args.savemodel, include_optimizer=False)
    print(f'Model saved as {args.savemodel}.')
