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

from tf_keras import Model
from tf_keras.losses import SparseCategoricalCrossentropy
from tf_keras.optimizers.legacy import Adam
from tf_keras.optimizers.schedules import CosineDecay
from tf_keras.callbacks import LearningRateScheduler
from tf_keras import regularizers
from tf_keras.layers import ReLU
from tf_keras.utils import set_random_seed

from cnn2snn import load_quantized_model

from vww_data import get_data



def train_vww(model, train_ds, val_ds, epochs, learning_rate, regularization=None):
    
    steps_per_epoch = len(train_ds)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup

    lr_scheduler = CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=total_steps - warmup_steps,
        warmup_target=learning_rate,
        warmup_steps=warmup_steps,
    )
    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    model.compile(optimizer=Adam(learning_rate=lr_scheduler),
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
    history = model.fit(
        train_ds,
        epochs=epochs,
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
