#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Speech Commands training

Example
-------
    python speech_commands_train.py \\
        -l models/speech_commands_untrained.h5 \\
        -s models/speech_commands.h5 \\
        -d /home/datasets/sc10/ \\
        --config configs/training_cfg.yml \\
"""
import argparse

import tensorflow as tf
from tf_keras.losses import SparseCategoricalCrossentropy
from tf_keras.optimizers import Adam
from tf_keras.optimizers.schedules import CosineDecay
from tf_keras.layers import ReLU
from tf_keras.utils import set_random_seed
import yaml

from cnn2snn import load_quantized_model
from cnn2snn.quantization_layers import QuantizedReLU

from speech_commands_data_loader import compute_mfcc_range, get_datasets
from regularizers_custom import HoyerSquare

# Must be called before any TF ops to make GPU ops (conv backward passes,
# bilinear resize, etc.) deterministic. Has a small throughput cost.
tf.config.experimental.enable_op_determinism()


def _steps_per_epoch(ds):
        """Return the number of batches in ds. Falls back to counting for unknown cardinality."""
        cardinality = ds.cardinality().numpy()
        if cardinality >= 0:
            return int(cardinality)
        # from_generator and some sharded datasets return UNKNOWN (-2); count by iterating once
        return sum(1 for _ in ds)

def _lr_schedule(peak_lr, total_steps, warmup_fraction=0.1, initial_learning_rate=1e-6):
        return CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=total_steps,
            warmup_target=peak_lr,
            warmup_steps=int(warmup_fraction * total_steps),
        )

def train_speech_commands(model, train_ds, val_ds, 
                          epochs,
                          peak_lr,
                          warmup_fraction,
                          act_reg_strength,
                          seed=111):
    set_random_seed(seed)
    
    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    if act_reg_strength>0:
        act_reg = HoyerSquare(act_reg_strength)
        print('Adding Activity Regularization (Hoyer-Square) to ReLU layers')
        for layer in model.layers:
            if isinstance(layer, (ReLU, QuantizedReLU)) or "re_lu" in layer.name.lower():
                layer.activity_regularizer = act_reg
    
    steps = epochs * _steps_per_epoch(train_ds) 
    schedule = _lr_schedule(peak_lr, steps,
                            warmup_fraction=warmup_fraction)
    model.compile(optimizer=Adam(learning_rate=schedule),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------
    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        verbose=1,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras or .fbz akida model)')
    parser.add_argument('-s', '--savemodel', required=True,
                        help='Model save path')

    parser.add_argument('-d', '--data', default='./data/sc10',
                        help='Speech Commands tfds data directory')

    parser.add_argument("--config", default="configs/training_cfg.yml",
                        help='Model training configuration file')
    parser.add_argument('--qat', action='store_true',
                        help='Use qat rather than [default] float config settings')
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Training config
    # ---------------------------------------------------------------------------
    with open(args.config) as f:
        cfg = yaml.safe_load(f)


    # ---------------------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------------------
    data_transform = compute_mfcc_range(data_dir=args.data)
    train_ds, test_ds, val_ds = get_datasets(
        data_dir=args.data,
        batch_size=cfg["batch_size"],
        data_transform=data_transform,
        aug_enabled=cfg.get("aug_enabled", False),
        aug_time_shift_max_ms=cfg.get("aug_time_shift_max_ms", 100),
        aug_freq_mask_param=cfg.get("aug_freq_mask_param", 2),
        aug_time_mask_param=cfg.get("aug_time_mask_param", 10),
        shuffle_seed=cfg.get("seed"),
        aug_seed=cfg.get("seed"),
    )

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    set_random_seed(cfg.get("seed"))
    model = load_quantized_model(args.loadmodel)


    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------
    warmup_fraction = cfg.get("warmup_fraction", 0.1)
    if args.qat:
        # Get QAT-specific training params
        act_reg_strength = cfg["activity_reg_hoyer_strength_qat"]
        epochs = cfg["epochs_qat"]
        peak_lr = cfg["lr_qat"]
    else:
        # Float training params
        act_reg_strength = cfg["activity_reg_hoyer_strength"]
        epochs = cfg["epochs_float"]
        peak_lr = cfg["lr_float"]
        

    train_speech_commands(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        epochs=epochs,
        peak_lr=peak_lr,
        warmup_fraction=warmup_fraction,
        act_reg_strength=act_reg_strength,
        seed=cfg.get("seed")
    )

    model.save(args.savemodel, include_optimizer=False)
    print(f'Model saved as {args.savemodel}.')
