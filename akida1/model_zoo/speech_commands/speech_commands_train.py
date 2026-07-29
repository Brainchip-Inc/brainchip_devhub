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
        --float
"""
import argparse

import tensorflow as tf
from tf_keras.losses import SparseCategoricalCrossentropy
from tf_keras.optimizers import Adam
from tf_keras.optimizers.schedules import CosineDecay
from tf_keras import regularizers
from tf_keras.layers import ReLU
from tf_keras.utils import set_random_seed
import yaml

from cnn2snn import load_quantized_model
from cnn2snn.quantization_layers import QuantizedReLU

from speech_commands_data_loader import compute_mfcc_range, get_datasets
from regularizers_custom import HoyerSquare
from speech_commands_model import build_ds_cnn

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

def train_speech_commands_float(model, train_ds, val_ds, config):
    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    print('Adding Activity Regularization to ReLU layers')
    reg_type = config.get("activity_reg_type", "l1l2")
    if reg_type == "hoyer":
        act_reg = HoyerSquare(config["activity_reg_hoyer_strength"])
    else:
        act_reg = regularizers.L1L2(config["activity_reg_l1"], config["activity_reg_l2"])
    for layer in model.layers:
        if isinstance(layer, ReLU) or "re_lu" in layer.name.lower():
            layer.activity_regularizer = act_reg
    
    steps = config["epochs_float"] * _steps_per_epoch(train_ds) 
    schedule = _lr_schedule(config["lr_float"], steps,
                            warmup_fraction=config.get("warmup_fraction", 0.1))
    model.compile(optimizer=Adam(learning_rate=schedule),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------
    model.fit(
        train_ds,
        epochs=config["epochs_float"],
        validation_data=val_ds,
        verbose=0,
    )


def train_speech_commands_qat(model, train_ds, val_ds, config):
    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    print('Adding Activity Regularization to QuantizedReLU layers')
    reg_type = config.get("activity_reg_type", "l1l2")
    if reg_type == "hoyer":
        qat_act_reg = HoyerSquare(config["activity_reg_hoyer_strength_qat"])
    else:
        qat_scale = float(config.get("qat_activity_reg_scale", 0.1))
        qat_act_reg = regularizers.L1L2(
            config["activity_reg_l1"] * qat_scale,
            config["activity_reg_l2"] * qat_scale,
        )
    for layer in model.layers:
        if isinstance(layer, QuantizedReLU):
            layer.activity_regularizer = qat_act_reg
    
    steps = config["epochs_qat"] * _steps_per_epoch(train_ds) 
    schedule = _lr_schedule(config["lr_qat"], steps,
                            warmup_fraction=config.get("warmup_fraction", 0.1))
    model.compile(optimizer=Adam(learning_rate=schedule),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------
    model.fit(
        train_ds,
        epochs=config["epochs_qat"],
        validation_data=val_ds,
        verbose=0,
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
    parser.add_argument('--float', action='store_true',
                        help='Whether this is the float phase (true) or the QAT phase (false)')
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Training config
    # ---------------------------------------------------------------------------
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_random_seed(cfg['seed'])

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
    model = load_quantized_model(args.loadmodel)

    if args.float:
        print('Float training')
        train_speech_commands_float(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            config=cfg
        )
    else:
        print('QAT')
        train_speech_commands_qat(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            config=cfg
        )

    model.save(args.savemodel, include_optimizer=False)
    print(f'Model saved as {args.savemodel}.')
