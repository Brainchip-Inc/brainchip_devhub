#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Train the Pico KWS model on your own keywords, from a config file.

It runs the same float training recipe as ``kws_train.py`` (AdamW, cosine decay, class-balanced
epochs) on a folder of your own recordings, with the augmentations you list in the config.
Unlike ``kws_train.py`` it picks the best epoch on a **validation split**, and reports
accuracy on a separate **test split** the training never saw.

Stages: load and split the data, build the augmentation pipeline, build the model, train,
save the best epoch, write the class names and a quantization calibration set next to it, and
score the held-out test split.

Usage:
    python kws_train_custom.py -c kws_config.yaml -s models/kws_custom.h5

See ``kws_config.yaml`` for the settings and the README for the folder layout. The model is
``tenn_recurrent_sc12`` with one output per class, unless the config's ``model`` names your
own ``module:function(num_classes)`` that returns an untrained keras model taking a
``(16384, 1)`` input.
"""
import argparse
import importlib
import json
import pathlib

import numpy as np
import tf_keras as keras
from tf_keras.utils import set_random_seed
from akida_models.tenn_recurrent.sc12_train import compile_model, train_model

from kws_augment import build_pipeline
from kws_custom_data import (calibration_samples, load_config, load_dataset_from_config,
                             make_eval_dataset, make_train_dataset)
from kws_eval import evaluate_kernelized
from kws_model import build_kws_model


def build_model(config, num_classes):
    """The model named in the config, or the default KWS model, with ``num_classes`` outputs."""
    if config['model'] is None:
        return build_kws_model(num_classes)
    module, function = config['model'].split(':', 1)
    return getattr(importlib.import_module(module), function)(num_classes=num_classes)


def sidecar(savemodel, suffix):
    """A file next to the saved model, named after it: ``models/x.h5`` -> ``models/x_<suffix>``."""
    path = pathlib.Path(savemodel)
    return path.with_name(f'{path.stem}_{suffix}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the Pico KWS model on your own keywords')
    parser.add_argument('-c', '--config', default='kws_config.yaml', help='Training config (YAML)')
    parser.add_argument('-s', '--savemodel', default='models/kws_custom.h5',
                        help='Where to save the trained model (.h5)')
    parser.add_argument('-d', '--data', default=None, help="Data folder (overrides the config's 'data')")
    parser.add_argument('-e', '--epochs', type=int, default=None, help="Overrides the config's 'epochs'")
    parser.add_argument('-b', '--batch-size', type=int, default=None,
                        help="Overrides the config's 'batch_size'")
    args = parser.parse_args()

    config = load_config(args.config)
    epochs = args.epochs or config['epochs']
    batch_size = args.batch_size or config['batch_size']
    set_random_seed(config['seed'])

    data = load_dataset_from_config(config, args.data)
    print(f'Classes: {data.labels}')
    for name in ('train', 'validation', 'test'):
        counts = np.bincount(getattr(data, name)[1], minlength=data.num_classes)
        print(f'  {name:<10} {int(counts.sum()):>6} clips  ' +
              ' '.join(f'{label}={c}' for label, c in zip(data.labels, counts)))
    if not len(data.validation[1]):
        raise SystemExit('The validation split is empty: add recordings or raise validation_fraction')

    augment = build_pipeline(config['augmentations'], data.noise)
    print(f'Augmentation: {augment if len(augment) else "none"}')

    train_ds, steps_per_epoch = make_train_dataset(
        *data.train, batch_size=batch_size,
        clips_per_class=config['clips_per_class'] or data.default_clips_per_class(),
        augment=augment, seed=config['seed'])
    val_ds = make_eval_dataset(*data.validation, batch_size=batch_size)

    model = build_model(config, data.num_classes)
    compile_model(model, steps_per_epoch * epochs)
    train_model(model, train_ds, val_ds, steps_per_epoch, epochs)

    save_path = pathlib.Path(args.savemodel)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path, include_optimizer=False)
    print(f'Saved {save_path}')
    labels_path = sidecar(save_path, 'labels.json')
    labels_path.write_text(json.dumps(data.labels) + '\n')
    calibration_path = sidecar(save_path, 'calibration.npz')
    np.savez(calibration_path, data=calibration_samples(data.train[0], seed=config['seed']))
    print(f'Saved {labels_path} and {calibration_path}')

    if len(data.test[1]):
        accuracy = evaluate_kernelized(model, *data.test, batch_size=batch_size)
        print(f'Held-out test accuracy (float, kernelized): {accuracy * 100:.2f}% '
              f'on {len(data.test[1])} clips')
    else:
        print('The test split is empty: no held-out accuracy to report.')
