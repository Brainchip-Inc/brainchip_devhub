#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Accuracy of the Pico KWS model at each stage of the pipeline, on the Speech Commands
12-class test set (4,890 clips).

The stage is picked from the file you load:

  * ``tenn_recurrent_sc12.h5``                  kernelized float  (whole clip at once)
  * ``*_stateful.h5``                           stateful float    (256-sample chunks)
  * ``*_stateful_i8_w8_a8.h5``                  quantized stateful
  * ``*.fbz``                                   Akida model

Stateful models are scored the way Pico runs them: the clip goes through one 256-sample
chunk at a time with the state carried between chunks, giving 64 logit vectors per clip.
The clip's prediction is the argmax of the **mean of those 64 logit vectors** (not of their
softmax probabilities).

An Akida model is evaluated in the software simulator unless you pass ``--hardware``, which
runs it on the attached Pico FPGA.

Example
-------
    python kws_eval.py -l pretrained_models/tenn_recurrent_sc12.h5
    python kws_eval.py -l models/tenn_recurrent_sc12_stateful_i8_w8_a8.h5
"""
import argparse
import json
import os
import pathlib

import numpy as np
import tensorflow as tf
import tf_keras as keras
from tqdm import tqdm

import akida
import quantizeml  # noqa: F401  registers the quantized layers for load_model
from akida_models.tenn_recurrent.convert_recurrent import convert_to_stateful  # noqa: F401
from quantizeml.layers import StatefulRecurrent, reset_states, update_batch_size
from quantizeml.models.transforms.transforms_utils import get_layers_by_type

from kws_data import CHUNK_SIZE, SAMPLE_LENGTH, load_split

# quantizeml asserts on integer ranges while tracing; they are for debugging and slow
# the evaluation down.
os.environ.setdefault('ASSERT_ENABLED', '0')


def _clip_predictions(logits_sum, num_chunks):
    """Mean of the per-chunk logits, then argmax."""
    return np.argmax(logits_sum / num_chunks, axis=-1)


def evaluate_kernelized(model, X, y, batch_size=64):
    """Score a kernelized float model, which takes a whole 16,384-sample clip."""
    logits = model.predict(X, batch_size=batch_size, verbose=0)
    # The kernelized model returns one logit vector per output step: average them.
    logits = logits.reshape(len(X), -1, logits.shape[-1]).mean(axis=1)
    return float(np.mean(np.argmax(logits, axis=-1) == y))


def evaluate_stateful(model, X, y, chunk=CHUNK_SIZE, batch_size=64):
    """Score a stateful keras model (float or quantized), a batch of clips at a time.

    The state is reset between batches, so every clip starts from a zero state.
    """
    model = update_batch_size(model, batch_size)
    run = tf.function(model)
    num_chunks = X.shape[1] // chunk
    correct = 0
    for start in tqdm(range(0, len(X), batch_size), desc='Evaluating stateful model'):
        clips = X[start:start + batch_size]
        n = len(clips)
        if n < batch_size:      # the last batch is zero-padded to the model's batch size
            clips = np.concatenate([clips, np.zeros((batch_size - n,) + clips.shape[1:],
                                                    clips.dtype)])
        total = 0
        for t in range(num_chunks):
            frame = clips[:, t * chunk:(t + 1) * chunk, :]
            total = total + np.asarray(run(tf.convert_to_tensor(frame))).reshape(batch_size, -1)
        reset_states(model)
        correct += int(np.sum(_clip_predictions(total[:n], num_chunks) == y[start:start + n]))
    return correct / len(X)


def evaluate_akida(model, X, y, chunk=CHUNK_SIZE):
    """Score an Akida model, one clip at a time (batch size 1).

    Akida keeps the recurrent state inside the model, so the model is rebuilt from its layers
    between clips to start each one from a zero state.
    """
    num_chunks = X.shape[1] // chunk
    correct = 0
    for i in tqdm(range(len(X)), desc='Evaluating Akida model'):
        total = 0
        for t in range(num_chunks):
            frame = X[i, t * chunk:(t + 1) * chunk, :].astype(np.int16)
            total = total + model.forward(frame[np.newaxis, np.newaxis]).reshape(-1)
        correct += int(_clip_predictions(total, num_chunks) == y[i])
        model = akida.Model(model.layers)
    return correct / len(X)


def metrics_key(path, hardware=False):
    """Metrics key for an evaluated file, from its name (see docs/metrics.json)."""
    stem = pathlib.Path(path).stem
    if path.endswith('.fbz'):
        return 'akida_hw_acc' if hardware else 'akida_acc'
    if 'i8_w8_a8' in stem:
        return 'quant_acc'
    if 'stateful' in stem:
        return 'stateful_acc'
    return 'float_acc'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a Pico KWS model on Speech Commands (12 classes)')
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 keras model or .fbz Akida model)')
    parser.add_argument('-d', '--data', default=None,
                        help='TFDS data directory (default: ~/tensorflow_datasets)')
    parser.add_argument('-b', '--batch-size', type=int, default=64,
                        help='Clips per batch for keras models (Akida runs one clip at a time)')
    parser.add_argument('--hardware', action='store_true',
                        help='Run an Akida model on the attached Pico FPGA instead of the simulator')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Write the accuracy (and parameter count for the float model) to docs/metrics.json')
    args = parser.parse_args()

    X, y = load_split('test', SAMPLE_LENGTH, args.data)
    print(f'Test set: {X.shape[0]} clips of {X.shape[1]} samples')

    if args.loadmodel.endswith('.fbz'):
        model = akida.Model(args.loadmodel)
        if args.hardware:
            devices = akida.devices()
            if not devices:
                raise SystemExit('No Akida device found: run without --hardware to use the simulator')
            model.map(devices[0], hw_only=True)
        accuracy = evaluate_akida(model, X, y)
    else:
        model = keras.models.load_model(args.loadmodel, compile=False)
        if get_layers_by_type(model, StatefulRecurrent):
            accuracy = evaluate_stateful(model, X, y, batch_size=args.batch_size)
        else:
            accuracy = evaluate_kernelized(model, X, y, batch_size=args.batch_size)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    if args.save_metrics:
        # Updates the stored metrics used to generate the README. For code maintenance
        # only, run against the pretrained model and its converted variants.
        metrics_path = pathlib.Path(__file__).parent / 'docs' / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        key = metrics_key(args.loadmodel, args.hardware)
        metrics[key] = f'{accuracy * 100:.2f}%'
        if key == 'float_acc':
            metrics['params'] = f'{model.count_params():,}'
        metrics_path.write_text(json.dumps(metrics, indent=4, sort_keys=True) + '\n')
        print(f'Metrics saved to {metrics_path}')
