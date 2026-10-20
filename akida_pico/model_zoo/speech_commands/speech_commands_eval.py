#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Evaluate the recurrent TENN keyword spotting model on the Speech Commands test split.

Works on every stage of the pipeline, picked from the model itself:
    - training (Kernelized) model, .h5: whole 16384-sample clips;
    - stateful model, float or 8-bit quantized, .h5: each clip streamed in chunks of
      256 samples, state carried between chunks;
    - Akida model, .fbz: the same streaming, run with the Akida runtime (software
      simulation of the Akida Pico IP when no device is attached).

For the streaming models there is one prediction per chunk; the clip's prediction is
the class with the highest logit averaged over all its chunks. Reported metrics are
accuracy and macro-F1 over the 12 classes, per 1 s clip.

Example
-------
    python speech_commands_eval.py -l models/tenn_recurrent_sc12.h5
    python speech_commands_eval.py -l models/tenn_recurrent_sc12_stateful_i8_w8_a8.fbz
    python speech_commands_eval.py --reference     # akida_models' pretrained models
"""
import argparse
import json
import os
import pathlib

import numpy as np
import tensorflow as tf
from tqdm import tqdm

import akida
from cnn2snn import convert
from quantizeml.layers import StatefulRecurrent, reset_states, update_batch_size
from quantizeml import load_model
from quantizeml.models.transforms.transforms_utils import get_layers_by_type

from akida_models.tenn_recurrent.sc12_training_utils import f1_score_stateful

from speech_commands_data import NUM_CLASSES, get_test_data

STATEFUL_BATCH = 163    # divides the 4890 test clips exactly, so none are dropped


def _scores(labels, preds):
    acc = float(np.mean(preds == labels))
    f1 = float(f1_score_stateful(labels, preds, num_classes=NUM_CLASSES, average='macro'))
    return acc, f1


def evaluate_training_model(model, clips, labels, batch_size=128):
    """Kernelized model: one prediction per whole clip."""
    logits = model.predict(clips, batch_size=batch_size, verbose=0)
    return _scores(labels, np.argmax(logits, axis=-1))


def evaluate_stateful_model(model, clips, labels, batch_size=STATEFUL_BATCH):
    """Stateful tf_keras model: stream chunks, average the per-chunk logits."""
    timestep = model.input_shape[1]
    model = update_batch_size(model, batch_size)
    step = tf.function(model)
    preds = []
    for start in tqdm(range(0, len(labels), batch_size), desc='Stateful model'):
        batch = clips[start:start + batch_size].reshape(batch_size, -1, timestep, 1)
        reset_states(model)
        summed = 0
        for t in range(batch.shape[1]):
            summed += tf.reduce_mean(step(batch[:, t]), axis=1)
        preds.append(np.argmax(summed, axis=-1))
    return _scores(labels, np.concatenate(preds))


def evaluate_akida_model(akida_model, clips, labels):
    """Akida model: stream each clip chunk by chunk, average the per-chunk logits."""
    timestep = akida_model.input_shape[1]
    preds = []
    for clip in tqdm(clips, desc='Akida model'):
        # A fresh model from the same layers starts from zero state
        model = akida.Model(akida_model.layers)
        chunks = clip.astype(np.int16).reshape(-1, 1, timestep, 1)
        summed = sum(model.forward(chunk[np.newaxis]).reshape(-1) for chunk in chunks)
        preds.append(np.argmax(summed))
    return _scores(labels, np.array(preds))


def pico_mapping(akida_model):
    """Maps a copy of the model onto the Akida Pico virtual device (akida.PicoIP).

    Mapping checks the model against the Pico hardware limits without a device
    attached. Returns metrics: hardware sequences, program size and NP components.
    """
    model = akida.Model(akida_model.layers)
    model.map(akida.PicoIP())
    hw = [s for s in model.sequences if s.backend == akida.BackendType.Hardware]
    components = {np_.type.name for s in hw for p in s.passes for layer in p.layers
                  if layer.mapping is not None for np_ in layer.mapping.nps}
    counts = {t: sum(1 for s in hw for p in s.passes for layer in p.layers
                     if layer.mapping is not None for np_ in layer.mapping.nps
                     if np_.type.name == t) for t in components}
    return {
        'pico_hw_sequences': str(len(hw)),
        'pico_sw_sequences': str(len(model.sequences) - len(hw)),
        'pico_program_bytes': f'{sum(len(s.program) for s in hw):,}',
        'pico_components': ', '.join(f'{n} {t}' for t, n in sorted(counts.items())),
    }


def evaluate(model, clips, labels):
    """Dispatches on the model type; returns (accuracy, macro_f1)."""
    if isinstance(model, akida.Model):
        return evaluate_akida_model(model, clips, labels)
    if get_layers_by_type(model, StatefulRecurrent):
        return evaluate_stateful_model(model, clips, labels)
    return evaluate_training_model(model, clips, labels)


def stage_of(path, model):
    """metrics.json key prefix for a model file."""
    if isinstance(model, akida.Model):
        return 'w8a8_akida'
    if '_i8_w8_a8' in path:
        return 'w8a8_quant'
    if get_layers_by_type(model, StatefulRecurrent):
        return 'stateful'
    return 'float'


def save_metrics(updates):
    path = pathlib.Path(__file__).parent / 'docs' / 'metrics.json'
    metrics = json.loads(path.read_text()) if path.exists() else {}
    metrics.update(updates)
    path.write_text(json.dumps(metrics, indent=4) + '\n')
    print(f'Metrics saved to {path}')


def _formatted(prefix, acc, f1):
    return {f'{prefix}_acc': f'{acc * 100:.2f}%', f'{prefix}_f1': f'{f1:.4f}'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', help='Model to evaluate (.h5 or .fbz)')
    parser.add_argument('-d', '--data', default='./data/speech_commands',
                        help='TFDS data_dir for the speech_commands dataset')
    parser.add_argument('--reference', action='store_true',
                        help="Evaluate akida_models' pretrained models instead")
    parser.add_argument('--save-metrics', action='store_true',
                        help='Write the results into docs/metrics.json (maintenance only)')
    args = parser.parse_args()
    if not (args.loadmodel or args.reference):
        parser.error('give -l/--loadmodel or --reference')

    os.environ['ASSERT_ENABLED'] = '0'
    clips, labels = get_test_data(args.data)
    updates = {}

    if args.reference:
        from akida_models import tenn_recurrent_sc12_pretrained
        float_ref = tenn_recurrent_sc12_pretrained(quantized=False)
        quant_ref = tenn_recurrent_sc12_pretrained(quantized=True)
        for prefix, model in [('ref_stateful', float_ref), ('ref_w8a8_quant', quant_ref),
                              ('ref_w8a8_akida', convert(quant_ref))]:
            acc, f1 = evaluate(model, clips, labels)
            print(f'{prefix}: accuracy {acc * 100:.2f}%, macro-F1 {f1:.4f}')
            updates.update(_formatted(prefix, acc, f1))
    else:
        path = args.loadmodel
        model = akida.Model(path) if path.endswith('.fbz') else load_model(path)
        prefix = stage_of(path, model)
        acc, f1 = evaluate(model, clips, labels)
        print(f'{path}: accuracy {acc * 100:.2f}%, macro-F1 {f1:.4f}')
        updates.update(_formatted(prefix, acc, f1))
        if prefix == 'w8a8_quant':
            updates['params'] = f'{model.count_params():,}'
        if prefix == 'w8a8_akida':
            mapping = pico_mapping(model)
            print(f'Akida Pico mapping: {mapping}')
            updates.update(mapping)

    if args.save_metrics:
        save_metrics(updates)
