#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Eyetracking evaluation for tf_keras/quantized or Akida models.

Reports mean angular error in degrees (this is a regression task, not
classification, so there is no "accuracy" metric here). Two numbers are
always reported:
  - overall: mean error across every timestep in every window.
  - last-timestep: mean error at the final (most causal context) timestep,
    the number comparable to a real streaming deployment's steady-state
    accuracy once the receptive field has filled in.

The model predicts continuously at every timestep from causal (past-only)
context, so evaluation must feed ONE timestep at a time and carry state
across the window (a Keras batch axis cannot stand in for the time axis
here). See docs/benchmark_report.md for the full account of why.

Example
-------
    python eyetracking_eval.py -d ./data/processed -l models/tinytemporalcnn_eyetracking.h5
"""
import argparse
import json
import os
import pathlib

import numpy as np

import akida
from cnn2snn import load_quantized_model

from eyetracking_data import get_data, to_signed_int8
from brainchip_utils.hardware_utils import get_akida_device

WINDOW = 64


def angular_error_deg(y_true, y_pred):
    return float(np.mean(np.linalg.norm(y_true - y_pred, axis=-1)))


def predict_streaming_keras(model, X_windows, batch_size=2048):
    """Feed a bufferized (BufferTempConv) Keras/quantized model one timestep
    at a time, resetting its FIFO state between batches of independent
    windows. Returns (N, window, 2) predictions."""
    from quantizeml.models import reset_buffers

    n = X_windows.shape[0]
    outputs = np.zeros((n, WINDOW, 2), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        reset_buffers(model)
        for t in range(WINDOW):
            frame = X_windows[start:end, t]
            outputs[start:end, t] = np.asarray(
                model.predict(frame, batch_size=end - start, verbose=0)).reshape(end - start, 2)
    return outputs


def evaluate_akida_model(akida_model, X_windows):
    """Run inference with an Akida model, one timestep at a time.

    Akida's buffered/FIFO layers only correctly maintain state for
    batch_size=1 (confirmed by direct test: sharing one FIFO across a
    batch of independent windows silently produces garbage), so each
    window is a fresh sequential pass. This is expensive (`window`
    sequential predict() calls per window), so callers should subsample
    the eval set -- see --num-eval below.
    """
    n = X_windows.shape[0]
    y_pred = np.zeros((n, WINDOW, 2), dtype=np.float32)
    layers = akida_model.layers
    for i in range(n):
        akida_model = type(akida_model)(layers)  # fresh FIFO state per window
        for t in range(WINDOW):
            frame = X_windows[i, t:t + 1]
            y_pred[i, t] = akida_model.predict(frame).reshape(2)
    return y_pred, akida_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras/quantized or .fbz akida model)')
    parser.add_argument('-d', '--data', default='./data/processed',
                        help='Path to the processed dataset directory (train/val/test _seq.npz)')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--scaler', default='./models/tinytemporalcnn_eyetracking.h5.yscaler.npz',
                        help='Target (y) normalization stats saved by eyetracking_train.py. '
                             'Reused across quantized/Akida stages -- normalization does not '
                             'change after the float training step.')
    parser.add_argument('--num-eval', type=int, default=None,
                        help='Subsample the eval set to this many windows. Per-window eval is '
                             'sequential (window sequential predict() calls each), so this '
                             'defaults to 4000 for quantized models and 300 for Akida models.')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Write error (and param count for the float .h5) to docs/metrics.json')
    args = parser.parse_args()

    scaler = np.load(args.scaler)
    y_mean, y_std = scaler['mean'], scaler['std']

    (X_train, y_train), (X_val, y_val) = get_data(args.data)
    split_data = {'train': (X_train, y_train), 'val': (X_val, y_val)}
    if args.split == 'test':
        d = np.load(os.path.join(args.data, 'test_seq.npz'))
        X_eval_all, y_eval_all = to_signed_int8(d['X']), d['y'].astype(np.float32)
    else:
        X_eval_all, y_eval_all = split_data[args.split]
        y_eval_all = y_eval_all[:, :, 0, 0, :]

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    stem = pathlib.Path(args.loadmodel).stem
    if args.loadmodel.endswith('.fbz'):
        model = akida.Model(args.loadmodel)
        isakida = True
        num_eval = args.num_eval or 300
    else:
        try:
            model = load_quantized_model(args.loadmodel)
        except Exception:
            import tf_keras as keras
            model = keras.models.load_model(args.loadmodel)
        isakida = False
        num_eval = args.num_eval or (4000 if 'i8_w8_a8' in stem else len(X_eval_all))

    rng = np.random.default_rng(0)
    num_eval = min(num_eval, len(X_eval_all))
    eval_idx = rng.choice(len(X_eval_all), size=num_eval, replace=False)
    X_eval, y_eval = X_eval_all[eval_idx], y_eval_all[eval_idx]

    # ---------------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------------
    if isakida:
        device = get_akida_device(target_version=model.ip_version)
        if device is not None:
            print('Akida hardware found -- note: this eval loop is sequential '
                  '(batch_size=1 per FIFO-state requirement) and does not use device mapping here.')
        y_pred, model = evaluate_akida_model(model, X_eval)
    else:
        y_pred = predict_streaming_keras(model, X_eval)

    y_pred = y_pred * y_std + y_mean
    overall_err = angular_error_deg(y_eval, y_pred)
    last_err = angular_error_deg(y_eval[:, -1], y_pred[:, -1])
    print(f'{args.split} (n={num_eval}): overall mean angular error={overall_err:.3f} deg, '
          f'last-timestep={last_err:.3f} deg')

    # ---------------------------------------------------------------------------
    # Persist metrics
    # ---------------------------------------------------------------------------
    if args.save_metrics:
        # Updates the stored metrics used to generate the README performance
        # table. For code maintenance only, run against the pretrained models.
        #   tinytemporalcnn_eyetracking.h5                -> float_*, params
        #   tinytemporalcnn_eyetracking_i8_w8_a8.h5       -> w8a8_quant_*
        #   tinytemporalcnn_eyetracking_i8_w8_a8.fbz      -> w8a8_akida_*, n_eval_windows
        metrics_path = pathlib.Path(__file__).parent / 'docs' / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}

        if 'i8_w8_a8' in stem:
            prefix = 'w8a8_akida' if isakida else 'w8a8_quant'
        else:
            prefix = 'float'
            metrics['params'] = f'{model.count_params():,}'

        metrics[f'{prefix}_overall_err_deg'] = f'{overall_err:.3f}°'
        metrics[f'{prefix}_last_err_deg'] = f'{last_err:.3f}°'
        if isakida:
            metrics['n_eval_windows'] = str(num_eval)
        metrics_path.write_text(json.dumps(metrics, indent=4) + '\n')
        print(f'Metrics saved to {metrics_path}')
