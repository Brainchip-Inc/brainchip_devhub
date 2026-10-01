#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Eyetracking hardware benchmark for Akida 2.

Runs a latency benchmark on an Akida eyetracking model and checks per-layer
activation sparsity. Differs from the classification examples in this zoo:

  * The model is a streaming/bufferized (BufferTempConv) network, so it must
    be benchmarked ONE TIMESTEP AT A TIME with batch_size=1 (Akida's FIFO
    layers only maintain correct state that way -- see eyetracking_eval.py).
    `full_model_benchmark`/`per_layer_benchmark` are written for ordinary
    batched models, so here we time a manual single-frame inference loop
    instead of calling them directly.
  * The reference Akida 2 hardware is an FPGA running at 25 MHz, so latency
    is measured at that clock and also projected to the AKD2500 production
    target clock (cycle count is fixed for a given model + mapping
    regardless of clock speed, so projected_latency = cycles / target_clock
    -- except cycle-accurate counts aren't exposed for this manual timing
    loop, so only wall-clock latency at the measured clock is reported;
    see the TODO below).
  * Power measurement is NOT performed -- the FPGA power-measurement path is
    still under development project-wide (see akida2/model_zoo/vww).

Example
-------
    python eyetracking_benchmark.py -l models/tinytemporalcnn_eyetracking_i8_w8_a8.fbz
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import akida
from akida_models.sparsity import compute_sparsity

from eyetracking_data import get_samples
from brainchip_utils.hardware_utils import get_akida_device, AKIDA_CLOCKS_HZ
from brainchip_utils.plot_utils import pretty_print_sparsity

MEASURED_CLOCK = AKIDA_CLOCKS_HZ['AKIDA2_FPGA']
PROJECTED_CLOCK = AKIDA_CLOCKS_HZ['AKD2500']

WINDOW = 64


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Hardware latency benchmark for an Akida 2 eyetracking model')
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.fbz akida model)')
    parser.add_argument('-d', '--data', default='./data/processed',
                        help='Path to the processed dataset directory')
    parser.add_argument('-n', '--num-windows', type=int, default=100,
                        help='Number of windows to time (each is WINDOW sequential frames)')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Write benchmark values to docs/metrics.json')
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    ak_model = akida.Model(args.loadmodel)

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------
    device = get_akida_device(target_version=ak_model.ip_version)
    if device is None:
        print('No compatible Akida hardware device found. Skipping benchmarking')
        sys.exit(0)
    ak_model.map(device, mode=akida.MapMode.Minimal)
    ak_model.summary()

    # -------------------------------------------------------------------------
    # Samples (real calibration frames -- Akida processing is activity
    # dependent, so synthetic random samples would not reflect real latency)
    # -------------------------------------------------------------------------
    samples = get_samples(args.data, num_samples=args.num_windows * WINDOW)
    samples = samples.reshape(args.num_windows, WINDOW, 1, 1, -1)

    # -------------------------------------------------------------------------
    # Sparsity (single-frame forward passes, same shape the device sees)
    # -------------------------------------------------------------------------
    sparsity_dict = compute_sparsity(ak_model, samples=samples[:, 0])
    pretty_print_sparsity(sparsity_dict)

    # -------------------------------------------------------------------------
    # Sequential (batch_size=1) latency timing
    # -------------------------------------------------------------------------
    # TODO: replace this wall-clock loop with a cycle-accurate measurement
    # (ak_model.statistics / per_layer_benchmark) once those utilities support
    # a single-frame-at-a-time streaming mapping -- today they assume a
    # batched, stateless model, which this BufferTempConv model is not.
    layers = ak_model.layers
    per_window_s = []
    for i in range(len(samples)):
        ak_model = type(ak_model)(layers)  # fresh FIFO state per window
        t0 = time.perf_counter()
        for t in range(WINDOW):
            ak_model.predict(samples[i, t:t + 1])
        per_window_s.append(time.perf_counter() - t0)

    mean_window_ms = float(np.mean(per_window_s)) * 1000
    mean_frame_us = mean_window_ms * 1000 / WINDOW
    print(f'Measured @ {MEASURED_CLOCK/1e6:.0f} MHz FPGA: '
          f'{mean_window_ms:.3f} ms/window ({mean_frame_us:.1f} us/frame), '
          f'n={len(samples)} windows')

    if args.save_metrics:
        metrics_path = pathlib.Path(__file__).parent / 'docs' / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        metrics['w8a8_sparsity'] = f'{np.mean(list(sparsity_dict.values())) * 100:.2f}%'
        metrics['w8a8_latency_ms_per_window'] = f'{mean_window_ms:.3f}'
        metrics['w8a8_latency_us_per_frame'] = f'{mean_frame_us:.1f}'
        metrics['w8a8_n_bench_windows'] = str(len(samples))
        metrics_path.write_text(json.dumps(metrics, indent=4) + '\n')
        print(f'Metrics saved to {metrics_path}')
