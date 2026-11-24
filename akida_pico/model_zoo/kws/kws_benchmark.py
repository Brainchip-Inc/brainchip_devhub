#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Latency benchmark for the Pico KWS model on the Akida Pico FPGA.

The model maps to a single hardware sequence on one ``TNP_R``, so there is one mapping to
report (no Minimal/AllNPs comparison, no per-layer breakdown). One forward pass consumes a
256-sample chunk, so the real-time budget is one chunk every 16 ms (62.5 chunks/s).

Latency comes from the hardware clock-cycle counter (``model.metrics['inference_clk']``):
cycles per chunk divided by the clock. The cycle count is fixed for a given model and
mapping, so the same count gives the latency at any other clock.

Power is NOT measured here. The Pico FPGA has no power measurement path, so this script is
latency-only.

Example
-------
    python kws_benchmark.py -l models/tenn_recurrent_sc12_stateful_i8_w8_a8.fbz
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import akida

from kws_data import CHUNK_SIZE, SAMPLE_RATE, load_split

# Clock of the Pico FPGA reference platform.
FPGA_CLOCK_HZ = 50e6
# The period between chunks in real-time streaming.
CHUNK_PERIOD_MS = CHUNK_SIZE / SAMPLE_RATE * 1000

NUM_FRAMES = 1000


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Latency benchmark for the Pico KWS model')
    parser.add_argument('-l', '--loadmodel', required=True, help='Akida model to load (.fbz)')
    parser.add_argument('-d', '--data', default=None,
                        help='TFDS data directory (default: ~/tensorflow_datasets)')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Write the benchmark values to docs/metrics.json')
    args = parser.parse_args()

    model = akida.Model(args.loadmodel)

    devices = akida.devices()
    if not devices:
        print('No Akida Pico device found. Skipping benchmarking')
        sys.exit(0)
    model.map(devices[0], hw_only=True)
    model.summary()

    # Processing is activity dependent, so use real audio: consecutive chunks of test clips
    # (a mix of keywords, silence and unknown) rather than random samples.
    X, _ = load_split('test', data_path=args.data)
    chunks = X.reshape(-1, CHUNK_SIZE, 1)[:NUM_FRAMES].astype(np.int16)
    frames = chunks[:, np.newaxis, :, :]                 # (NUM_FRAMES, 1, 256, 1)

    model.forward(frames)                                # warm-up
    model.forward(frames)
    cycles = model.metrics['inference_clk'] / len(frames)
    latency_ms = cycles / FPGA_CLOCK_HZ * 1000
    fps = FPGA_CLOCK_HZ / cycles

    print(f'Cycles per chunk:   {cycles:.0f}')
    print(f'Latency @ {FPGA_CLOCK_HZ / 1e6:.0f} MHz:   {latency_ms:.3f} ms '
          f'({latency_ms / CHUNK_PERIOD_MS * 100:.1f}% of the {CHUNK_PERIOD_MS:.0f} ms chunk period)')
    print(f'Throughput:         {fps:.1f} chunks/s (real time needs {1000 / CHUNK_PERIOD_MS:.1f})')

    if args.save_metrics:
        # Updates the stored metrics used to generate the README. For code maintenance
        # only, run against the pretrained model's quantized variant.
        metrics_path = pathlib.Path(__file__).parent / 'docs' / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        metrics['cycles'] = f'{cycles:.0f}'
        metrics['latency_ms'] = f'{latency_ms:.3f}'
        metrics['fps'] = f'{fps:.1f}'
        metrics['chunk_period_ms'] = f'{CHUNK_PERIOD_MS:.0f}'
        metrics['hw_clock_mhz'] = f'{FPGA_CLOCK_HZ / 1e6:.0f}'
        metrics_path.write_text(json.dumps(metrics, indent=4, sort_keys=True) + '\n')
        print(f'Metrics saved to {metrics_path}')
