#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Segmentation hardware benchmark for Akida 2.

Runs a latency benchmark on an Akida segmentation model, prints a summary,
checks per-layer activation sparsity, and generates summary plots. Same
pattern as vww_benchmark.py -- unlike the eyetracking example, this model is
a plain feedforward network (no causal/streaming state), so the standard
batched `full_model_benchmark`/`per_layer_benchmark` helpers apply directly,
one 384x384 tile per inference.

Differences from the Akida 1 benchmark style (see vww_benchmark.py for the
Akida 1 version): the reference Akida 2 hardware is an FPGA running at 25
MHz, so latency is also projected to the AKD2500 production target clock
(cycle count is fixed for a given model + mapping regardless of clock
speed). Power measurement is NOT performed -- the FPGA power-measurement
path is still under development project-wide.

Example
-------
    python segmentation_benchmark.py -l pretrained_models/unetv4akida_segmentation_i8_w8_a8.fbz
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import akida
from akida_models.sparsity import compute_sparsity

from segmentation_data import get_samples
from brainchip_utils.hardware_utils import (get_mapping_stats, get_akida_device,
                                             per_layer_benchmark, full_model_benchmark,
                                             AKIDA_CLOCKS_HZ)
from brainchip_utils.plot_utils import (plot_full_model_results, plot_per_layer_results,
                                         pretty_print_sparsity)

MEASURED_CLOCK = AKIDA_CLOCKS_HZ['AKIDA2_FPGA']
PROJECTED_CLOCK = AKIDA_CLOCKS_HZ['AKD2500']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Hardware latency benchmark for an Akida 2 segmentation model')
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.fbz akida model)')
    parser.add_argument('-d', '--data', default='./data/cityscapes',
                        help='Cityscapes root (contains leftImg8bit/ and gtFine/)')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Write benchmark values to docs/metrics.json')
    args = parser.parse_args()

    NUM_SAMPLES = 100

    ak_model = akida.Model(args.loadmodel)
    imsize = tuple(ak_model.input_shape)

    device = get_akida_device(target_version=ak_model.ip_version)
    if device is None:
        print('No compatible Akida hardware device found. Skipping benchmarking')
        sys.exit(0)

    samples = get_samples(args.data, input_shape=imsize, num_samples=NUM_SAMPLES)

    map_modes = ['Minimal', 'AllNps', 'HwPr']
    full_results = {}
    for mm in map_modes:
        map_mode = getattr(akida.MapMode, mm)
        print(f'\nRunning full-model benchmark (MapMode={mm})...')
        res = full_model_benchmark(ak_model, device, samples, map_mode=map_mode, clock_freq=MEASURED_CLOCK)
        res['projected_clk_ms'] = res['mean_inf_clk'] / PROJECTED_CLOCK * 1000
        full_results[mm] = res

        ak_model.map(device, mode=map_mode)
        num_nps, num_passes, num_sequences = get_mapping_stats(ak_model)
        full_results[mm]['num_nps'] = num_nps
        full_results[mm]['num_passes'] = num_passes
        print(f'  Mapping: {num_nps} NP(s), {num_passes} pass(es), {num_sequences} sequence(s)')
        print(f'  Measured latency @ {MEASURED_CLOCK / 1e6:.0f} MHz:  {res["mean_clk_ms"]:.3f} ms')
        print(f'  Projected latency @ {PROJECTED_CLOCK / 1e6:.0f} MHz: {res["projected_clk_ms"]:.3f} ms')
        if num_sequences > 1:
            print('WARNING: note, model not completely mapped to hardware')

    ak_model.map(device, mode=akida.MapMode.Minimal, hw_only=True)
    ak_model.summary()

    sparsity_dict = compute_sparsity(ak_model, samples=samples)
    pretty_print_sparsity(sparsity_dict)

    print(f'Running per-layer benchmark ({NUM_SAMPLES} samples)...')
    per_layer_results = per_layer_benchmark(ak_model, device, samples, repeats=NUM_SAMPLES,
                                             clock_freq=MEASURED_CLOCK)

    ak_model.map(device, mode=akida.MapMode.Minimal)
    perlayer_savepath = 'benchmark_results_layers.png'
    if args.save_metrics:
        perlayer_savepath = pathlib.Path(__file__).parent / 'docs' / ('ref_' + perlayer_savepath)
    plot_per_layer_results(per_layer_results, ak_model, sparsity_dict,
                            model_name=args.loadmodel, savepath=perlayer_savepath)
    print('\nPer-layer results plot saved to ' + str(perlayer_savepath))

    full_savepath = 'benchmark_results_full.png'
    if args.save_metrics:
        full_savepath = pathlib.Path(__file__).parent / 'docs' / ('ref_' + full_savepath)
    plot_full_model_results(full_results, ak_model, device, model_name=args.loadmodel, savepath=full_savepath)
    print('Full model results plot saved to ' + str(full_savepath))

    if args.save_metrics:
        metrics_path = pathlib.Path(__file__).parent / 'docs' / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        metrics['w8a8_sparsity'] = f'{np.mean(list(sparsity_dict.values())) * 100:.2f}%'
        for mm, res in full_results.items():
            mode = mm.lower()
            metrics[f'w8a8_{mode}_nps'] = str(res['num_nps'])
            metrics[f'w8a8_{mode}_passes'] = str(res['num_passes'])
            metrics[f'w8a8_{mode}_cycles'] = f'{res["mean_inf_clk"]:.0f}'
            metrics[f'w8a8_{mode}_latency_ms'] = f'{res["mean_clk_ms"]:.3f}'
            metrics[f'w8a8_{mode}_projected_ms'] = f'{res["projected_clk_ms"]:.3f}'
        metrics_path.write_text(json.dumps(metrics, indent=4) + '\n')
        print(f'Metrics saved to {metrics_path}')
