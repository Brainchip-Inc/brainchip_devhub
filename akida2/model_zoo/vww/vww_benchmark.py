#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
VWW hardware benchmark for Akida 2.

Runs a latency benchmark on an Akida VWW model, prints a summary, checks
per-layer activation sparsity, and generates summary plots.

Differences from the Akida 1 benchmark:
  * The reference Akida 2 hardware is an FPGA running at 25 MHz, so the
    measured clock frequency is 25e6 (not 400 MHz as on AKD1500).
  * Because the FPGA runs at a low clock, we also report a PROJECTED latency
    at a higher target clock. Cycle count is fixed for a given model + mapping
    regardless of clock speed, so projected_latency_ms = cycles / target_clock.
  * Power measurement is NOT performed. The FPGA power-measurement path is
    still under development, so this script is latency-only for now (the
    power columns are omitted from the metrics and README table).

Example
-------
    python vww_benchmark.py -l models/akidanet_vww_i8_w8_a8.fbz
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import akida
from akida_models.sparsity import compute_sparsity

from vww_data import get_samples
from brainchip_utils.hardware_utils import (get_mapping_stats, get_akida_device,
                                            per_layer_benchmark, full_model_benchmark)
from brainchip_utils.plot_utils import (plot_full_model_results, plot_per_layer_results,
                                        pretty_print_sparsity)

# Measured clock: Akida 2 reference hardware is an FPGA clocked at 25 MHz.
MEASURED_CLOCK = 25e6  # 25 MHz FPGA

# Projected clock: latency is also projected to a higher target clock to
# indicate expected performance on faster (e.g. ASIC) hardware. Cycle count is
# clock-independent, so the projection is exact given the target frequency.
# TODO: confirm the intended projection target clock for Akida 2 silicon.
#       100 MHz is a provisional placeholder only.
PROJECTED_CLOCK = 100e6  # provisional


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Hardware latency benchmark for an Akida 2 VWW model')
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.fbz akida model)')
    parser.add_argument('-d', '--data', default='./data/vw_coco2014_96',
                        help='VWW dataset root (contains train/ and val/ subdirs)')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Write benchmark values to metrics.json')
    args = parser.parse_args()

    NUM_SAMPLES = 1000

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    ak_model = akida.Model(args.loadmodel)
    imsize = tuple(ak_model.input_shape)

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------
    device = get_akida_device(target_version=ak_model.ip_version)
    if device is None:
        sys.exit('No compatible Akida hardware device found. Skipping benchmarking')

    # -------------------------------------------------------------------------
    # Sample
    # -------------------------------------------------------------------------
    # Processing in Akida is activity dependent (because it exploits sparsity)
    # and that activity is dependent on the input. That makes it imperative to
    # use real inputs when benchmarking Akida, rather than synthetic random
    # samples.
    samples = get_samples(args.data, imsize, num_samples=NUM_SAMPLES)

    # -------------------------------------------------------------------------
    # Full-model benchmark (latency only)
    # -------------------------------------------------------------------------
    # NOTE: power measurement is intentionally not requested here -- the FPGA
    # power path is still under development, so we report latency only. When a
    # power path exists, this is where full_model_benchmark's power fields would
    # be surfaced (as in the Akida 1 example).
    map_modes = ['Minimal', 'AllNps']
    full_results = dict()
    for mm in map_modes:
        map_mode = getattr(akida.MapMode, mm)
        print(f'\nRunning full-model benchmark (MapMode={mm})...')
        res = full_model_benchmark(ak_model, device, samples,
                                   map_mode=map_mode,
                                   clock_freq=MEASURED_CLOCK)

        # Projected latency at a higher target clock. Cycle count (mean_inf_clk)
        # is clock-independent, so this is an exact rescale, not an estimate of
        # host overhead.
        res['projected_clk_ms'] = res['mean_inf_clk'] / PROJECTED_CLOCK * 1000
        full_results[mm] = res

        # Re-map without hw_only to populate ak_model.sequences for stats
        ak_model.map(device, mode=map_mode)
        num_nps, num_passes, num_sequences = get_mapping_stats(ak_model)
        full_results[mm]['num_nps'] = num_nps
        full_results[mm]['num_passes'] = num_passes
        print(f'  Mapping: {num_nps} NP(s), {num_passes} pass(es), {num_sequences} sequence(s)')
        print(f'  Measured latency @ {MEASURED_CLOCK/1e6:.0f} MHz:  '
              f'{res["mean_clk_ms"]:.3f} ms')
        print(f'  Projected latency @ {PROJECTED_CLOCK/1e6:.0f} MHz: '
              f'{res["projected_clk_ms"]:.3f} ms')
        if num_sequences > 1:
            print('WARNING: note, model not completely mapped to hardware')

    # -------------------------------------------------------------------------
    # Per-layer benchmark. Minimal mapping mode, batch-size 1
    # -------------------------------------------------------------------------
    ak_model.map(device, mode=akida.MapMode.Minimal, hw_only=True)
    ak_model.summary()

    # Check sparsity per-layer
    sparsity_dict = compute_sparsity(ak_model, samples=samples)
    pretty_print_sparsity(sparsity_dict)

    print(f'Running per-layer benchmark ({NUM_SAMPLES} samples)...')
    per_layer_results = per_layer_benchmark(ak_model, device, samples,
                                            repeats=NUM_SAMPLES,
                                            clock_freq=MEASURED_CLOCK)

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    # Map without hw_only so ak_model.sequences is available for plot_mapping
    ak_model.map(device, mode=akida.MapMode.Minimal)
    perlayer_savepath = 'benchmark_results_layers.png'
    if args.save_metrics:
        perlayer_savepath = pathlib.Path(__file__).parent / 'docs' / ('ref_' + perlayer_savepath)
    plot_per_layer_results(per_layer_results, ak_model, sparsity_dict,
                           model_name=args.loadmodel,
                           savepath=perlayer_savepath)
    print('\nPer-layer results plot saved to ' + str(perlayer_savepath))

    full_savepath = 'benchmark_results_full.png'
    if args.save_metrics:
        full_savepath = pathlib.Path(__file__).parent / 'docs' / ('ref_' + full_savepath)
    plot_full_model_results(full_results, ak_model, device,
                            model_name=args.loadmodel,
                            savepath=full_savepath)
    print('Full model results plot saved to ' + str(full_savepath))

    if args.save_metrics:
        # Updates the stored metrics used to generate the README performance
        # tables. For code maintenance only, run against the pretrained models.
        #
        # Keys are variant-prefixed to match the multi-variant v2 README table.
        # The variant is inferred from the loaded .fbz filename (same scheme as
        # vww_eval.py). Power keys are intentionally absent: benchmarking is
        # latency-only until the FPGA power path exists.
        stem = pathlib.Path(args.loadmodel).stem
        if 'i8_w8_a8' in stem:
            variant = 'w8a8'
        elif 'i8_w4_a8_qat' in stem or ('i8_w4_a8' in stem and 'qat' in stem):
            variant = 'w4a8_qat'
        elif 'i8_w4_a8' in stem:
            variant = 'w4a8_ptq'
        else:
            variant = 'w8a8'  # fallback; benchmarking is only meaningful for a quantized .fbz

        metrics_path = pathlib.Path(__file__).parent / 'docs' / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        metrics[f'{variant}_sparsity'] = f'{np.mean(list(sparsity_dict.values())) * 100:.2f}%'
        for mm, res in full_results.items():
            mode = mm.lower()
            metrics[f'{variant}_{mode}_nps'] = str(res['num_nps'])
            metrics[f'{variant}_{mode}_passes'] = str(res['num_passes'])
            metrics[f'{variant}_{mode}_cycles'] = f'{res["mean_inf_clk"]:.0f}'
            metrics[f'{variant}_{mode}_latency_ms'] = f'{res["mean_clk_ms"]:.3f}'
            metrics[f'{variant}_{mode}_projected_ms'] = f'{res["projected_clk_ms"]:.3f}'
        metrics_path.write_text(json.dumps(metrics, indent=4) + '\n')
        print(f'Metrics saved to {metrics_path}')
