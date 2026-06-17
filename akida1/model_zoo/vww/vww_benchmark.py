#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
VWW per-layer hardware benchmark.

Runs a per-layer timing benchmark on an Akida VWW model, prints a summary
table, saves results as CSV, and generates a summary plot.

Example
-------
    python vww_benchmark.py -l akidanet_vww.fbz
"""
import argparse
import json
import pathlib
import time

import numpy as np
import akida
from akida_models.sparsity import compute_sparsity

from vww_data import get_samples
from brainchip_utils.hardware_utils import get_mapping_stats, get_akida_device, per_layer_benchmark, full_model_benchmark
from brainchip_utils.plot_utils import plot_full_model_results, plot_per_layer_results, pretty_print_sparsity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Per-layer hardware benchmark for an Akida VWW model')
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
    assert device is not None, 'No compatible Akida hardware device found.'
    # TODO: Add a check to get clock frequency specific to device
    CLOCK_FREQUENCY = 400e6 # 400 MHz for AKD1500
    
    # -------------------------------------------------------------------------
    # Sample
    # -------------------------------------------------------------------------
    # Processing in Akida is activity dependent (because it exploits sparsity)
    # and that activity is dependent on the input.
    # That makes it imperative to use real inputs when benchmarking Akida,
    # rather than synthetic random samples.
    samples = get_samples(args.data, imsize, num_samples=NUM_SAMPLES)

    # -------------------------------------------------------------------------
    # Benchmarks
    # -------------------------------------------------------------------------
    # Simple benchmarking
    # Run a simple benchmark at batch-size=1, Minimal mapping mode
    print('Running Simple Benchmark, Minimal map mode, batch-size=1')
    
    ak_model.map(device, mode=akida.MapMode.Minimal, hw_only=True)

    inf_clks = []
    inf_times = []
    for rr in range(NUM_SAMPLES):
        start_t = time.perf_counter_ns()
        ak_model.forward(samples[rr:rr + 1], batch_size=1)
        inf_times.append(time.perf_counter_ns() - start_t)
        # Get the number of on-device clock cycles for that inference
        inf_clks.append(ak_model.metrics['inference_clk'])
        
    mean_inf_clk = np.mean(inf_clks)/CLOCK_FREQUENCY*1e3 # s to ms
    mean_inf_time = np.mean(inf_times)*1e-6 # ns to ms
    # The timing reported by the device should be very close to that 
    # measured on the system
    print(f'\n  Mean inference time (system clock):    {mean_inf_time:.3f} ms  ')
    print(f'  Mean on-chip time (via chip clock cycles):      {mean_inf_clk:.3f} ms  ')

    # -------------------------------------------------------------------------
    # Full-model benchmark (latency + optional power)
    # -------------------------------------------------------------------------
    # Run full benchmarking including power measurement if available.
    # This test is fundamentally the same as the simple version shown above.
    # However, in order to run the power meausurements simultaneously, we need to use
    # multiprocessing. Also, the power measurement tools are complex and not of 
    # interest to most Akida users. For that reason we do not present that code here.
    # If interested, consult the details of the full_model_benchmark function.
    map_modes = ['Minimal', 'AllNps']
    POWER_REPEATS = 10
    full_results = dict()
    for mm in map_modes:
        map_mode = getattr(akida.MapMode, mm)
        print(f'\nRunning full-model benchmark (MapMode={mm}, {POWER_REPEATS} repeat(s))...')
        full_results[mm] = full_model_benchmark(ak_model, device, samples,
                                            map_mode=map_mode,
                                            repeats=POWER_REPEATS)
        # Re-map without hw_only to populate ak_model.sequences for stats
        ak_model.map(device, mode=map_mode)
        num_nps, num_passes, num_sequences = get_mapping_stats(ak_model)
        full_results[mm]['num_nps'] = num_nps
        full_results[mm]['num_passes'] = num_passes
        print(f'  Mapping: {num_nps} NP(s), {num_passes} pass(es), {num_sequences} sequence(s)')
        if num_sequences>1:
            print('WARNING: note, model not completely mapped to hardware')
    

    # -------------------------------------------------------------------------
    # Per-layer Benchmark. Minimal mapping mode, batch-size 1
    # -------------------------------------------------------------------------
    ak_model.map(device, mode=akida.MapMode.Minimal, hw_only=True)
    ak_model.summary()

    # Check sparsity per-layer
    sparsity_dict = compute_sparsity(ak_model, samples=samples)
    pretty_print_sparsity(sparsity_dict)

    print(f'Running per-layer benchmark ({NUM_SAMPLES} samples)...')
    per_layer_results = per_layer_benchmark(ak_model, device, samples, repeats=NUM_SAMPLES)

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    # Map without hw_only so ak_model.sequences is available for plot_mapping
    ak_model.map(device, mode=akida.MapMode.Minimal)
    perlayer_savepath = 'benchmark_results_layers.png'
    if args.save_metrics:
        perlayer_savepath = pathlib.Path(__file__).parent / 'docs' / ('ref_'+perlayer_savepath)
    plot_per_layer_results(per_layer_results, ak_model, sparsity_dict,
                           model_name=args.loadmodel,
                           savepath=perlayer_savepath)
    print('\nPer-layer results plot saved to ' + str(perlayer_savepath))

    full_savepath = 'benchmark_results_full.png'
    if args.save_metrics:
        full_savepath = pathlib.Path(__file__).parent / 'docs' / ('ref_'+full_savepath)
    plot_full_model_results(full_results, ak_model, device,
                            model_name=args.loadmodel,
                            savepath=full_savepath)
    print('Full model results plot saved to ' + str(full_savepath))


    if args.save_metrics:
        # The is used to update the stored metrics that are used to generate the
        # performance tables in the README of this folder. 
        # This should only be used for code maintenance, when the model or training
        # pipeline is updated and a new trained model integrated.
        metrics_path = pathlib.Path(__file__).parent / 'docs' / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        metrics['sparsity'] = f'{np.mean(list(sparsity_dict.values())) * 100:.2f}%'
        for mm, res in full_results.items():
            prefix = mm.lower()
            metrics[f'{prefix}_nps'] = str(res['num_nps'])
            metrics[f'{prefix}_passes'] = str(res['num_passes'])
            metrics[f'{prefix}_cycles'] = f'{res["mean_inf_clk"]:.0f}'
            metrics[f'{prefix}_latency_ms'] = f'{res["mean_clk_ms"]:.3f}'
            if res['power'] is not None:
                metrics[f'{prefix}_total_P'] = f'{res["power"]["avg_total_mw"]:.1f}'
                metrics[f'{prefix}_total_E'] = f'{res["power"]["avg_energy_mj"]:.3f}'
                metrics[f'{prefix}_dyn_P'] = f'{res["power"]["avg_dynamic_mw"]:.1f}'
                metrics[f'{prefix}_dyn_E'] = f'{res["power"]["avg_dynamic_energy_mj"]:.3f}'
        metrics_path.write_text(json.dumps(metrics, indent=4) + '\n')
        print(f'Metrics saved to {metrics_path}')
