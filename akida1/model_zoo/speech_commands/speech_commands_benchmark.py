#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Speech Commands per-layer hardware benchmark.

Runs a per-layer timing benchmark on an Akida Speech Commands model, prints a summary
table, saves results as CSV, and generates a summary plot.

Example
-------
    python speech_commands_benchmark.py -l model_qat.fbz
"""
import argparse
import json
import multiprocessing as mp
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import akida
from akida_models.sparsity import compute_sparsity

from speech_commands_data_loader import compute_mfcc_range, get_samples
from brainchip_utils.hardware_utils import (
    get_mapping_stats, get_akida_device, per_layer_benchmark,
    _check_power_available, _power_process_fn,
    _AKD1500_I2C_URL, _AKD1500_INA_CONFIGS, _PYFTDI_AVAILABLE,
)
from brainchip_utils.plot_utils import (
    plot_per_layer_results, pretty_print_sparsity, plot_mapping,
)


def continuous_burst_benchmark(ak_model, device, samples, map_mode,
                               n_forward_calls=2000, clock_freq=400e6,
                               floor_duration=1.0,
                               i2c_url=_AKD1500_I2C_URL,
                               ina_configs=None):
    
    if ina_configs is None:
        ina_configs = _AKD1500_INA_CONFIGS

    measure_power = False
    if device.ip_version != akida.IpVersion.v1:
        print('Power measurement not available: only supported on AKD1500 (IpVersion.v1).')
    elif not _PYFTDI_AVAILABLE:
        print('Power measurement not available: pyftdi not installed.')
    else:
        print('Checking I2C power measurement hardware...')
        if _check_power_available(i2c_url, ina_configs):
            measure_power = True
            print('  I2C INA sensor detected -- power measurement enabled.')
        else:
            print('  I2C INA sensor not found -- running timing-only benchmark.')

    try:
        ak_model.map(device, mode=map_mode, hw_only=True)
    except Exception:
        print('This model does not map fully to hardware. Quitting benchmarking...')
        return None

    ak_model.forward(samples[0:1])  # priming pass
    n = len(samples)

    if measure_power:
        ctx = mp.get_context('fork')
        stop_event = ctx.Event()
        ready_event = ctx.Event()
        power_queue = ctx.Queue()
        power_proc = ctx.Process(
            target=_power_process_fn,
            args=(stop_event, ready_event, power_queue, i2c_url, ina_configs),
            daemon=True,
        )
        power_proc.start()
        ready_event.wait()
        print(f'  Power measurement active. Recording pre-burst floor ({floor_duration:.1f} s)...')
        time.sleep(floor_duration)

    # burst benchmark
    inf_clks = []
    burst_start = time.perf_counter()
    for ii in range(n_forward_calls):
        ak_model.forward(samples[ii % n:ii % n + 1])
        inf_clks.append(ak_model.metrics['inference_clk'])
    burst_end = time.perf_counter()

    power_results = None
    if measure_power:
        print(f'  Recording post-burst floor ({floor_duration:.1f} s)...')
        time.sleep(floor_duration)

        stop_event.set()
        readings = []
        while power_proc.is_alive():
            try:
                readings.append(power_queue.get(timeout=0.05))
            except Exception:
                pass
        power_proc.join()
        while not power_queue.empty():
            try:
                readings.append(power_queue.get_nowait())
            except Exception:
                break

        readings_arr = np.array(readings) if readings else np.empty((0, 3))
        abs_times = readings_arr[:, 0] if len(readings_arr) else np.array([])
        powers = readings_arr[:, 1] * readings_arr[:, 2] if len(readings_arr) else np.array([])

        floor_mask = (abs_times < burst_start) | (abs_times > burst_end)
        busy_mask = (abs_times >= burst_start) & (abs_times <= burst_end)

        burst_duration_s = burst_end - burst_start
        avg_floor = float(np.mean(powers[floor_mask])) if np.any(floor_mask) else float('nan')
        avg_total = float(np.mean(powers[busy_mask])) if np.any(busy_mask) else float('nan')
        avg_dyn = avg_total - avg_floor
        power_results = {
            'readings':              readings,
            'burst_start':           burst_start,
            'burst_end':             burst_end,
            'avg_floor_mw':          avg_floor,
            'avg_total_mw':          avg_total,
            'avg_dynamic_mw':        avg_dyn,
            'avg_energy_mj':         avg_total * burst_duration_s / n_forward_calls,
            'avg_dynamic_energy_mj': avg_dyn * burst_duration_s / n_forward_calls,
        }
        print(f'  Avg floor power:    {avg_floor:.1f} mW')
        print(f'  Avg dynamic power:  {avg_dyn:.1f} mW')
        print(f'  Avg dynamic energy: {power_results["avg_dynamic_energy_mj"]:.4f} mJ/inf')

    inf_clks = np.array(inf_clks)
    mean_inf_clk = float(np.mean(inf_clks))
    std_inf_clk = float(np.std(inf_clks))
    mean_clk_ms = mean_inf_clk / clock_freq * 1000
    mean_inf_ms = (burst_end - burst_start) / n_forward_calls * 1000

    print(f'\n  Mean inference time:    {mean_inf_ms:.3f} ms')
    print(f'  Mean on-chip time:      {mean_clk_ms:.3f} ms  ({mean_inf_clk:.0f} clocks)')
    print(f'  Total inferences run:   {n_forward_calls}')

    return {
        'map_mode':      map_mode,
        'mean_inf_clk':  mean_inf_clk,
        'std_inf_clk':   std_inf_clk,
        'mean_clk_ms':   mean_clk_ms,
        'mean_inf_ms':   mean_inf_ms,
        'n_inferences':  n_forward_calls,
        'power':         power_results,
    }


def plot_continuous_power_trace(power_data, ax):
    readings_arr = np.array(power_data['readings'])
    abs_times = readings_arr[:, 0]
    powers = readings_arr[:, 1] * readings_arr[:, 2]  # V x mA = mW

    burst_start = power_data['burst_start']
    burst_end = power_data['burst_end']
    avg_floor = power_data['avg_floor_mw']
    avg_dyn = power_data['avg_dynamic_mw']
    avg_dyn_e = power_data['avg_dynamic_energy_mj']

    t_rel = abs_times - burst_start
    ax.plot(t_rel, powers, lw=0.8, color='steelblue', alpha=0.8)

    ax.axvline(0, color='seagreen', ls='--', lw=1.2, label='Burst start')
    ax.axvline(burst_end - burst_start, color='firebrick', ls='--', lw=1.2, label='Burst end')
    ax.axhline(avg_floor, color='gray', ls=':', lw=1.0, label=f'Floor: {avg_floor:.1f} mW')
    ax.set_ylabel('Power (mW)')
    ax.set_xlabel('Time relative to burst start (s)')
    ax.set_title('Power timeline')
    ax.legend(fontsize=8, loc='lower center')

    stats_text = (
        f'Avg dynamic power:  {avg_dyn:.1f} mW\n'
        f'Avg dynamic energy: {avg_dyn_e:.4f} mJ/inf'
    )
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            va='top', ha='right', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))


def plot_continuous_full_model_results(full_results, ak_model, device, model_name=None, savepath=None):
    modes = list(full_results.keys())
    ncols = len(modes)
    fig, axs = plt.subplots(2, ncols, figsize=(7 * ncols, 8), constrained_layout=True, squeeze=False)

    if model_name is not None:
        fig.suptitle(model_name, fontsize=14, fontweight='bold')

    for i, mode_name in enumerate(modes):
        result = full_results[mode_name]
        ak_model.map(device, mode=getattr(akida.MapMode, mode_name))

        ax_pwr = axs[0, i]
        ax_map = axs[1, i]

        power_data = result.get('power')
        if power_data is not None:
            plot_continuous_power_trace(power_data, ax_pwr)
        else:
            ax_pwr.set_facecolor('#f5f5f5')
            ax_pwr.text(0.5, 0.5, 'Power measurements\n(not yet available)',
                        ha='center', va='center', transform=ax_pwr.transAxes,
                        fontsize=11, color='#999999', style='italic')
            ax_pwr.set_title('Power', color='#999999')
        ax_pwr.set_title(f'MapMode: {mode_name}', loc='left', fontsize=12, color='#444444')

        plot_mapping(ak_model, ax_map)
        ax_map.set_title(f'MapMode: {mode_name}', loc='left', fontsize=12, color='#444444')

    if savepath is not None:
        fig.savefig(savepath, dpi=150)
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Per-layer hardware benchmark for an Akida Speech Commands model')
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.fbz akida model)')
    parser.add_argument('-d', '--data', default='./data/sc10',
                        help='Speech Commands tfds data directory')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Write benchmark values to metrics.json')
    args = parser.parse_args()

    NUM_SAMPLES = 100


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
    data_transform = compute_mfcc_range(data_dir=args.data)
    samples = get_samples(args.data, data_transform=data_transform, num_samples=NUM_SAMPLES)
    

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
    map_modes = ['Minimal', 'AllNps']
    N_FORWARD_CALLS = 20000
    full_results = dict()
    for mm in map_modes:
        map_mode = getattr(akida.MapMode, mm)
        print(f'\nRunning full-model benchmark (MapMode={mm}, '
              f'{N_FORWARD_CALLS}-call continuous burst)...')
        full_results[mm] = continuous_burst_benchmark(ak_model, device, samples,
                                                      map_mode=map_mode,
                                                      n_forward_calls=N_FORWARD_CALLS)
        # Re-map without hw_only to populate ak_model.sequences for stats
        ak_model.map(device, mode=map_mode)
        num_nps, num_passes, num_sequences = get_mapping_stats(ak_model)
        full_results[mm]['num_nps'] = num_nps
        full_results[mm]['num_passes'] = num_passes
        print(f'  Mapping: {num_nps} NP(s), {num_passes} pass(es), {num_sequences} sequence(s)')
        if num_sequences > 1:
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
    plot_continuous_full_model_results(full_results, ak_model, device,
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
