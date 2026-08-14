#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Hardware benchmark for a published AkidaNet ImageNet model.

Measures latency and power on a physical AKD1500 device, in three mapping modes,
then breaks the timing down per layer. Prints a summary, writes plots to
``docs/``, and can record the numbers into ``docs/metrics.json``.

Requires a connected AKD1500. Without one the script exits early -- accuracy and
activation sparsity do not need hardware and are measured by
``imagenet_akidanet_eval.py`` instead.

By default the benchmark runs on the 10-image ImageNet-like sample pack (cycled
up to 100 inferences), so it needs no dataset setup. Akida is event-driven and
its latency and power depend on activation sparsity -- which depends on the
input -- so benchmarking on real images matters. Pass ``-d`` to draw the samples
from the ImageNet validation split instead, for a more representative spread.

Example
-------
    python imagenet_akidanet_benchmark.py -a 1.0 -i 224
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import akida
from akida_models.sparsity import compute_sparsity

from imagenet_akidanet_data import get_samples
from imagenet_akidanet_model import (ALPHAS, RESOLUTIONS, model_path,
                                     metrics_prefix)
from brainchip_utils.hardware_utils import (get_mapping_stats, get_akida_device,
                                            per_layer_benchmark, full_model_benchmark)
from brainchip_utils.plot_utils import (plot_full_model_results,
                                        plot_per_layer_results,
                                        pretty_print_sparsity)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='AKD1500 hardware benchmark for an AkidaNet ImageNet model')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, choices=ALPHAS,
                        help='Width multiplier. Defaults to %(default)s.')
    parser.add_argument('-i', '--input-resolution', type=int, default=224,
                        choices=RESOLUTIONS, dest='resolution',
                        help='Input resolution. Defaults to %(default)s.')
    parser.add_argument('-d', '--data', default=None,
                        help='Optional ImageNet directory; if omitted, the '
                             '10-image sample pack is used')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Write benchmark values to docs/metrics.json')
    args = parser.parse_args()

    NUM_SAMPLES = 100

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    loadmodel = model_path(args.alpha, args.resolution, 'akida')
    if not loadmodel.exists():
        raise FileNotFoundError(
            f'{loadmodel} not found. These weights are tracked with Git LFS - '
            'run `git lfs pull` to fetch them.')
    print(f'Benchmarking {loadmodel.name}  '
          f'(alpha={args.alpha}, {args.resolution}x{args.resolution})')

    ak_model = akida.Model(str(loadmodel))
    imsize = tuple(ak_model.input_shape)

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------
    device = get_akida_device(target_version=ak_model.ip_version)
    if device is None:
        sys.exit('No compatible Akida hardware device found. Skipping benchmarking')

    # TODO: Add a check to get clock frequency specific to device
    CLOCK_FREQUENCY = 400e6  # 400 MHz for AKD1500

    # -------------------------------------------------------------------------
    # Samples
    # -------------------------------------------------------------------------
    # Processing in Akida is activity dependent (because it exploits sparsity)
    # and that activity is dependent on the input.
    # That makes it imperative to use real inputs when benchmarking Akida,
    # rather than synthetic random samples.
    samples = get_samples(imsize, num_samples=NUM_SAMPLES, data_path=args.data)

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

    mean_inf_clk = np.mean(inf_clks) / CLOCK_FREQUENCY * 1e3  # s to ms
    mean_inf_time = np.mean(inf_times) * 1e-6  # ns to ms
    # The timing reported by the device should be very close to that
    # measured on the system
    print(f'\n  Mean inference time (system clock):    {mean_inf_time:.3f} ms  ')
    print(f'  Mean on-chip time (via chip clock cycles):      {mean_inf_clk:.3f} ms  ')

    # -------------------------------------------------------------------------
    # Full-model benchmark (latency + optional power)
    # -------------------------------------------------------------------------
    # Minimal uses the fewest NPs; AllNps spreads the model over every NP in one
    # hardware pass; HwPr also uses every NP but splits the work over more passes.
    map_modes = ['Minimal', 'AllNps', 'HwPr']
    POWER_REPEATS = 10
    full_results = dict()
    for mm in map_modes:
        map_mode = getattr(akida.MapMode, mm)
        print(f'\nRunning full-model benchmark (MapMode={mm}, {POWER_REPEATS} repeat(s))...')
        res = full_model_benchmark(ak_model, device, samples,
                                   map_mode=map_mode,
                                   repeats=POWER_REPEATS)
        if res is None:
            # Not every mode maps every model onto a single hardware sequence.
            # Drop the mode rather than losing the whole run.
            print(f'  MapMode={mm} did not map to hardware - skipping this mode.')
            continue
        full_results[mm] = res
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

    # Plots are namespaced per model, since six models share this docs/ folder
    tag = f'a{int(args.alpha * 100)}_{args.resolution}'

    perlayer_savepath = f'benchmark_results_layers_{tag}.png'
    if args.save_metrics:
        perlayer_savepath = pathlib.Path(__file__).parent / 'docs' / ('ref_' + perlayer_savepath)
    plot_per_layer_results(per_layer_results, ak_model, sparsity_dict,
                           model_name=str(loadmodel),
                           savepath=perlayer_savepath)
    print('\nPer-layer results plot saved to ' + str(perlayer_savepath))

    full_savepath = f'benchmark_results_full_{tag}.png'
    if args.save_metrics:
        full_savepath = pathlib.Path(__file__).parent / 'docs' / ('ref_' + full_savepath)
    plot_full_model_results(full_results, ak_model, device,
                            model_name=f'akidanet_imagenet alpha={args.alpha} {args.resolution}px',
                            savepath=full_savepath)
    print('Full model results plot saved to ' + str(full_savepath))

    if args.save_metrics:
        # Used to update the stored metrics behind the README performance tables.
        # This is a maintenance step, run when the models or pipeline change.
        metrics_path = pathlib.Path(__file__).parent / 'docs' / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        prefix = metrics_prefix(args.alpha, args.resolution)

        # Sparsity is deliberately not written here: imagenet_akidanet_eval.py
        # owns that metric, and measures it on the evaluation data rather than
        # on the handful of images this benchmark cycles through.
        for mm, res in full_results.items():
            mode = mm.lower()
            metrics[f'{prefix}{mode}_nps'] = str(res['num_nps'])
            metrics[f'{prefix}{mode}_passes'] = str(res['num_passes'])
            metrics[f'{prefix}{mode}_cycles'] = f'{res["mean_inf_clk"]:.0f}'
            metrics[f'{prefix}{mode}_latency_ms'] = f'{res["mean_clk_ms"]:.3f}'
            if res['power'] is not None:
                metrics[f'{prefix}{mode}_total_P'] = f'{res["power"]["avg_total_mw"]:.1f}'
                metrics[f'{prefix}{mode}_total_E'] = f'{res["power"]["avg_energy_mj"]:.3f}'
                metrics[f'{prefix}{mode}_dyn_P'] = f'{res["power"]["avg_dynamic_mw"]:.1f}'
                metrics[f'{prefix}{mode}_dyn_E'] = \
                    f'{res["power"]["avg_dynamic_energy_mj"]:.3f}'
        metrics_path.write_text(json.dumps(metrics, indent=4, sort_keys=True) + '\n')
        print(f'Metrics saved to {metrics_path}')
