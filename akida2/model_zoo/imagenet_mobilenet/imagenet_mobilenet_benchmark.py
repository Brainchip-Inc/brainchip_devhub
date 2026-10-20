#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Hardware benchmark for an ImageNet MobileNet on Akida 2.

Measures latency on the Akida 2 FPGA in three mapping modes, then breaks the
timing down per layer. Prints a summary, writes plots to ``docs/``, and can
record the numbers into ``docs/metrics.json``.

Requires a connected Akida 2 device. Without one the script exits early --
accuracy and activation sparsity do not need hardware and are measured by
``imagenet_mobilenet_eval.py`` instead.

Benchmark details:
  * The Akida 2 reference hardware is an FPGA running at 25 MHz, so the
    measured clock frequency is 25e6.
  * Because the FPGA runs at a low clock, we also report a PROJECTED latency
    at the AKD2500 target clock. Cycle count is fixed for a given model and
    mapping regardless of clock speed, so projected_latency_ms = cycles / target_clock.
  * Power is not measured: the FPGA has no power measurement path.
  * The FPGA has 6 nodes (24 NPs). The larger models need more NPs in a single
    pass than it has, so they don't map, and the script says so and exits.

By default the benchmark runs on the 10-image ImageNet-like sample pack (cycled
up to 100 inferences), so it needs no dataset setup. Akida is event-driven and
its latency depends on activation sparsity -- which depends on the input -- so
benchmarking on real images matters. Pass ``-d`` to draw the samples from the
ImageNet validation split instead, for a more representative spread.

Example
-------
    python imagenet_mobilenet_benchmark.py -v v1 -a 0.5
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import akida
from akida_models.sparsity import compute_sparsity

from imagenet_mobilenet_data import get_samples
from imagenet_mobilenet_model import VERSIONS, model_path, metrics_prefix
from brainchip_utils.hardware_utils import (get_mapping_stats, get_akida_device,
                                            per_layer_benchmark, full_model_benchmark,
                                            AKIDA_CLOCKS_HZ)
from brainchip_utils.plot_utils import (plot_full_model_results, plot_per_layer_results,
                                        pretty_print_sparsity)

# Measured clock: Akida 2 reference hardware is an FPGA clocked at 25 MHz.
MEASURED_CLOCK = AKIDA_CLOCKS_HZ['AKIDA2_FPGA']

# Projected clock: latency is also projected to the AKD2500 target silicon clock, to
# indicate expected performance on production hardware. Cycle count is clock-independent,
# so the projection is exact given the target frequency. AKD2500 production silicon does
# not exist yet, so this is BrainChip's current internal target (1 GHz), not a measured
# value -- it may still change before production. See brainchip_utils.hardware_utils.AKIDA_CLOCKS_HZ.
PROJECTED_CLOCK = AKIDA_CLOCKS_HZ['AKD2500']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Akida 2 hardware benchmark for an ImageNet MobileNet model')
    parser.add_argument('-v', '--version', default='v1', choices=VERSIONS,
                        help='MobileNet version. Defaults to %(default)s.')
    parser.add_argument('-a', '--alpha', type=float, default=1.0,
                        help='Width multiplier. Defaults to %(default)s.')
    parser.add_argument('-m', '--models-dir', default=None,
                        help='Directory holding the Akida models. '
                             'Defaults to pretrained_models/.')
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
    loadmodel = model_path(args.version, args.alpha, 'akida', args.models_dir)
    if not loadmodel.exists():
        raise FileNotFoundError(
            f'{loadmodel} not found. These models are tracked with Git LFS - '
            'run `git lfs pull` to fetch them.')
    print(f'Benchmarking {loadmodel.name}')
    model_stem = loadmodel.stem

    ak_model = akida.Model(str(loadmodel))
    imsize = tuple(ak_model.input_shape)

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------
    device = get_akida_device(target_version=ak_model.ip_version)
    if device is None:
        print("No compatible Akida hardware device found. Skipping benchmarking")
        sys.exit(0)

    # -------------------------------------------------------------------------
    # Sample
    # -------------------------------------------------------------------------
    # Processing in Akida is activity dependent (because it exploits sparsity)
    # and that activity is dependent on the input. That makes it imperative to
    # use real inputs when benchmarking Akida, rather than synthetic random
    # samples.
    samples = get_samples(imsize, num_samples=NUM_SAMPLES, data_path=args.data)

    # -------------------------------------------------------------------------
    # Full-model benchmark (latency only)
    # -------------------------------------------------------------------------
    map_modes = ['Minimal', 'AllNps', 'HwPr']
    full_results = dict()
    for mm in map_modes:
        map_mode = getattr(akida.MapMode, mm)
        print(f'\nRunning full-model benchmark (MapMode={mm})...')
        res = full_model_benchmark(ak_model, device, samples,
                                   map_mode=map_mode,
                                   clock_freq=MEASURED_CLOCK)
        if res is None:
            # full_model_benchmark returns None when the model does not map
            # fully onto the device in this mode.
            print(f'  MapMode={mm} did not map to hardware - skipping this mode.')
            continue

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

    if not full_results:
        sys.exit(f'{loadmodel.name} does not map onto {device.desc} in any mode: '
                 'it needs a larger Akida 2 device. Nothing to benchmark.')

    # -------------------------------------------------------------------------
    # Per-layer benchmark. Minimal mapping mode, batch-size 1
    # -------------------------------------------------------------------------
    ak_model.map(device, mode=akida.MapMode.Minimal, hw_only=True)
    ak_model.summary()

    # Check sparsity per-layer (for the plot only: the sparsity recorded in
    # metrics.json is measured over the validation set by imagenet_mobilenet_eval.py)
    sparsity_dict = compute_sparsity(ak_model, samples=samples)
    pretty_print_sparsity(sparsity_dict)

    print(f'Running per-layer benchmark ({NUM_SAMPLES} samples)...')
    per_layer_results = per_layer_benchmark(ak_model, device, samples,
                                            repeats=NUM_SAMPLES,
                                            clock_freq=MEASURED_CLOCK)

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    # Map without hw_only so ak_model.sequences is available for plot_mapping.
    # Plot filenames include the model stem so benchmarking different models
    # doesn't overwrite the previous run's plots.
    ak_model.map(device, mode=akida.MapMode.Minimal)
    perlayer_savepath = f'benchmark_results_layers_{model_stem}.png'
    if args.save_metrics:
        perlayer_savepath = pathlib.Path(__file__).parent / 'docs' / ('ref_' + perlayer_savepath)
    plot_per_layer_results(per_layer_results, ak_model, sparsity_dict,
                           model_name=loadmodel.name,
                           example='ImageNet MobileNet (Akida 2)',
                           savepath=perlayer_savepath)
    print('\nPer-layer results plot saved to ' + str(perlayer_savepath))

    full_savepath = f'benchmark_results_full_{model_stem}.png'
    if args.save_metrics:
        full_savepath = pathlib.Path(__file__).parent / 'docs' / ('ref_' + full_savepath)
    plot_full_model_results(full_results, ak_model, device,
                            model_name=loadmodel.name,
                            example='ImageNet MobileNet (Akida 2)',
                            savepath=full_savepath,
                            show_power=False)  # no power measurement on the FPGA
    print('Full model results plot saved to ' + str(full_savepath))

    if args.save_metrics:
        # Updates the stored metrics used to generate the README performance
        # tables. For code maintenance only, run against the pretrained models.
        # Sparsity is deliberately not written here (see above), and power keys
        # are absent: the FPGA benchmark is latency-only.
        metrics_path = pathlib.Path(__file__).parent / 'docs' / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        prefix = metrics_prefix(args.version, args.alpha)
        for mm, res in full_results.items():
            mode = mm.lower()
            metrics[f'{prefix}{mode}_nps'] = str(res['num_nps'])
            metrics[f'{prefix}{mode}_passes'] = str(res['num_passes'])
            metrics[f'{prefix}{mode}_cycles'] = f'{res["mean_inf_clk"]:.0f}'
            metrics[f'{prefix}{mode}_latency_ms'] = f'{res["mean_clk_ms"]:.3f}'
            metrics[f'{prefix}{mode}_projected_ms'] = f'{res["projected_clk_ms"]:.3f}'
        metrics_path.write_text(json.dumps(metrics, indent=4, sort_keys=True) + '\n')
        print(f'Metrics saved to {metrics_path}')
