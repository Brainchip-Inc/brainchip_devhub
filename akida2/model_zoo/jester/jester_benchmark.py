#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Hardware benchmark for the Jester TENN model on Akida 2.

Measures **per-frame** latency on the Akida 2 FPGA in three mapping modes, then
breaks the timing down per layer. Prints a summary, writes plots to ``docs/``,
and can record the numbers into ``docs/metrics.json``.

Requires a connected Akida 2 device. Without one the script exits early:
accuracy and activation sparsity don't need hardware and are measured by
``jester_eval.py`` instead.

Benchmark details:
  * The model is streaming: it takes one frame per inference and emits a
    prediction every frame, so the cost that matters is the latency of one
    frame. A clip's 16 frames are 16 inferences; there is no per-clip latency
    to add up.
  * The samples are consecutive frames of real clips, fed one per inference in
    stream order, so the temporal buffers hold real history, as they would on a
    live camera feed. Akida is event-driven and its latency depends on
    activation sparsity, which depends on the input, so random or shuffled
    frames would give the wrong numbers. The buffers are not reset between
    clips inside the timed loop: that is continuous streaming, and the cost of a
    reset is not measured.
  * The Akida 2 reference hardware is an FPGA running at 25 MHz. Latency is also
    projected to the AKD2500 target clock: the cycle count is fixed for a given
    model and mapping, so projected_latency_ms = cycles / target_clock.
  * Power is not measured: the FPGA has no power measurement path.
  * The FPGA has 6 nodes (24 NPs). The model maps in one hardware sequence
    (hw_only=True), over more than one pass.

Before timing, the script checks that the hardware streams a clip exactly as the
software backend does, buffers included.

Example
-------
    python jester_benchmark.py
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import akida
from akida_models.sparsity import compute_sparsity

from jester_data import get_data_dir, get_stream
from jester_model import model_path, verify_model
from jester_preprocessing import FRAMES_PER_CLIP
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

# Consecutive clips streamed by the benchmark: 10 clips of 16 frames
NUM_CLIPS = 10


def check_hw_matches_sw(ak_model, device, clip):
    """Streams one clip on hardware and on the software backend, from empty buffers.

    Returns:
        bool: True if every frame's output is bit-identical.
    """
    hw = akida.Model(layers=ak_model.layers)
    hw.map(device, mode=akida.MapMode.Minimal, hw_only=True)
    sw = akida.Model(layers=ak_model.layers)
    hw_out = np.concatenate([hw.forward(clip[t:t + 1]) for t in range(len(clip))])
    sw_out = np.concatenate([sw.forward(clip[t:t + 1]) for t in range(len(clip))])
    return np.array_equal(hw_out, sw_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Akida 2 hardware benchmark for the Jester TENN model')
    parser.add_argument('-m', '--models-dir', default=None,
                        help='Directory holding the Akida model. Defaults to pretrained_models/.')
    parser.add_argument('-d', '--data', default=None,
                        help='Jester directory. Defaults to the BrainChip Jester subset, '
                             'downloaded into data/.')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Write benchmark values to docs/metrics.json')
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    loadmodel = model_path('akida', args.models_dir)
    if args.models_dir is None:
        verify_model(loadmodel)
    print(f'Benchmarking {loadmodel.name}')
    model_stem = loadmodel.stem

    ak_model = akida.Model(str(loadmodel))

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
    # samples. For a streaming model the frames must also come in order.
    samples = get_stream(get_data_dir(args.data), NUM_CLIPS)
    num_frames = len(samples)
    print(f'Stream of {NUM_CLIPS} clips: {num_frames} frames {samples.shape[1:]}')

    match = check_hw_matches_sw(ak_model, device, samples[:FRAMES_PER_CLIP])
    print(f'Hardware streams a clip bit-identically to the software backend: {match}')
    if not match:
        sys.exit('Hardware and software outputs differ - not benchmarking.')

    # -------------------------------------------------------------------------
    # Full-model benchmark (latency only): one forward call per frame
    # -------------------------------------------------------------------------
    map_modes = ['Minimal', 'AllNps', 'HwPr']
    full_results = dict()
    for mm in map_modes:
        map_mode = getattr(akida.MapMode, mm)
        print(f'\nRunning full-model benchmark (MapMode={mm})...')
        res = full_model_benchmark(ak_model, device, samples,
                                   map_mode=map_mode,
                                   batch_size=1,
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
        print(f'  Measured latency per frame @ {MEASURED_CLOCK/1e6:.0f} MHz:  '
              f'{res["mean_clk_ms"]:.3f} ms')
        print(f'  Projected latency per frame @ {PROJECTED_CLOCK/1e6:.0f} MHz: '
              f'{res["projected_clk_ms"]:.3f} ms')

    if not full_results:
        sys.exit(f'{loadmodel.name} does not fit onto {device.desc} in a single '
                 'hardware sequence in any mode. Nothing to benchmark.')

    # -------------------------------------------------------------------------
    # Full-model plot and metrics
    # -------------------------------------------------------------------------
    # Map without hw_only so ak_model.sequences is available for plot_mapping.
    ak_model.map(device, mode=akida.MapMode.Minimal)
    full_savepath = f'benchmark_results_full_{model_stem}.png'
    if args.save_metrics:
        full_savepath = pathlib.Path(__file__).parent / 'docs' / ('ref_' + full_savepath)
    plot_full_model_results(full_results, ak_model, device,
                            model_name=loadmodel.name,
                            example='Jester TENN (Akida 2)',
                            savepath=full_savepath,
                            show_power=False)  # no power measurement on the FPGA
    print('Full model results plot saved to ' + str(full_savepath))

    if args.save_metrics:
        # Updates the stored metrics used to generate the README performance
        # tables. For code maintenance only, run against the pretrained model.
        # Sparsity is deliberately not written here (it is measured by
        # jester_eval.py), and power keys are absent: the FPGA benchmark is
        # latency-only.
        metrics_path = pathlib.Path(__file__).parent / 'docs' / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        for mm, res in full_results.items():
            mode = mm.lower()
            metrics[f'w8a8_{mode}_nps'] = str(res['num_nps'])
            metrics[f'w8a8_{mode}_passes'] = str(res['num_passes'])
            metrics[f'w8a8_{mode}_cycles'] = f'{res["mean_inf_clk"]:.0f}'
            metrics[f'w8a8_{mode}_latency_ms'] = f'{res["mean_clk_ms"]:.3f}'
            metrics[f'w8a8_{mode}_projected_ms'] = f'{res["projected_clk_ms"]:.3f}'
        metrics_path.write_text(json.dumps(metrics, indent=4, sort_keys=True) + '\n')
        print(f'Metrics saved to {metrics_path}')

    # -------------------------------------------------------------------------
    # Per-layer benchmark. Minimal mapping mode, one frame per inference
    # -------------------------------------------------------------------------
    ak_model.map(device, mode=akida.MapMode.Minimal, hw_only=True)
    ak_model.summary()

    # Check sparsity per-layer (for the plot only, see above). compute_sparsity
    # streams the frames in order from empty buffers.
    sparsity_dict = compute_sparsity(ak_model, samples=samples, batch_size=num_frames)
    pretty_print_sparsity(sparsity_dict)

    # Each sub-model is timed on the frames in stream order, so its buffers fill
    # as they would in the full model.
    print(f'Running per-layer benchmark ({num_frames} frames)...')
    per_layer_results = per_layer_benchmark(ak_model, device, samples,
                                            repeats=num_frames,
                                            clock_freq=MEASURED_CLOCK)

    ak_model.map(device, mode=akida.MapMode.Minimal)
    perlayer_savepath = f'benchmark_results_layers_{model_stem}.png'
    if args.save_metrics:
        perlayer_savepath = pathlib.Path(__file__).parent / 'docs' / ('ref_' + perlayer_savepath)
    plot_per_layer_results(per_layer_results, ak_model, sparsity_dict,
                           model_name=loadmodel.name,
                           example='Jester TENN (Akida 2)',
                           savepath=perlayer_savepath)
    print('\nPer-layer results plot saved to ' + str(perlayer_savepath))
