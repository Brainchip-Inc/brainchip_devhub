"""Benchmark results files: save what the benchmark plots need, and redraw them from it.

A hardware benchmark takes from minutes to hours, but its figures only need the
numbers it measured. save_benchmark_results() writes those numbers to a JSON
file next to the figures: full-model timing and power per map mode, the
hardware mapping of each mode, per-layer timing and per-layer sparsity. The
figures can then be redrawn from the file, with no device, model or akida
install, for instance after a change to how they look:

    python -m brainchip_utils.benchmark_results akida1/model_zoo/vww/docs/ref_benchmark_results.json

Power is stored as plot_utils.power_trace() resamples it for plotting (one trace
per repeat on a shared time grid), not as raw sensor readings.
"""
import argparse
import json
import pathlib

import numpy as np

from brainchip_utils.plot_utils import (figure_title, plot_full_model_results,
                                        plot_per_layer_results, power_trace)

FORMAT_VERSION = 1

# Full-model result keys stored as they are (all scalars)
_FULL_MODEL_KEYS = ('mean_inf_clk', 'std_inf_clk', 'mean_clk_ms', 'mean_inf_ms', 'std_inf_ms',
                    'n_inferences', 'num_nps', 'num_passes', 'num_sequences', 'projected_clk_ms')
_POWER_KEYS = ('avg_floor_mw', 'avg_total_mw', 'avg_dynamic_mw', 'avg_energy_mj',
               'avg_dynamic_energy_mj')


def _round(values, decimals):
    """Round a number or nested list of numbers, to keep the file compact."""
    return np.round(np.asarray(values, dtype=float), decimals).tolist()


def _ints(values):
    """Whole-number arrays (clock counts, nanoseconds) as nested lists of ints."""
    return np.rint(np.asarray(values, dtype=float)).astype(np.int64).tolist()


def _full_model_entry(result):
    entry = {k: result[k] for k in _FULL_MODEL_KEYS if k in result}
    entry['mapping'] = result['mapping']
    power = result.get('power')
    if power is not None:
        trace = power_trace(power)
        if trace is not None:
            trace = {'t_s': _round(trace['t_s'], 5),
                     'repeats_mw': _round(trace['repeats_mw'], 2),
                     'inf_duration_s': _round(trace['inf_duration_s'], 6),
                     'inferences_per_repeat': trace['inferences_per_repeat']}
        power = {**{k: power[k] for k in _POWER_KEYS}, 'trace': trace}
    entry['power'] = power
    return entry


def _per_layer_entry(per_layer_results, mapping, sparsity_dict):
    return {
        'layer_names':       list(per_layer_results['layer_names']),
        'per_layer_clocks':  np.asarray(per_layer_results['per_layer_clocks']).tolist(),
        'per_layer_ms':      np.asarray(per_layer_results['per_layer_ms']).tolist(),
        # Raw (num_layers, repeats) arrays, behind the per-layer spread
        'cumulative_clocks': _ints(per_layer_results['cumulative_clocks']),
        'cumulative_times':  _ints(per_layer_results['cumulative_times']),
        'mapping':           mapping,
        'sparsity':          {name: float(s) for name, s in sparsity_dict.items()},
    }


def save_benchmark_results(path, full_results, per_layer_results, per_layer_mapping,
                           sparsity_dict, model_name, example, figures, show_power=True,
                           **info):
    """Write a benchmark's results to a JSON file, and return them as loaded back.

    Plot from the returned dict (plot_benchmark_results()), so the figures drawn
    now are exactly the ones the file reproduces later.

    Args:
        path:              Where to write the file (.json).
        full_results:      dict mapping mode name to the full_model_benchmark()
                           result, each with 'num_nps', 'num_passes' and a
                           'mapping' (from plot_utils.mapping_summary()) added.
        per_layer_results: Result dict from per_layer_benchmark().
        per_layer_mapping: mapping_summary() of the model as the per-layer plot
                           shows it.
        sparsity_dict:     Per-layer output sparsity, from compute_sparsity().
        model_name:        Model name or path; a path is stored as its file stem.
        example:           Example name, shown in the figure titles.
        figures:           dict with the file names of the 'full' and 'layers'
                           figures, which plot_benchmark_results() writes next to
                           the results file.
        show_power:        Whether the full-model figure has a power row.
        **info:            Extra provenance stored as it is (e.g. device,
                           akida_version, clock_hz).
    """
    results = {
        'format_version': FORMAT_VERSION,
        'example':        example,
        'model_name':     figure_title(model_name),
        **info,
        'figures':        dict(figures),
        'show_power':     show_power,
        'full_model':     {mode: _full_model_entry(res) for mode, res in full_results.items()},
        'per_layer':      _per_layer_entry(per_layer_results, per_layer_mapping, sparsity_dict),
    }
    path = pathlib.Path(path)
    path.write_text(json.dumps(results, separators=(',', ':')) + '\n')
    return load_benchmark_results(path)


def load_benchmark_results(path):
    """Read a file written by save_benchmark_results()."""
    results = json.loads(pathlib.Path(path).read_text())
    if results.get('format_version') != FORMAT_VERSION:
        raise ValueError(f'{path}: unsupported benchmark results format '
                         f'{results.get("format_version")!r} (expected {FORMAT_VERSION})')
    return results


def plot_benchmark_results(results, outdir):
    """Draw the full-model and per-layer figures from saved results.

    Args:
        results: dict from save_benchmark_results() or load_benchmark_results().
        outdir:  Directory to write the figures to, under the file names stored
                 in results['figures'].

    Returns:
        (full-model figure, per-layer figure)
    """
    outdir = pathlib.Path(outdir)
    full_fig = plot_full_model_results(results['full_model'],
                                       model_name=results['model_name'],
                                       example=results['example'],
                                       show_power=results['show_power'],
                                       savepath=outdir / results['figures']['full'])
    per_layer = results['per_layer']
    layers_fig = plot_per_layer_results(per_layer, per_layer['mapping'], per_layer['sparsity'],
                                        model_name=results['model_name'],
                                        example=results['example'],
                                        savepath=outdir / results['figures']['layers'])
    return full_fig, layers_fig


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Redraw benchmark figures from a saved benchmark results file')
    parser.add_argument('results', nargs='+', help='Benchmark results file(s) (.json)')
    parser.add_argument('-o', '--outdir',
                        help='Where to write the figures (default: next to each results file)')
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for path in map(pathlib.Path, args.results):
        results = load_benchmark_results(path)
        outdir = pathlib.Path(args.outdir) if args.outdir else path.parent
        figs = plot_benchmark_results(results, outdir)
        for fig in figs:
            plt.close(fig)
        for name in results['figures'].values():
            print(f'Saved {outdir / name}')


if __name__ == '__main__':
    main()
