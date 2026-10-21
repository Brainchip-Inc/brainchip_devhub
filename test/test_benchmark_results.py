"""Unit tests for brainchip_utils.benchmark_results (no hardware or model files required)."""
import json
import subprocess
import sys
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

from brainchip_utils.benchmark_results import (load_benchmark_results, main,  # noqa: E402
                                               plot_benchmark_results, save_benchmark_results)
from brainchip_utils.plot_utils import (mapping_summary, plot_full_model_results,  # noqa: E402
                                        plot_per_layer_results)


def _fake_power(level_mw, rng, repeats=3):
    """Raw power data in the shape full_model_benchmark() returns: floor, inference, floor."""
    readings, repeat_meta = [], []
    for rr in range(repeats):
        t0 = 10.0 * rr
        t = t0 + np.arange(0.0, 3.0, 0.001)
        power = 110 + level_mw * ((t >= t0 + 1.0) & (t <= t0 + 2.0)) + rng.normal(0, 0.5, t.size)
        readings += [(ti, 0.8, pi / 0.8) for ti, pi in zip(t, power)]
        repeat_meta.append({'floor_pre_start': t0, 'floor_post_end': t0 + 3.0,
                            'inf_timestamps': list(np.linspace(t0 + 1.0, t0 + 2.0, 101))})
    return {'readings': readings, 'repeat_meta': repeat_meta,
            'avg_floor_mw': 110.0, 'avg_total_mw': 110.0 + level_mw, 'avg_dynamic_mw': level_mw,
            'avg_energy_mj': 1.0, 'avg_dynamic_energy_mj': level_mw / 100}


def _fake_model(nps_per_pass_per_seq):
    """A mapped-model stand-in: one list of passes (lists of NPs per layer) per sequence."""
    count = iter(range(1000))
    layer = lambda n: SimpleNamespace(name=f'layer_{next(count)}',  # noqa: E731
                                      mapping=SimpleNamespace(nps=[0] * n))
    return SimpleNamespace(sequences=[
        SimpleNamespace(passes=[SimpleNamespace(layers=[layer(n) for n in p]) for p in seq])
        for seq in nps_per_pass_per_seq])


def _fake_benchmark(rng, with_power=True):
    model = _fake_model([[[1, 2], [3]], [[1]]])
    full_results = {}
    for mode, level in [('Minimal', 20.0), ('AllNps', 40.0)]:
        full_results[mode] = {
            'map_mode': object(),  # an akida.MapMode in a real run; not stored
            'mean_inf_clk': 1.2e6, 'std_inf_clk': 1e3, 'mean_clk_ms': 3.0, 'mean_inf_ms': 3.1,
            'std_inf_ms': 0.1, 'n_inferences': 300, 'num_nps': 7, 'num_passes': 3,
            'num_sequences': 2, 'mapping': mapping_summary(model),
            'power': _fake_power(level, rng) if with_power else None}
    clocks = np.cumsum(rng.integers(1000, 2000, size=(4, 50)), axis=0).astype(float)
    per_layer_clocks = np.copy(clocks)
    per_layer_clocks[1:] = clocks[1:] - clocks[:-1]
    per_layer = {'layer_names': [f'layer_{i}' for i in range(4)],
                 'per_layer_clocks': per_layer_clocks.mean(axis=1),
                 'per_layer_ms': per_layer_clocks.mean(axis=1) / 400e6 * 1000,
                 'cumulative_clocks': clocks, 'cumulative_times': clocks * 2.5}
    sparsity = {f'layer_{i}': 0.1 * i for i in range(4)}
    return model, full_results, per_layer, sparsity


def _save(tmp_path, rng, **kwargs):
    model, full_results, per_layer, sparsity = _fake_benchmark(rng, **kwargs)
    results = save_benchmark_results(
        tmp_path / 'results.json', full_results, per_layer, mapping_summary(model), sparsity,
        model_name='./models/x.fbz', example='Example',
        figures={'full': 'full.png', 'layers': 'layers.png'}, device='PCIe/AKD1500/16MB/0')
    return model, full_results, per_layer, sparsity, results


def test_mapping_summary_marks_pass_and_sequence_ends():
    summary = mapping_summary(_fake_model([[[1, 2], [3]], [[1]]]))
    assert summary == {'layer_names': ['layer_0', 'layer_1', 'layer_2', 'layer_3'],
                       'layer_nps': [1, 2, 3, 1], 'pass_ends': [1], 'sequence_ends': [2]}


def test_saved_file_is_plain_json_without_raw_power_readings(tmp_path):
    *_, results = _save(tmp_path, np.random.default_rng(0))
    stored = json.loads((tmp_path / 'results.json').read_text())
    assert stored == results
    assert stored['model_name'] == 'x' and stored['device'] == 'PCIe/AKD1500/16MB/0'
    power = stored['full_model']['Minimal']['power']
    assert 'readings' not in power and 'repeat_meta' not in power
    assert len(power['trace']['repeats_mw']) == 3
    assert 'map_mode' not in stored['full_model']['Minimal']
    assert np.shape(stored['per_layer']['cumulative_times']) == (4, 50)


def test_plots_from_saved_results_match_plots_from_live_results(tmp_path):
    model, full_results, per_layer, sparsity, results = _save(tmp_path, np.random.default_rng(0))
    full_fig, layers_fig = plot_benchmark_results(results, tmp_path)
    assert (tmp_path / 'full.png').exists() and (tmp_path / 'layers.png').exists()

    live_full = plot_full_model_results(full_results, model_name='./models/x.fbz',
                                        example='Example')
    live_layers = plot_per_layer_results(per_layer, model, sparsity,
                                         model_name='./models/x.fbz', example='Example')
    for saved_fig, live_fig in [(full_fig, live_full), (layers_fig, live_layers)]:
        assert saved_fig._suptitle.get_text() == live_fig._suptitle.get_text()
        for saved_ax, live_ax in zip(saved_fig.axes, live_fig.axes, strict=True):
            for saved_line, live_line in zip(saved_ax.get_lines(), live_ax.get_lines(), strict=True):
                np.testing.assert_allclose(np.asarray(saved_line.get_ydata(), dtype=float),
                                           np.asarray(live_line.get_ydata(), dtype=float), atol=0.01)
            assert ([p.get_height() for p in saved_ax.patches] ==
                    pytest.approx([p.get_height() for p in live_ax.patches]))
            assert ([t.get_text() for t in saved_ax.get_xticklabels()] ==
                    [t.get_text() for t in live_ax.get_xticklabels()])


def test_results_without_power_redraw_without_a_power_row(tmp_path):
    *_, results = _save(tmp_path, np.random.default_rng(0), with_power=False)
    results['show_power'] = False
    full_fig, _ = plot_benchmark_results(results, tmp_path)
    assert len(full_fig.axes) == 2   # one mapping panel per mode


def test_command_line_redraws_next_to_the_results_file(tmp_path):
    _save(tmp_path, np.random.default_rng(0))
    for name in ('full.png', 'layers.png'):
        (tmp_path / name).unlink(missing_ok=True)
    main([str(tmp_path / 'results.json')])
    assert (tmp_path / 'full.png').exists() and (tmp_path / 'layers.png').exists()


def test_unknown_format_version_is_rejected(tmp_path):
    path = tmp_path / 'results.json'
    path.write_text(json.dumps({'format_version': 99}))
    with pytest.raises(ValueError, match='unsupported'):
        load_benchmark_results(path)


def test_redrawing_does_not_need_akida():
    # Block the akida import, as on a machine without it installed
    code = ("import sys; sys.modules['akida'] = None; "
            "import brainchip_utils.benchmark_results")
    subprocess.run([sys.executable, '-c', code], check=True)
