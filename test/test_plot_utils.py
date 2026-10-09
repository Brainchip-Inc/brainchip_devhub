"""Unit tests for brainchip_utils.plot_utils (no hardware or model files required)."""
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

from brainchip_utils.plot_utils import (_mode_summary, figure_title,  # noqa: E402
                                        plot_full_model_results)


@pytest.mark.parametrize("model_name, example, expected", [
    ('./models/akidanet_vww_qat.fbz', 'Visual Wake Words', 'Visual Wake Words · akidanet_vww_qat'),
    ('pretrained_models/akidanet_vww_qat.fbz', None, 'akidanet_vww_qat'),
    ('akidanet_imagenet alpha=0.5 160px', 'ImageNet', 'ImageNet · akidanet_imagenet alpha=0.5 160px'),
    (None, 'ImageNet', 'ImageNet'),
    (None, None, None),
])
def test_figure_title_never_shows_a_path(model_name, example, expected):
    assert figure_title(model_name, example) == expected


def test_mode_summary_with_and_without_power():
    result = {'num_nps': 17, 'num_passes': 1, 'mean_clk_ms': 8.0265, 'power': None}
    assert _mode_summary(result) == '17 NPs, 1 pass · 8.027 ms/inference'
    result.update(num_passes=2, power={'avg_dynamic_mw': 25.12, 'avg_dynamic_energy_mj': 0.2041})
    assert _mode_summary(result) == ('17 NPs, 2 passes · 8.027 ms/inference · '
                                     '25.1 mW, 0.204 mJ dynamic')


def _fake_power(level_mw, rng):
    """A power trace in the shape full_model_benchmark() returns: floor, inference, floor."""
    t = np.arange(0.0, 3.0, 0.01)
    power = 110 + level_mw * ((t >= 1.0) & (t <= 2.0)) + rng.normal(0, 0.5, t.size)
    return {'readings': [[ti, 1.0, pi] for ti, pi in zip(t, power)],
            'repeat_meta': [{'floor_pre_start': 0.0, 'floor_post_end': 3.0,
                             'inf_timestamps': list(np.linspace(1.0, 2.0, 11))}],
            'avg_floor_mw': 110.0, 'avg_dynamic_mw': level_mw,
            'avg_dynamic_energy_mj': level_mw / 10}


def _fake_model(nps_per_pass):
    layer = lambda i, n: SimpleNamespace(name=f'layer_{i}',  # noqa: E731
                                         mapping=SimpleNamespace(nps=[0] * n))
    passes = [SimpleNamespace(layers=[layer(i, n) for i, n in enumerate(p)]) for p in nps_per_pass]
    return SimpleNamespace(map=lambda *a, **kw: None, sequences=[SimpleNamespace(passes=passes)])


def test_full_model_figure_has_one_legend_and_shared_zoomed_power_axes():
    rng = np.random.default_rng(0)
    results = {mode: {'num_nps': 4, 'num_passes': 2, 'mean_clk_ms': 1.0,
                      'power': _fake_power(level, rng)}
               for mode, level in [('Minimal', 20.0), ('AllNps', 40.0)]}
    fig = plot_full_model_results(results, _fake_model([[1, 1], [2]]), device=None,
                                  model_name='./models/x.fbz', example='Example')

    assert fig._suptitle.get_text() == 'Example · x'
    assert len(fig.legends) == 1
    assert all(ax.get_legend() is None for ax in fig.axes)
    assert 'Pass boundary' in [t.get_text() for t in fig.legends[0].get_texts()]

    power_axes = fig.axes[:2]
    assert power_axes[0].get_ylim() == power_axes[1].get_ylim()
    assert power_axes[0].get_ylim()[0] > 0     # zoomed to the data, not from 0


def test_many_layers_get_vertical_labels():
    rng = np.random.default_rng(0)
    results = {'Minimal': {'num_nps': 30, 'num_passes': 1, 'mean_clk_ms': 1.0, 'power': None}}
    fig = plot_full_model_results(results, _fake_model([[1] * 30]), device=None,
                                  model_name='x', show_power=False)
    labels = fig.axes[0].get_xticklabels()
    assert len(labels) == 30 and all(label.get_rotation() == 90 for label in labels)
