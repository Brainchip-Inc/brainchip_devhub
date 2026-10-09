"""Unit tests for brainchip_utils.zoo_table and the landing README generators (no toolchain needed)."""
import runpy
from pathlib import Path

from brainchip_utils.zoo_table import html_table, value

ROOT = Path(__file__).resolve().parents[1]


def test_value_treats_placeholders_as_missing():
    metrics = {'a': '1.0', 'b': 'TBD', 'c': ''}
    assert [value(metrics, k) for k in 'abcd'] == ['1.0', None, None, None]


def test_leading_cells_merge_down_rows_of_the_same_example_only():
    rows = [('vww', ['Person', 'Cls', '8-bit']),
            ('vww', ['Person', 'Cls', '4-bit']),
            ('kws', ['Keyword', 'Cls', '8-bit'])]
    html = html_table(['Task', 'Category', 'Variant'], rows, right_aligned={2}, mergeable=2)
    assert html.count('rowspan="2"') == 2            # Task and Category of the two vww rows
    assert html.count('<td>Cls</td>') == 1           # kws keeps its own Category cell
    assert html.count('align="right"') == 3


def test_akida2_table_picks_the_mapping_with_fewest_cycles():
    gen = runpy.run_path(str(ROOT / 'akida2' / 'update_readme.py'))
    metrics = {'w8a8_minimal_cycles': '300', 'w8a8_allnps_cycles': '200', 'w8a8_hwpr_cycles': '250',
               'w8a8_allnps_latency_ms': '8.000', 'w8a8_allnps_projected_ms': '0.200',
               'w8a8_akida_acc': '90.00%'}
    row = {'task': 'T', 'category': 'C', 'dataset': 'D', 'variant': '8-bit',
           'metric_key': 'w8a8_akida_acc', 'metric_label': 'acc.', 'bench_prefix': 'w8a8_'}
    cells = gen['_cells']('ex', row, metrics)
    assert cells[4:8] == ['90.00% acc.', 'AllNPs', '8.000', '0.200']



def test_akida2_table_prefers_the_simpler_mapping_on_a_tie():
    gen = runpy.run_path(str(ROOT / 'akida2' / 'update_readme.py'))
    metrics = {'w8a8_minimal_cycles': '1044066', 'w8a8_allnps_cycles': '874556', 'w8a8_hwpr_cycles': '874554'}
    assert gen['_fastest_mapping'](metrics, 'w8a8_') == 'allnps'
    metrics['w8a8_hwpr_cycles'] = '600000'
    assert gen['_fastest_mapping'](metrics, 'w8a8_') == 'hwpr'
