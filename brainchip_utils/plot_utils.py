import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


MAPPING_COLOR = 'steelblue'
PASS_STYLE = dict(color='gray', linestyle='--', linewidth=1.0)
SEQUENCE_STYLE = dict(color='firebrick', linestyle='--', linewidth=1.5)
REPEAT_STYLE = dict(lw=0.6, color='steelblue', alpha=0.25)
MEAN_STYLE = dict(lw=1.5, color='steelblue')
INF_START_STYLE = dict(color='seagreen', ls='--', lw=1.2)
INF_END_STYLE = dict(color='firebrick', ls='--', lw=1.2)
FLOOR_STYLE = dict(color='gray', ls=':', lw=1.0)

# Points per repeat in a power trace, on a grid shared by every repeat of a map mode
POWER_TRACE_POINTS = 500


def pretty_print_sparsity(sparsity_dict):
    col_w = max(len(k) for k in sparsity_dict) + 2
    print(f"\n{'Layer':<{col_w}} {'Sparsity':>10}")
    print("-" * (col_w + 11))
    for layer, sparsity in sparsity_dict.items():
        print(f"{layer:<{col_w}} {sparsity:>9.2%}")
    print("-" * (col_w + 11))
    mean_sparsity = np.mean(list(sparsity_dict.values()))
    print(f"{'Mean':<{col_w}} {mean_sparsity:>9.2%}")


def figure_title(model_name, example=None):
    """Figure title from an example name and a model name or path.

    A path such as './models/akidanet_vww_qat.fbz' is reduced to its stem, so
    no machine-specific path ends up in a committed figure.
    """
    if model_name is None:
        return example
    name = str(model_name)
    if '/' in name or name.endswith(('.fbz', '.h5')):
        name = pathlib.Path(name).stem
    return f'{example} · {name}' if example else name


def _set_title(fig, model_name, example):
    title = figure_title(model_name, example)
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
        # A little more room between the figure title and the panel titles below it
        fig.get_layout_engine().set(h_pad=0.1)


def _figure_legend(fig, handles):
    """One legend for the whole figure, below the panels."""
    if handles:
        fig.legend(handles=handles, loc='outside lower center', ncols=len(handles),
                   fontsize=11, frameon=False)


def _power_legend_handles():
    return [Line2D([], [], label='Single repeat', **{**REPEAT_STYLE, 'alpha': 0.5, 'lw': 1.2}),
            Line2D([], [], label='Mean power', **MEAN_STYLE),
            Line2D([], [], label='Inference run start', **INF_START_STYLE),
            Line2D([], [], label='Inference run end', **INF_END_STYLE),
            Line2D([], [], label='Idle floor', **FLOOR_STYLE)]


def _mapping_legend_handles(has_pass, has_seq):
    handles = []
    if has_pass:
        handles.append(Line2D([], [], label='Pass boundary', **PASS_STYLE))
    if has_seq:
        handles.append(Line2D([], [], label='Sequence boundary', **SEQUENCE_STYLE))
    return handles


def plot_per_layer_timing(results, ax, colors=None):
    layer_names = results['layer_names']
    per_layer_ms = np.asarray(results['per_layer_ms'], dtype=float)  # null (saved NaN) -> NaN
    num_layers = len(layer_names)

    if colors is None:
        colors = ['steelblue'] * num_layers

    raw_times = np.asarray(results['cumulative_times'], dtype=float)  # (num_layers, repeats) in ns
    per_layer_raw = np.copy(raw_times)
    per_layer_raw[1:] = raw_times[1:] - raw_times[:-1]
    per_layer_ms_std = np.std(per_layer_raw / 1e6, axis=1)

    x = np.arange(num_layers)
    ax.bar(x, per_layer_ms, color=colors, edgecolor='white',
           yerr=per_layer_ms_std, capsize=3, error_kw={'linewidth': 0.8, 'ecolor': 'dimgray'})
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Per-layer timing')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)


def plot_cumulative_timing(results, ax, colors=None):
    layer_names = results['layer_names']
    num_layers = len(layer_names)

    if colors is None:
        colors = ['seagreen'] * num_layers

    cumulative_ms = np.mean(np.asarray(results['cumulative_times'], dtype=float), axis=1) / 1e6

    x = np.arange(num_layers)
    ax.bar(x, cumulative_ms, color=colors, edgecolor='white')
    ax.set_ylabel('Cumulative latency (ms)')
    ax.set_title('Cumulative timing')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)


def mapping_summary(ak_model):
    """NPs per layer in mapping order, with the pass and sequence boundaries.

    ak_model must be mapped without hw_only=True, so that ak_model.sequences is
    populated. The result is a plain dict that plot_mapping() draws and that a
    benchmark results file stores, so the mapping can be plotted again without
    a device.

    Returns:
        dict with keys:
            layer_names   - list of layer names, in mapping order
            layer_nps     - list of NPs used by each layer
            pass_ends     - indices of layers that end a pass (within a sequence)
            sequence_ends - indices of layers that end a sequence
        The last layer of the model ends neither.
    """
    layer_names = []
    layer_nps = []
    pass_ends = []
    sequence_ends = []

    for si, seq in enumerate(ak_model.sequences):
        for pi, hwpass in enumerate(seq.passes):
            for layer in hwpass.layers:
                layer_names.append(layer.name)
                layer_nps.append(len(layer.mapping.nps))
            is_last_pass = (pi == len(seq.passes) - 1)
            is_last_seq = (si == len(ak_model.sequences) - 1)
            if not (is_last_pass and is_last_seq):
                if is_last_pass:
                    sequence_ends.append(len(layer_names) - 1)
                else:
                    pass_ends.append(len(layer_names) - 1)

    return {'layer_names': layer_names, 'layer_nps': layer_nps,
            'pass_ends': pass_ends, 'sequence_ends': sequence_ends}


def plot_mapping(mapping, ax, colors=None, legend=True):
    """Bar chart of NPs per layer, with pass and sequence boundaries.

    mapping is either a mapped akida.Model (see mapping_summary()) or the dict
    mapping_summary() returns.

    Returns (has_pass_boundary, has_sequence_boundary), so a caller drawing
    several panels can build one shared legend (pass legend=False).
    """
    if hasattr(mapping, 'sequences'):
        mapping = mapping_summary(mapping)
    layer_names = mapping['layer_names']
    pass_ends = [i + 0.5 for i in mapping['pass_ends']]
    seq_ends = [i + 0.5 for i in mapping['sequence_ends']]

    if colors is None:
        colors = MAPPING_COLOR

    x = np.arange(len(layer_names))
    ax.bar(x, mapping['layer_nps'], color=colors, edgecolor='white')

    for pos in pass_ends:
        ax.axvline(x=pos, **PASS_STYLE)
    for pos in seq_ends:
        ax.axvline(x=pos, **SEQUENCE_STYLE)

    ax.set_xticks(list(x))
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('NPs used')
    ax.set_title('Hardware mapping')
    ax.yaxis.get_major_locator().set_params(integer=True)

    handles = _mapping_legend_handles(bool(pass_ends), bool(seq_ends))
    if legend and handles:
        ax.legend(handles=handles, fontsize=8)
    return bool(pass_ends), bool(seq_ends)


def _fit_layer_labels(ax, max_slanted=20):
    """Layer names at 45° for up to max_slanted layers; above that, vertical and smaller so
    neighbouring names don't overlap in a 7-inch column."""
    labels = ax.get_xticklabels()
    if len(labels) > max_slanted:
        plt.setp(labels, rotation=90, ha='center', fontsize=8)
    else:
        plt.setp(labels, fontsize=11)


def power_trace(power_data, num_points=POWER_TRACE_POINTS):
    """Per-repeat power traces on a common time grid, t=0 at inference start.

    This is the data plot_power_trace() draws, and what a benchmark results
    file stores instead of the raw sensor readings (which run to hundreds of
    thousands of points per map mode).

    Args:
        power_data: The 'power' dict returned by full_model_benchmark (with
                    'readings' and 'repeat_meta'), or a dict that already holds
                    a 'trace', which is returned as it is.
        num_points: Points per repeat on the common time grid.

    Returns:
        dict with keys:
            t_s                   - common time grid (s), t=0 at inference start
            repeats_mw            - one power trace (mW) per repeat on that grid
            inf_duration_s        - length of each repeat's inference run (s)
            inferences_per_repeat - forward calls in each inference run
        or None if no repeat has enough readings to draw.
    """
    if 'trace' in power_data:
        return power_data['trace']

    readings_arr = np.array(power_data['readings'])
    abs_times = readings_arr[:, 0]
    powers    = readings_arr[:, 1] * readings_arr[:, 2]  # V × mA = mW
    repeat_meta = power_data['repeat_meta']

    # Build one time-normalised power trace per repeat
    repeat_traces = []  # list of (t_rel, p) arrays, t=0 at inference start
    for meta in repeat_meta:
        t_i0  = meta['inf_timestamps'][0]
        t_pre = meta['floor_pre_start']
        t_end = meta['floor_post_end']
        mask  = (abs_times >= t_pre) & (abs_times <= t_end)
        t_rel = abs_times[mask] - t_i0
        p_sel = powers[mask]
        if len(t_rel) >= 2:
            repeat_traces.append((t_rel, p_sel))

    if not repeat_traces:
        return None

    # Common time grid spanning all repeats
    t_grid = np.linspace(
        min(tr[0][0]  for tr in repeat_traces),
        max(tr[0][-1] for tr in repeat_traces),
        num_points,
    )
    return {
        't_s':                   t_grid.tolist(),
        'repeats_mw':            [np.interp(t_grid, t_rel, p_sel).tolist()
                                  for t_rel, p_sel in repeat_traces],
        'inf_duration_s':        [meta['inf_timestamps'][-1] - meta['inf_timestamps'][0]
                                  for meta in repeat_meta],
        'inferences_per_repeat': len(repeat_meta[0]['inf_timestamps']) - 1,
    }


def plot_power_trace(power_data, ax, legend=True, stats=True):
    """Plot per-repeat power traces with the mean overlaid.

    Each repeat is time-normalised so that t=0 coincides with the start of the
    inference window.  Individual repeat traces are drawn with transparency;
    the mean trace is drawn on top.

    Args:
        power_data: The 'power' dict returned by full_model_benchmark (with
                    'readings' and 'repeat_meta'), or its saved form, with a
                    'trace' from power_trace() in their place.
        legend:     Draw a legend on the axes. Pass False when the figure has a
                    shared legend.
        stats:      Draw the time/power/energy summary box on the axes.
    """
    trace = power_trace(power_data)
    if trace is None:
        return

    avg_floor   = power_data['avg_floor_mw']
    avg_dyn     = power_data['avg_dynamic_mw']
    avg_dyn_e   = power_data['avg_dynamic_energy_mj']

    t_grid = np.asarray(trace['t_s'])
    interp_powers = np.asarray(trace['repeats_mw'])
    for p_interp in interp_powers:
        ax.plot(t_grid, p_interp, **REPEAT_STYLE)

    ax.plot(t_grid, np.mean(interp_powers, axis=0), label='Mean power', **MEAN_STYLE)

    mean_inf_duration = np.mean(trace['inf_duration_s'])
    avg_inf_time_ms = mean_inf_duration / trace['inferences_per_repeat'] * 1000

    ax.axvline(0, label='Inference start', **INF_START_STYLE)
    ax.axvline(mean_inf_duration, label='Inference end', **INF_END_STYLE)
    ax.axhline(avg_floor, label=f'Floor: {avg_floor:.1f} mW', **FLOOR_STYLE)
    ax.set_ylabel('Power (mW)')
    ax.set_xlabel('Time relative to inference start (s)')
    ax.set_title('Power timeline')
    if legend:
        ax.legend(fontsize=8, loc="lower center")
    if not stats:
        return

    stats_text = (
        f'Avg time per inference: {avg_inf_time_ms:.2f} ms\n'
        f'Avg dynamic power:      {avg_dyn:.1f} mW\n'
        f'Avg dynamic energy:     {avg_dyn_e:.3f} mJ'
    )
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
            va='top', ha='right', fontsize=9, family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))


def _power_placeholder(ax):
    ax.set_facecolor('#f5f5f5')
    ax.text(0.5, 0.5, 'Power measurements\n(not yet available)',
            ha='center', va='center', transform=ax.transAxes,
            fontsize=11, color='#999999', style='italic')
    ax.set_title('Power', color='#999999')
    for spine in ax.spines.values():
        spine.set_edgecolor('#cccccc')


def _mode_summary(result):
    """One-line summary of a full-model result, shown under the map-mode name."""
    parts = []
    if 'num_nps' in result:
        passes = result.get('num_passes')
        parts.append(f"{result['num_nps']} NPs, {passes} pass{'es' if passes != 1 else ''}"
                     if passes is not None else f"{result['num_nps']} NPs")
    if 'mean_clk_ms' in result:
        parts.append(f"{result['mean_clk_ms']:.3f} ms/inference")
    power = result.get('power')
    if power is not None:
        parts.append(f"{power['avg_dynamic_mw']:.1f} mW, "
                     f"{power['avg_dynamic_energy_mj']:.3f} mJ dynamic")
    return ' · '.join(parts)


def plot_full_model_results(full_results, ak_model=None, device=None, model_name=None,
                            savepath=None, show_power=True, example=None):
    """Plot power trace and hardware mapping for each map mode.

    Layout: rows = plot type (power, mapping); cols = map mode. If show_power is
    False the power row is omitted and only the mapping row is drawn. Each column
    is headed by its map mode and a one-line summary; one legend below the panels
    covers every column. Axis limits are synchronised across columns so modes are
    directly comparable.

    Args:
        full_results: dict mapping mode name (e.g. 'Minimal', 'AllNps') to the
                      result dict returned by full_model_benchmark(), optionally
                      with 'num_nps' and 'num_passes' added. A result that holds
                      a 'mapping' (from mapping_summary()) is plotted from it, as
                      results loaded from a benchmark results file are.
        ak_model:     akida.Model, re-mapped for each mode without a 'mapping'.
                      Left in the state of the last mode re-mapped. Not needed
                      when every result holds a 'mapping'.
        device:       Akida device passed to ak_model.map().
        model_name:   Model name or path; a path is shown as its file stem.
        savepath:     If provided, save the figure to this path (PNG).
        show_power:   If False, omit the power row (use when power measurement is
                      not available, so no empty placeholder panels are drawn).
        example:      Optional example name, shown before the model name.

    Returns:
        matplotlib Figure.
    """
    modes = list(full_results.keys())
    ncols = len(modes)
    nrows = 2 if show_power else 1
    fig, axs = plt.subplots(nrows, ncols, figsize=(7 * ncols, 8.5 if show_power else 5.5),
                            layout='constrained', squeeze=False)
    _set_title(fig, model_name, example)

    has_power = False
    has_pass = has_seq = False
    for i, mode_name in enumerate(modes):
        result = full_results[mode_name]
        mapping = result.get('mapping')
        if mapping is None:
            import akida
            ak_model.map(device, mode=getattr(akida.MapMode, mode_name))
            mapping = ak_model

        ax_top = axs[0, i]
        ax_map = axs[nrows - 1, i]

        if show_power:
            power_data = result.get('power')
            if power_data is not None:
                plot_power_trace(power_data, axs[0, i], legend=False, stats=False)
                axs[0, i].set_xlabel('Time relative to inference run start (s)', fontsize=12)
                axs[0, i].yaxis.label.set_fontsize(12)
                axs[0, i].tick_params(axis='both', labelsize=11)
                has_power = True
            else:
                _power_placeholder(axs[0, i])

        p, q = plot_mapping(mapping, ax_map, legend=False)
        has_pass, has_seq = has_pass or p, has_seq or q
        ax_map.yaxis.label.set_fontsize(12)
        ax_map.tick_params(axis='y', labelsize=11)
        _fit_layer_labels(ax_map)
        if show_power:
            ax_map.set_title('')

        # Column header: the map mode as the axes title (so the layout reserves room for
        # it below the figure title), with the summary line between it and the axes
        ax_top.set_title(f'MapMode.{mode_name}', fontsize=13, fontweight='bold', pad=26)
        ax_top.annotate(_mode_summary(result), xy=(0.5, 1.0), xycoords='axes fraction',
                        xytext=(0, 6), textcoords='offset points', ha='center', va='bottom',
                        fontsize=11, color='#444444')

    # Share the power axes' limits across columns (only for columns with real data).
    # The y-axis is zoomed to the data rather than starting at 0, so the step from
    # idle floor to inference power stays visible; a shared range keeps the
    # columns directly comparable.
    pwr_axes = [axs[0, i] for i, mm in enumerate(modes)
                if show_power and full_results[mm].get('power') is not None]
    if len(pwr_axes) > 1:
        x_min = min(ax.get_xlim()[0] for ax in pwr_axes)
        x_max = max(ax.get_xlim()[1] for ax in pwr_axes)
        y_min = min(ax.get_ylim()[0] for ax in pwr_axes)
        y_max = max(ax.get_ylim()[1] for ax in pwr_axes)
        for ax in pwr_axes:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

    # Synchronise mapping y-axis limits across columns
    map_axes = [axs[nrows - 1, i] for i in range(ncols)]
    y_max_map = max(ax.get_ylim()[1] for ax in map_axes)
    for ax in map_axes:
        ax.set_ylim(0, y_max_map)

    handles = _power_legend_handles() if has_power else []
    _figure_legend(fig, handles + _mapping_legend_handles(has_pass, has_seq))

    if savepath is not None:
        fig.savefig(savepath, dpi=150)

    return fig


def plot_per_layer_results(per_layer_results, ak_model, sparsity_dict,
                           model_name=None, savepath=None, example=None):
    """Plot per-layer timing, input sparsity and hardware mapping, stacked vertically.

    ak_model must already be mapped (without hw_only=True) so that
    ak_model.sequences is populated. This function does not call ak_model.map().

    Args:
        per_layer_results: Result dict from per_layer_benchmark().
        ak_model:          akida.Model with sequences populated by a prior map() call,
                           or the dict mapping_summary() returns for it.
        sparsity_dict:     dict[layer_name, float] of output sparsity per layer,
                           as returned by compute_sparsity(). Shifted by one position
                           to derive per-layer input sparsity (first layer = 0).
        model_name:        Model name or path; a path is shown as its file stem.
        savepath:          If provided, save the figure to this path (PNG).
        example:           Optional example name, shown before the model name.

    Returns:
        matplotlib Figure.
    """
    layer_names = per_layer_results['layer_names']
    num_layers = len(layer_names)

    # Some layers (e.g. Dequantizer) are not included in the sparsity dict
    # sparsity_dict.get(name, 0) returns the default value 0 in that case
    output_sparsity = [sparsity_dict.get(name, 0) for name in layer_names]
    input_sparsity_pct = [0.0] + [s * 100.0 for s in output_sparsity[:-1]]

    fig, axs = plt.subplots(3, 1, figsize=(14, 10), layout='constrained', sharex=True)
    _set_title(fig, model_name, example)

    # Row 0: per-layer timing
    plot_per_layer_timing(per_layer_results, axs[0])
    axs[0].yaxis.label.set_fontsize(12)
    axs[0].title.set_fontsize(12)
    axs[0].tick_params(axis='y', labelsize=11)

    # Row 1: input sparsity
    x = np.arange(num_layers)
    axs[1].bar(x, input_sparsity_pct, color='mediumpurple', edgecolor='white')
    axs[1].set_ylim(0, 100)
    axs[1].set_ylabel('Input sparsity (%)', fontsize=12)
    axs[1].set_title('Per-layer input sparsity', fontsize=12)
    axs[1].tick_params(axis='y', labelsize=11)

    # Row 2: hardware mapping (layer name labels only on bottom subplot)
    has_pass, has_seq = plot_mapping(ak_model, axs[2], legend=False)
    axs[2].yaxis.label.set_fontsize(12)
    axs[2].title.set_fontsize(12)
    axs[2].tick_params(axis='y', labelsize=11)
    plt.setp(axs[2].get_xticklabels(), fontsize=11)
    for ax in axs[:2]:
        ax.tick_params(axis='x', labelbottom=False)

    _figure_legend(fig, _mapping_legend_handles(has_pass, has_seq))

    if savepath is not None:
        fig.savefig(savepath, dpi=150)

    return fig


def plot_benchmark_summary(results, ak_model, model_name=None, power_data=None, savepath=None,
                           example=None):
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), layout='constrained')
    _set_title(fig, model_name, example)

    plot_per_layer_timing(results, axs[0, 0])
    plot_cumulative_timing(results, axs[0, 1])
    plot_mapping(ak_model, axs[1, 0])

    if power_data is not None:
        plot_power_trace(power_data, axs[1, 1])
    else:
        _power_placeholder(axs[1, 1])

    if savepath is not None:
        fig.savefig(savepath, dpi=150)

    return fig
