import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib._color_data as mcd
import akida


_COLOR_OFFSET = 50


def pretty_print_sparsity(sparsity_dict):
    col_w = max(len(k) for k in sparsity_dict) + 2
    print(f"\n{'Layer':<{col_w}} {'Sparsity':>10}")
    print("-" * (col_w + 11))
    for layer, sparsity in sparsity_dict.items():
        print(f"{layer:<{col_w}} {sparsity:>9.2%}")
    print("-" * (col_w + 11))
    mean_sparsity = np.mean(list(sparsity_dict.values()))
    print(f"{'Mean':<{col_w}} {mean_sparsity:>9.2%}")


def _layer_colors(n):
    return list(mcd.XKCD_COLORS.values())[_COLOR_OFFSET:_COLOR_OFFSET + n]


def plot_per_layer_timing(results, ax, colors=None):
    layer_names = results['layer_names']
    per_layer_ms = results['per_layer_ms']
    num_layers = len(layer_names)

    if colors is None:
        colors = ['steelblue'] * num_layers

    raw_times = results['cumulative_times']  # (num_layers, repeats) in ns
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

    cumulative_ms = np.mean(results['cumulative_times'], axis=1) / 1e6

    x = np.arange(num_layers)
    ax.bar(x, cumulative_ms, color=colors, edgecolor='white')
    ax.set_ylabel('Cumulative latency (ms)')
    ax.set_title('Cumulative timing')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)


def plot_mapping(ak_model, ax, colors=None):
    layer_names = []
    layer_nps = []
    pass_ends = []
    seq_ends = []

    for si, seq in enumerate(ak_model.sequences):
        for pi, hwpass in enumerate(seq.passes):
            for layer in hwpass.layers:
                layer_names.append(layer.name)
                layer_nps.append(len(layer.mapping.nps))
            is_last_pass = (pi == len(seq.passes) - 1)
            is_last_seq = (si == len(ak_model.sequences) - 1)
            if not (is_last_pass and is_last_seq):
                pass_end_idx = len(layer_names) - 0.5
                if is_last_pass:
                    seq_ends.append(pass_end_idx)
                else:
                    pass_ends.append(pass_end_idx)

    if colors is None:
        colors = _layer_colors(len(layer_names))

    x = np.arange(len(layer_names))
    ax.bar(x, layer_nps, color=colors, edgecolor='white')

    for pos in pass_ends:
        ax.axvline(x=pos, color='gray', linestyle='--', linewidth=1.0)
    for pos in seq_ends:
        ax.axvline(x=pos, color='firebrick', linestyle='--', linewidth=1.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('NPs used')
    ax.set_title('Hardware mapping')
    ax.yaxis.get_major_locator().set_params(integer=True)

    legend_handles = []
    if pass_ends:
        legend_handles.append(mpatches.Patch(color='gray', label='Pass boundary'))
    if seq_ends:
        legend_handles.append(mpatches.Patch(color='firebrick', label='Sequence boundary'))
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=8)


def plot_power_trace(power_data, ax):
    """Plot per-repeat power traces with the mean overlaid.

    Each repeat is time-normalised so that t=0 coincides with the start of the
    inference window.  Individual repeat traces are drawn with transparency;
    the mean trace is drawn on top.

    Args:
        power_data: The 'power' dict returned by full_model_benchmark, which
                    must contain 'readings' and 'repeat_meta'.
    """
    readings_arr = np.array(power_data['readings'])
    abs_times = readings_arr[:, 0]
    powers    = readings_arr[:, 1] * readings_arr[:, 2]  # V × mA = mW

    repeat_meta = power_data['repeat_meta']
    avg_floor   = power_data['avg_floor_mw']
    avg_dyn     = power_data['avg_dynamic_mw']
    avg_dyn_e   = power_data['avg_dynamic_energy_mj']

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
        return

    # Common time grid spanning all repeats
    t_grid = np.linspace(
        min(tr[0][0]  for tr in repeat_traces),
        max(tr[0][-1] for tr in repeat_traces),
        500,
    )

    interp_powers = []
    for t_rel, p_sel in repeat_traces:
        p_interp = np.interp(t_grid, t_rel, p_sel)
        interp_powers.append(p_interp)
        ax.plot(t_grid, p_interp, lw=0.6, color='steelblue', alpha=0.25)

    ax.plot(t_grid, np.mean(interp_powers, axis=0),
            lw=1.5, color='steelblue', label='Mean power')

    mean_inf_duration = np.mean([
        meta['inf_timestamps'][-1] - meta['inf_timestamps'][0]
        for meta in repeat_meta
    ])
    n_inf = len(repeat_meta[0]['inf_timestamps']) - 1
    avg_inf_time_ms = mean_inf_duration / n_inf * 1000

    ax.axvline(0,                  color='seagreen',  ls='--', lw=1.2, label='Inference start')
    ax.axvline(mean_inf_duration,  color='firebrick', ls='--', lw=1.2, label='Inference end')
    ax.axhline(avg_floor,          color='gray',      ls=':',  lw=1.0,
               label=f'Floor: {avg_floor:.1f} mW')
    ax.set_ylabel('Power (mW)')
    ax.set_xlabel('Time relative to inference start (s)')
    ax.set_title('Power timeline')
    ax.legend(fontsize=8, loc="lower center")

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


def plot_full_model_results(full_results, ak_model, device, model_name=None, savepath=None):
    """Plot power trace and hardware mapping for each map mode.

    Layout: rows = plot type (power, mapping); cols = map mode.
    Axis limits are synchronised across columns so modes are directly comparable.

    Args:
        full_results: dict mapping mode name (e.g. 'Minimal', 'AllNps') to the
                      result dict returned by full_model_benchmark().
        ak_model:     akida.Model — re-mapped internally for each mode. Left in
                      the state of the last mode after this call returns.
        device:       Akida device passed to ak_model.map().
        model_name:   Optional figure suptitle string.
        savepath:     If provided, save the figure to this path (PNG).

    Returns:
        matplotlib Figure.
    """
    modes = list(full_results.keys())
    ncols = len(modes)
    fig, axs = plt.subplots(2, ncols, figsize=(7 * ncols, 8),
                            constrained_layout=True, squeeze=False)

    if model_name is not None:
        fig.suptitle(model_name, fontsize=14, fontweight='bold')

    for i, mode_name in enumerate(modes):
        result = full_results[mode_name]
        ak_model.map(device, mode=getattr(akida.MapMode, mode_name))

        ax_pwr = axs[0, i]
        ax_map = axs[1, i]

        power_data = result.get('power')
        if power_data is not None:
            plot_power_trace(power_data, ax_pwr)
            ax_pwr.yaxis.label.set_fontsize(12)
            ax_pwr.xaxis.label.set_fontsize(12)
            ax_pwr.tick_params(axis='both', labelsize=11)
            if ax_pwr.get_legend() is not None:
                plt.setp(ax_pwr.get_legend().get_texts(), fontsize=11)
            for txt in ax_pwr.texts:
                txt.set_fontsize(10)
        else:
            _power_placeholder(ax_pwr)
        ax_pwr.set_title(f'MapMode: {mode_name}', loc='left', fontsize=12, color='#444444')

        plot_mapping(ak_model, ax_map)
        ax_map.yaxis.label.set_fontsize(12)
        ax_map.tick_params(axis='y', labelsize=11)
        plt.setp(ax_map.get_xticklabels(), fontsize=11)
        if ax_map.get_legend() is not None:
            plt.setp(ax_map.get_legend().get_texts(), fontsize=11)
        ax_map.set_title(f'MapMode: {mode_name}', loc='left', fontsize=12, color='#444444')

    # Synchronise power axes limits across columns (only for columns with real data)
    pwr_axes = [axs[0, i] for i, mm in enumerate(modes)
                if full_results[mm].get('power') is not None]
    if len(pwr_axes) > 1:
        x_min = min(ax.get_xlim()[0] for ax in pwr_axes)
        x_max = max(ax.get_xlim()[1] for ax in pwr_axes)
        y_min = 0. 
        y_max = max(ax.get_ylim()[1] for ax in pwr_axes)
        for ax in pwr_axes:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

    # Synchronise mapping y-axis limits across columns
    map_axes = [axs[1, i] for i in range(ncols)]
    y_max_map = max(ax.get_ylim()[1] for ax in map_axes)
    for ax in map_axes:
        ax.set_ylim(0, y_max_map)

    if savepath is not None:
        fig.savefig(savepath, dpi=150)

    return fig


def plot_per_layer_results(per_layer_results, ak_model, sparsity_dict,
                           model_name=None, savepath=None):
    """Plot hardware mapping, per-layer timing, and input sparsity, stacked vertically.

    ak_model must already be mapped (without hw_only=True) so that
    ak_model.sequences is populated. This function does not call ak_model.map().

    Args:
        per_layer_results: Result dict from per_layer_benchmark().
        ak_model:          akida.Model with sequences populated by a prior map() call.
        sparsity_dict:     dict[layer_name, float] of output sparsity per layer,
                           as returned by compute_sparsity(). Shifted by one position
                           to derive per-layer input sparsity (first layer = 0).
        model_name:        Optional figure suptitle string.
        savepath:          If provided, save the figure to this path (PNG).

    Returns:
        matplotlib Figure.
    """
    layer_names = per_layer_results['layer_names']
    num_layers = len(layer_names)

    output_sparsity = [sparsity_dict[name] for name in layer_names]
    input_sparsity_pct = [0.0] + [s * 100.0 for s in output_sparsity[:-1]]

    fig, axs = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)

    if model_name is not None:
        fig.suptitle(model_name, fontsize=14, fontweight='bold')

    # Row 0: per-layer timing
    plot_per_layer_timing(per_layer_results, axs[0])
    axs[0].set_xticklabels([])
    axs[0].set_xlabel('')
    axs[0].yaxis.label.set_fontsize(12)
    axs[0].title.set_fontsize(12)
    axs[0].tick_params(axis='y', labelsize=11)

    # Row 1: input sparsity
    x = np.arange(num_layers)
    axs[1].bar(x, input_sparsity_pct, color='mediumpurple', edgecolor='white')
    axs[1].set_ylim(0, 100)
    axs[1].set_ylabel('Input sparsity (%)', fontsize=12)
    axs[1].set_title('Per-layer input sparsity', fontsize=12)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([])
    axs[1].set_xlabel('')
    axs[1].tick_params(axis='y', labelsize=11)

    # Row 2: hardware mapping (layer name labels only on bottom subplot)
    plot_mapping(ak_model, axs[2], colors='steelblue')
    axs[2].yaxis.label.set_fontsize(12)
    axs[2].title.set_fontsize(12)
    axs[2].tick_params(axis='y', labelsize=11)
    plt.setp(axs[2].get_xticklabels(), fontsize=11)
    if axs[2].get_legend() is not None:
        plt.setp(axs[2].get_legend().get_texts(), fontsize=11)

    if savepath is not None:
        fig.savefig(savepath, dpi=150)

    return fig


def plot_benchmark_summary(results, ak_model, model_name=None, power_data=None, savepath=None):
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)

    if model_name is not None:
        fig.suptitle(model_name, fontsize=13, fontweight='bold')

    plot_per_layer_timing(results, axs[0, 0])
    plot_cumulative_timing(results, axs[0, 1])
    plot_mapping(ak_model, axs[1, 0])

    ax_pwr = axs[1, 1]
    if power_data is not None:
        plot_power_trace(power_data, ax_pwr)
    else:
        ax_pwr.set_facecolor('#f5f5f5')
        ax_pwr.text(0.5, 0.5, 'Power measurements\n(not yet available)',
                    ha='center', va='center', transform=ax_pwr.transAxes,
                    fontsize=11, color='#999999', style='italic')
        ax_pwr.set_title('Power', color='#999999')
        for spine in ax_pwr.spines.values():
            spine.set_edgecolor('#cccccc')

    if savepath is not None:
        fig.savefig(savepath, dpi=150)

    return fig
