"""Visualize the gaze -> simulated-photodiode pipeline on one recording.

Usage:
    python visualize.py                      # synthetic demo trace
    python visualize.py path/to/recording.csv  # real GazeBase CSV

Writes a PNG to ../docs/figures/.
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from psog_simulate import simulate_photodiode_array, sensor_ring_positions, make_windows

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "docs", "figures")


def synthetic_gaze_trace(duration_s=4.0, fs=120):
    """A step-saccade-and-fixate trace roughly like the PSOG calibration task:
    gaze jumps between random targets and holds briefly, similar to GazeBase's
    random-saccade task."""
    n = int(duration_s * fs)
    t_ms = np.arange(n) * (1000.0 / fs)
    rng = np.random.default_rng(0)
    n_targets = 8
    hold = n // n_targets
    targets = rng.uniform(-15, 15, size=(n_targets, 2))
    xy = np.zeros((n, 2))
    for i in range(n_targets):
        start = i * hold
        end = n if i == n_targets - 1 else (i + 1) * hold
        # fast saccade over 5 samples then hold, with small fixational jitter
        seg = np.tile(targets[i], (end - start, 1))
        ramp = min(5, end - start)
        prev = targets[i - 1] if i > 0 else targets[0]
        for k in range(ramp):
            seg[k] = prev + (targets[i] - prev) * (k + 1) / ramp
        seg = seg + rng.normal(0, 0.15, size=seg.shape)
        xy[start:end] = seg
    return t_ms, xy


def main():
    if len(sys.argv) > 1:
        from gazebase_loader import load_recording

        t_ms, xy = load_recording(sys.argv[1])
        label = os.path.basename(sys.argv[1])
        fs = 1000.0

        # full recordings run ~100s at 1000 Hz; show a representative slice
        duration_s = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
        n = int(duration_s * fs)
        t_ms, xy = t_ms[:n], xy[:n]
    else:
        t_ms, xy = synthetic_gaze_trace()
        label = "synthetic demo trace"
        fs = 120.0

    signal, sensors = simulate_photodiode_array(xy, n_channels=8, seed=0)
    X, y = make_windows(signal, xy, window=64, stride=32)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f"Gaze -> simulated 8-channel photodiode pipeline ({label})")

    # 1. Scanpath
    ax = axes[0, 0]
    ax.plot(xy[:, 0], xy[:, 1], "-", lw=0.6, alpha=0.6, color="tab:blue")
    ax.scatter(xy[:, 0], xy[:, 1], c=np.arange(len(xy)), cmap="viridis", s=4)
    sc = ax.scatter(sensors[:, 0], sensors[:, 1], marker="x", c="red", s=80, label="virtual sensors")
    ax.set_title("Gaze scanpath (deg. visual angle)")
    ax.set_xlabel("x (deg)")
    ax.set_ylabel("y (deg)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")

    # 2. Gaze x, y vs time
    ax = axes[0, 1]
    t_s = (t_ms - t_ms[0]) / 1000.0
    ax.plot(t_s, xy[:, 0], label="gaze x", lw=1)
    ax.plot(t_s, xy[:, 1], label="gaze y", lw=1)
    ax.set_title("Raw gaze trajectory vs time")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("degrees")
    ax.legend(fontsize=8)

    # 3. Simulated photodiode channels, stacked with vertical offset
    ax = axes[1, 0]
    offset = 1.3
    for c in range(signal.shape[1]):
        ax.plot(t_s, signal[:, c] + c * offset, lw=0.8)
    ax.set_title("Simulated 8-channel photodiode signal (offset per channel)")
    ax.set_xlabel("time (s)")
    ax.set_yticks([c * offset for c in range(signal.shape[1])])
    ax.set_yticklabels([f"ch{c}" for c in range(signal.shape[1])])

    # 4. One training window example
    ax = axes[1, 1]
    xi = min(3, len(X) - 1)
    for c in range(X.shape[2]):
        ax.plot(X[xi, :, c], lw=0.8)
    ax.set_title(f"One (64, 8) training window -> target gaze {np.round(y[xi], 1)}")
    ax.set_xlabel("timestep in window")
    ax.set_ylabel("simulated intensity")

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    suffix = "synthetic" if len(sys.argv) <= 1 else "gazebase"
    out_path = os.path.join(FIG_DIR, f"pipeline_demo_{suffix}.png")
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")
    print(f"trace: {len(xy)} samples @ ~{fs:.0f} Hz, windows: X={X.shape}, y={y.shape}")


if __name__ == "__main__":
    main()
