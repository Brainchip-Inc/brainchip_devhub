"""Simulate a small photosensor (PSOG) array response from a gaze trajectory.

Real photodiode/PSOG hardware measures reflected IR intensity at a handful of
fixed positions around the eye; intensity drops when the (IR-absorbing) pupil
sweeps under a sensor's field of view. Following the model-based approach in
Rigas, Raffle & Komogortsev, "Photosensor Oculography: Survey and Parametric
Analysis of Designs Using Model-Based Simulation" (arXiv:1707.05413) and the
Gaussian-windowed sensor response used in Katrychuk, Griffith & Komogortsev's
PS-OG pipeline (https://github.com/pseudowolfvn/psog_nn), each sensor's
response is modeled as a Gaussian dip in reflected intensity centered on the
gaze direction, without requiring a full 3D eyeball render.

This is a simplified analytic stand-in for that pipeline: it lets us validate
the training pipeline shape end-to-end before deciding whether the full
render-based simulation is worth reproducing.
"""

import numpy as np


def sensor_ring_positions(n_channels=8, radius_deg=12.0):
    """Angular (x, y) positions, in degrees of visual angle, of `n_channels`
    sensors evenly spaced around the eye at `radius_deg` from center."""
    angles = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
    return np.stack([radius_deg * np.cos(angles), radius_deg * np.sin(angles)], axis=1)


def simulate_photodiode_array(
    gaze_xy,
    n_channels=8,
    radius_deg=12.0,
    sigma_deg=9.0,
    dip_depth=0.8,
    baseline=1.0,
    noise_std=0.02,
    sensor_shift=(0.0, 0.0),
    seed=None,
):
    """Convert a (T, 2) gaze trajectory (degrees of visual angle) into a
    (T, n_channels) simulated photodiode intensity signal.

    Each channel starts near `baseline` and dips toward
    `baseline * (1 - dip_depth)` as gaze approaches that sensor's position,
    following a Gaussian falloff with width `sigma_deg`. Gaussian sensor
    noise with std `noise_std` is added per channel per sample.

    `sensor_shift` is a constant (dx, dy) offset applied to every sensor's
    position before computing the response - a stand-in for the whole array
    physically shifting relative to the eye (e.g. headset slippage between
    sessions), following the shift-robustness stress test in Katrychuk,
    Griffith & Komogortsev 2019 (real shifts there are reported in mm; here
    the shift is expressed directly in the same degrees-of-visual-angle units
    as everything else in this simulation, since we have no lens/eye-relief
    geometry to convert mm to degrees - treat the magnitude as a relative
    stress level, not a calibrated physical distance).
    """
    gaze_xy = np.asarray(gaze_xy, dtype=np.float64)
    sensors = sensor_ring_positions(n_channels, radius_deg) + np.asarray(sensor_shift)

    # (T, 1, 2) - (1, C, 2) -> (T, C, 2) -> (T, C)
    dist_sq = np.sum((gaze_xy[:, None, :] - sensors[None, :, :]) ** 2, axis=-1)
    dip = dip_depth * np.exp(-dist_sq / (2 * sigma_deg**2))
    signal = baseline * (1.0 - dip)

    if noise_std > 0:
        rng = np.random.default_rng(seed)
        signal = signal + rng.normal(0.0, noise_std, size=signal.shape)

    return signal, sensors


def make_windows(signal, gaze_xy, window=64, stride=32):
    """Slice a (T, C) signal and (T, 2) gaze target into overlapping windows
    for supervised training: X is (N, window, C), y is the gaze at the last
    timestep of each window, (N, 2)."""
    signal = np.asarray(signal)
    gaze_xy = np.asarray(gaze_xy)
    T = signal.shape[0]
    starts = range(0, T - window + 1, stride)
    X = np.stack([signal[s : s + window] for s in starts])
    y = np.stack([gaze_xy[s + window - 1] for s in starts])
    return X, y


def make_windows_seq(signal, gaze_xy, window=64, stride=32):
    """Like make_windows, but keeps the FULL per-timestep gaze trajectory as
    the target instead of only the window's last sample: X is
    (N, window, C), y is (N, window, 2). Used for the causal/streaming Akida
    model, which predicts gaze continuously at every timestep from
    past-only context rather than once per whole window."""
    signal = np.asarray(signal)
    gaze_xy = np.asarray(gaze_xy)
    T = signal.shape[0]
    starts = range(0, T - window + 1, stride)
    X = np.stack([signal[s : s + window] for s in starts])
    y = np.stack([gaze_xy[s : s + window] for s in starts])
    return X, y
