"""Per-session calibration: correct for sensor shift using a handful of
known-gaze calibration windows from the same recording, instead of trying to
train shift-robustness into the model (see eval_shift.py's negative result
on shift-augmented training).

Why a constant bias correction should work: in psog_simulate.py, a sensor
shift enters only as sensors = ring_positions + shift, so
    dist(gaze, ring_i + shift) == dist(gaze - shift, ring_i)
i.e. a shifted array reading for true gaze g is IDENTICAL to a zero-shift
array reading for gaze (g - shift). A model trained on zero-shift data
should therefore predict approximately (g - shift) when fed shifted-array
input for true gaze g - a near-constant additive bias, not a complex
distortion. If so, estimating that bias from a few calibration points (like
a real eye-tracker's calibration routine: look at K known points, measure
the offset, apply it going forward) should recover most of the accuracy
lost to shift, without retraining or changing the architecture.

Usage:
    python eval_calibration.py
"""

import json
import os

import numpy as np
from tensorflow import keras

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data", "processed")
FIG_DIR = os.path.join(ROOT, "docs", "figures")

MODELS = ["cnn", "mlp"]
CALIB_COUNTS = [1, 2, 3, 5, 10, 20]


def load_model(name):
    run_dir = os.path.join(HERE, "runs", name)
    model = keras.models.load_model(os.path.join(run_dir, "model.keras"))
    scaler = np.load(os.path.join(run_dir, "y_scaler.npz"))
    return model, scaler["mean"], scaler["std"]


def group_recordings(subject_id, shift_deg):
    """Group window indices by (subject, shift) - a unique shift per
    recording stands in for a unique (subject, session) recording, since we
    didn't separately store a session column."""
    groups = {}
    for i, key in enumerate(zip(subject_id.tolist(), shift_deg.tolist())):
        groups.setdefault(key, []).append(i)
    return groups


def main():
    d = np.load(os.path.join(DATA_DIR, "test_shifted.npz"))
    X, y, subject_id, shift_deg = d["X"], d["y"], d["subject_id"], d["shift_deg"]
    groups = group_recordings(subject_id, shift_deg)
    print(f"{len(groups)} recordings in shifted test set")

    rng = np.random.default_rng(0)
    results = {}

    for name in MODELS:
        model, y_mean, y_std = load_model(name)
        pred_raw = model.predict(X, batch_size=2048, verbose=0) * y_std + y_mean

        uncorrected_errs = []
        for idx in groups.values():
            idx = np.array(idx)
            uncorrected_errs.append(np.linalg.norm(y[idx] - pred_raw[idx], axis=1))
        uncorrected_mean = float(np.concatenate(uncorrected_errs).mean())

        curve = []
        for k in CALIB_COUNTS:
            errs = []
            for idx in groups.values():
                idx = np.array(idx)
                if len(idx) <= k:
                    continue
                perm = rng.permutation(len(idx))
                calib_idx = idx[perm[:k]]
                eval_idx = idx[perm[k:]]
                bias = np.mean(y[calib_idx] - pred_raw[calib_idx], axis=0)
                corrected = pred_raw[eval_idx] + bias
                errs.append(np.linalg.norm(y[eval_idx] - corrected, axis=1))
            curve.append(float(np.concatenate(errs).mean()))
            print(f"{name}: k={k} calib windows -> mean angular error {curve[-1]:.3f} deg")

        results[name] = {
            "uncorrected_shifted_mean_angular_error_deg": uncorrected_mean,
            "calib_counts": CALIB_COUNTS,
            "corrected_mean_angular_error_deg": curve,
        }
        print(f"{name}: uncorrected={uncorrected_mean:.3f} deg, "
              f"corrected(k={CALIB_COUNTS[-1]})={curve[-1]:.3f} deg")

    with open(os.path.join(HERE, "runs", "calibration_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    _plot(results)


def _plot(results):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, r in results.items():
        ax.axhline(r["uncorrected_shifted_mean_angular_error_deg"], linestyle=":", alpha=0.5,
                    label=f"{name} uncorrected")
        ax.plot(r["calib_counts"], r["corrected_mean_angular_error_deg"], marker="o", label=f"{name} calibrated")
    ax.set_xlabel("number of calibration windows")
    ax.set_ylabel("mean angular error (deg)")
    ax.set_title("Per-session bias calibration recovers shifted-sensor accuracy")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "calibration_recovery.png")
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
