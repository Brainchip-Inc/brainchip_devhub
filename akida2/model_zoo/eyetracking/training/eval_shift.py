"""Evaluate already-trained models on the sensor-shift stress test.

Usage:
    python eval_shift.py
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
SHIFT_BINS = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]


def load_model(name):
    run_dir = os.path.join(HERE, "runs", name)
    model = keras.models.load_model(os.path.join(run_dir, "model.keras"))
    scaler = np.load(os.path.join(run_dir, "y_scaler.npz"))
    return model, scaler["mean"], scaler["std"]


def main():
    d = np.load(os.path.join(DATA_DIR, "test_shifted.npz"))
    X, y, shift_deg = d["X"], d["y"], d["shift_deg"]

    d0 = np.load(os.path.join(DATA_DIR, "test.npz"))
    X0, y0 = d0["X"], d0["y"]

    results = {}
    curves = {}

    for name in MODELS:
        model, y_mean, y_std = load_model(name)

        pred0 = model.predict(X0, batch_size=2048, verbose=0) * y_std + y_mean
        err0 = float(np.mean(np.linalg.norm(y0 - pred0, axis=1)))

        pred = model.predict(X, batch_size=2048, verbose=0) * y_std + y_mean
        err = np.linalg.norm(y - pred, axis=1)

        binned = []
        for lo, hi in zip(SHIFT_BINS[:-1], SHIFT_BINS[1:]):
            m = (shift_deg >= lo) & (shift_deg < hi)
            binned.append(float(err[m].mean()) if m.any() else None)
        curves[name] = binned

        results[name] = {
            "zero_shift_mean_angular_error_deg": err0,
            "shifted_mean_angular_error_deg": float(err.mean()),
            "degradation_ratio": float(err.mean() / err0),
        }
        print(f"{name}: zero-shift={err0:.3f} deg, shifted={err.mean():.3f} deg "
              f"({results[name]['degradation_ratio']:.1f}x worse)")

    with open(os.path.join(HERE, "runs", "shift_eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    _plot(curves, shift_deg)


def _plot(curves, shift_deg):
    import matplotlib.pyplot as plt

    centers = [(lo + hi) / 2 for lo, hi in zip(SHIFT_BINS[:-1], SHIFT_BINS[1:])]
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, binned in curves.items():
        ax.plot(centers, binned, marker="o", label=name)
    ax.set_xlabel("sensor shift magnitude (deg, proxy for physical misalignment)")
    ax.set_ylabel("mean angular error (deg)")
    ax.set_title("Accuracy vs. sensor-array shift (models trained on zero-shift data)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "shift_robustness.png")
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
