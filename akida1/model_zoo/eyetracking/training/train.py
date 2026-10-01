"""Train the Tiny Temporal CNN (or Tiny MLP) on the windowed GazeBase-PSOG
dataset built by data/build_dataset.py.

Usage:
    python train.py [cnn|mlp] [epochs] [clean|shift_aug]

`shift_aug` trains on data/processed/{train,val}_shift.npz (built by
data/build_dataset_shift_aug.py, one random 0-4 deg sensor shift per
recording) instead of the zero-shift {train,val}.npz, and evaluates on
BOTH the zero-shift test.npz and the shift-stress test_shifted.npz so the
zero-shift/shift-robustness trade-off is visible in one run.
"""

import json
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data", "processed")
FIG_DIR = os.path.join(ROOT, "docs", "figures")

sys.path.insert(0, os.path.join(ROOT, "models"))


def load_split(name):
    d = np.load(os.path.join(DATA_DIR, f"{name}.npz"))
    return d["X"], d["y"]


def angular_error_deg(y_true, y_pred):
    """Mean Euclidean error in degrees of visual angle (x, y both in degrees,
    small-angle approximation - matches how GazeBase/PSOG papers report error)."""
    return float(np.mean(np.linalg.norm(y_true - y_pred, axis=1)))


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "cnn"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    variant = sys.argv[3] if len(sys.argv) > 3 else "clean"

    if model_name == "cnn":
        from tiny_temporal_cnn import build_model
    elif model_name == "mlp":
        from tiny_mlp import build_model
    else:
        raise ValueError(f"unknown model {model_name}")

    train_split = "train_shift" if variant == "shift_aug" else "train"
    val_split = "val_shift" if variant == "shift_aug" else "val"
    X_train, y_train = load_split(train_split)
    X_val, y_val = load_split(val_split)
    X_test, y_test = load_split("test")
    print(f"train={X_train.shape} val={X_val.shape} test={X_test.shape}")

    # normalize gaze targets to roughly unit scale for stable training;
    # keep the scaler so predictions can be converted back to degrees
    y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0)

    def to_norm(y):
        return (y - y_mean) / y_std

    def from_norm(y):
        return y * y_std + y_mean

    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    model.summary()
    print(f"total params: {model.count_params():,}")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    history = model.fit(
        X_train,
        to_norm(y_train),
        validation_data=(X_val, to_norm(y_val)),
        epochs=epochs,
        batch_size=512,
        callbacks=callbacks,
        verbose=2,
    )

    y_pred = from_norm(model.predict(X_test, batch_size=1024, verbose=0))
    test_mae = float(np.mean(np.abs(y_test - y_pred)))
    test_rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
    test_ang_err = angular_error_deg(y_test, y_pred)
    print(f"test (zero-shift) MAE={test_mae:.3f} deg, RMSE={test_rmse:.3f} deg, mean angular error={test_ang_err:.3f} deg")

    run_name = model_name if variant == "clean" else f"{model_name}_shift_aug"
    out_dir = os.path.join(HERE, "runs", run_name)
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "model.keras"))
    np.savez(os.path.join(out_dir, "y_scaler.npz"), mean=y_mean, std=y_std)
    metrics = {
        "model": model_name,
        "variant": variant,
        "params": int(model.count_params()),
        "epochs_trained": len(history.history["loss"]),
        "test_mae_deg": test_mae,
        "test_rmse_deg": test_rmse,
        "test_mean_angular_error_deg": test_ang_err,
    }

    if variant == "shift_aug":
        ds = np.load(os.path.join(DATA_DIR, "test_shifted.npz"))
        Xs, ys = ds["X"], ds["y"]
        pred_shifted = from_norm(model.predict(Xs, batch_size=1024, verbose=0))
        shifted_ang_err = angular_error_deg(ys, pred_shifted)
        metrics["test_shifted_mean_angular_error_deg"] = shifted_ang_err
        print(f"test (shifted) mean angular error={shifted_ang_err:.3f} deg")

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    _plot(history, y_test, y_pred, run_name)


def _plot(history, y_test, y_pred, model_name):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f"{model_name} training results")

    ax = axes[0]
    ax.plot(history.history["loss"], label="train")
    ax.plot(history.history["val_loss"], label="val")
    ax.set_title("Loss (normalized MSE)")
    ax.set_xlabel("epoch")
    ax.legend()

    ax = axes[1]
    n = min(3000, len(y_test))
    idx = np.random.default_rng(0).choice(len(y_test), n, replace=False)
    ax.scatter(y_test[idx, 0], y_test[idx, 1], s=2, alpha=0.3, label="true")
    ax.scatter(y_pred[idx, 0], y_pred[idx, 1], s=2, alpha=0.3, label="pred")
    ax.set_title("Test gaze: true vs predicted")
    ax.set_xlabel("x (deg)")
    ax.set_ylabel("y (deg)")
    ax.legend(markerscale=4)
    ax.set_aspect("equal")

    ax = axes[2]
    err = np.linalg.norm(y_test - y_pred, axis=1)
    ax.hist(err, bins=60)
    ax.axvline(np.mean(err), color="red", linestyle="--", label=f"mean={np.mean(err):.2f} deg")
    ax.set_title("Angular error distribution")
    ax.set_xlabel("error (deg)")
    ax.legend()

    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, f"training_{model_name}.png")
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
