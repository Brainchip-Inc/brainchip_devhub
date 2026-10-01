"""Train the causal/streaming Tiny Temporal CNN (models/tiny_temporal_cnn_streaming.py)
on the per-timestep sequence dataset (data/build_dataset_seq.py).

Unlike the original window-to-single-gaze model, this predicts gaze at
EVERY timestep from causal (past-only) context, matching the actual Akida
BufferTempConv deployment target. Early timesteps in a window have little
receptive field ("warm-up") and are expected to be less accurate - this is
reported explicitly via a per-position error curve, not hidden.

Usage:
    python train_akida_streaming.py [epochs]
"""

import json
import os
import sys

import numpy as np
import tf_keras as keras

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data", "processed")
FIG_DIR = os.path.join(ROOT, "docs", "figures")

sys.path.insert(0, os.path.join(ROOT, "models"))
from tiny_temporal_cnn_streaming import build_model  # noqa: E402


def load_split_int8(name):
    d = np.load(os.path.join(DATA_DIR, f"{name}_seq.npz"))
    X = d["X"]  # (N, 64, 8) float32
    X_u8 = np.clip(np.round(X * 255.0), 0, 255).astype(np.int16)
    X_i8 = (X_u8 - 128).astype(np.int8)  # Akida's buffered temporal conv needs signed input
    X_i8 = X_i8[:, :, None, None, :]  # (N, 64, 1, 1, 8)
    y = d["y"][:, :, None, None, :]  # (N, 64, 1, 1, 2) - matches model output shape
    return X_i8, y


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    X_train, y_train = load_split_int8("train")
    X_val, y_val = load_split_int8("val")
    X_test, y_test = load_split_int8("test")
    print(f"train={X_train.shape} val={X_val.shape} test={X_test.shape}")

    # normalize using ALL timesteps' statistics (not just window-end)
    y_mean = y_train.reshape(-1, 2).mean(axis=0)
    y_std = y_train.reshape(-1, 2).std(axis=0)

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

    y_pred = from_norm(model.predict(X_test, batch_size=1024, verbose=0))  # (N, 64, 1, 1, 2)
    y_pred = y_pred[:, :, 0, 0, :]  # squeeze trivial spatial dims -> (N, 64, 2)
    y_test_sq = y_test[:, :, 0, 0, :]

    # per-position (warm-up) error curve
    err_per_t = np.linalg.norm(y_test_sq - y_pred, axis=2)  # (N, 64)
    per_position_mean = err_per_t.mean(axis=0)  # (64,)

    last_step_err = float(per_position_mean[-1])
    overall_err = float(err_per_t.mean())
    # "settled" error: average over the back half of the window, once the
    # receptive field has mostly filled in
    settled_err = float(per_position_mean[32:].mean())

    print(f"test: overall mean angular error={overall_err:.3f} deg, "
          f"last-timestep={last_step_err:.3f} deg, settled(t>=32)={settled_err:.3f} deg")

    out_dir = os.path.join(HERE, "runs", "cnn_streaming")
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "model.h5"))
    np.savez(os.path.join(out_dir, "y_scaler.npz"), mean=y_mean, std=y_std)
    metrics = {
        "model": "cnn_streaming_float",
        "params": int(model.count_params()),
        "epochs_trained": len(history.history["loss"]),
        "test_overall_mean_angular_error_deg": overall_err,
        "test_last_timestep_mean_angular_error_deg": last_step_err,
        "test_settled_mean_angular_error_deg": settled_err,
        "per_position_error_deg": per_position_mean.tolist(),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_position_error_deg"}, indent=2))

    _plot(per_position_mean, model.name)


def _plot(per_position_mean, name):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(per_position_mean, marker=".")
    ax.set_xlabel("timestep position within window (causal context length)")
    ax.set_ylabel("mean angular error (deg)")
    ax.set_title("Streaming model: accuracy improves as causal context accumulates")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, "streaming_warmup.png")
    fig.savefig(out_path, dpi=130)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
