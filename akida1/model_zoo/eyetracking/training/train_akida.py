"""Train the Akida-compatible Tiny Temporal CNN (Conv2D reformulation, see
models/tiny_temporal_cnn_akida.py) on the zero-shift GazeBase-PSOG dataset.

Uses tf_keras (not tf.keras/Keras 3) since that's what quantizeml/cnn2snn
require. Input is quantized to uint8 here (matching a real ADC-digitized
sensor reading) rather than kept as float32.

Usage:
    python train_akida.py [epochs]
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
from tiny_temporal_cnn_akida import build_model  # noqa: E402


def load_split_uint8(name):
    d = np.load(os.path.join(DATA_DIR, f"{name}.npz"))
    X = d["X"]  # (N, 64, 8) float32, roughly in [0.1, 1.05]
    X_u8 = np.clip(np.round(X * 255.0), 0, 255).astype(np.uint8)
    X_u8 = X_u8[:, :, None, :]  # (N, 64, 1, 8)
    return X_u8, d["y"]


def angular_error_deg(y_true, y_pred):
    return float(np.mean(np.linalg.norm(y_true - y_pred, axis=1)))


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    X_train, y_train = load_split_uint8("train")
    X_val, y_val = load_split_uint8("val")
    X_test, y_test = load_split_uint8("test")
    print(f"train={X_train.shape} val={X_val.shape} test={X_test.shape}, dtype={X_train.dtype}")

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
    test_ang_err = angular_error_deg(y_test, y_pred)
    print(f"test (float, Conv2D reformulation) mean angular error={test_ang_err:.3f} deg")

    out_dir = os.path.join(HERE, "runs", "cnn_akida")
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "model.h5"))
    np.savez(os.path.join(out_dir, "y_scaler.npz"), mean=y_mean, std=y_std)
    metrics = {
        "model": "cnn_akida_float",
        "params": int(model.count_params()),
        "epochs_trained": len(history.history["loss"]),
        "test_mean_angular_error_deg": test_ang_err,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
