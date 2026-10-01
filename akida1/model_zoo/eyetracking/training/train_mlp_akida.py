"""Train the Akida-compatible Tiny MLP (see models/tiny_mlp_akida.py) on the
zero-shift GazeBase-PSOG dataset.

Uses tf_keras (not tf.keras/Keras 3) since that's what quantizeml/cnn2snn
require. Input is quantized to uint8 here (matching a real ADC-digitized
sensor reading) rather than kept as float32. Unlike the Tiny Temporal CNN,
the MLP needed no architecture rework for Akida - see
docs/benchmark_report.md.

Usage:
    python train_mlp_akida.py [epochs]
"""

import json
import os
import sys

import numpy as np
import tf_keras as keras

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data", "processed")

sys.path.insert(0, os.path.join(ROOT, "models"))
from tiny_mlp_akida import build_model  # noqa: E402


def load_split_int8(name):
    d = np.load(os.path.join(DATA_DIR, f"{name}.npz"))
    X = d["X"]  # (N, 64, 8) float32, roughly in [0.1, 1.05]
    X_u8 = np.clip(np.round(X * 255.0), 0, 255).astype(np.int16)
    X_i8 = (X_u8 - 128).astype(np.int8).reshape(len(X_u8), 1, 1, -1)  # (N, 1, 1, 512) - pre-flattened, see model docstring
    return X_i8, d["y"]


def angular_error_deg(y_true, y_pred):
    return float(np.mean(np.linalg.norm(y_true - y_pred, axis=1)))


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    X_train, y_train = load_split_int8("train")
    X_val, y_val = load_split_int8("val")
    X_test, y_test = load_split_int8("test")
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

    # model output is (N, 1, 1, 2) - reshape targets to match, or MSE
    # broadcasts (N, 2) against (N, 1, 1, 2) incorrectly (padding y_true's
    # shape with LEADING 1s per numpy broadcasting rules gives (1, 1, N, 2),
    # not the intended per-sample match)
    def to_model_shape(y):
        return to_norm(y).reshape(len(y), 1, 1, 2)

    history = model.fit(
        X_train,
        to_model_shape(y_train),
        validation_data=(X_val, to_model_shape(y_val)),
        epochs=epochs,
        batch_size=512,
        callbacks=callbacks,
        verbose=2,
    )

    y_pred = from_norm(model.predict(X_test, batch_size=1024, verbose=0).reshape(len(X_test), 2))
    test_ang_err = angular_error_deg(y_test, y_pred)
    print(f"test (float, tf_keras MLP) mean angular error={test_ang_err:.3f} deg")

    out_dir = os.path.join(HERE, "runs", "mlp_akida")
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "model.h5"))
    np.savez(os.path.join(out_dir, "y_scaler.npz"), mean=y_mean, std=y_std)
    metrics = {
        "model": "mlp_akida_float",
        "params": int(model.count_params()),
        "epochs_trained": len(history.history["loss"]),
        "test_mean_angular_error_deg": test_ang_err,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
