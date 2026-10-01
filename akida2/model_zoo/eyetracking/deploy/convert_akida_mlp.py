"""Convert the quantized Tiny MLP to an Akida model and evaluate it.

No buffered/FIFO layers here (plain feedforward Dense network), so unlike
the streaming CNN, a single batched predict() call over the whole test set
is valid - no per-window sequential loop needed.

Usage:
    python convert_akida_mlp.py
"""

import json
import os

import numpy as np
import tf_keras as keras
from cnn2snn import convert

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data", "processed")
QUANT_DIR = os.path.join(ROOT, "quantize")
FLOAT_MODEL_DIR = os.path.join(ROOT, "training", "runs", "mlp_akida")


def load_split_int8(name):
    d = np.load(os.path.join(DATA_DIR, f"{name}.npz"))
    X = d["X"]
    X_u8 = np.clip(np.round(X * 255.0), 0, 255).astype(np.int16)
    X_i8 = (X_u8 - 128).astype(np.int8).reshape(len(X_u8), 1, 1, -1)  # (N, 1, 1, 512)
    return X_i8, d["y"]


def angular_error_deg(y_true, y_pred):
    return float(np.mean(np.linalg.norm(y_true - y_pred, axis=1)))


def main():
    q_model = keras.models.load_model(os.path.join(QUANT_DIR, "model_mlp_quantized.h5"))
    scaler = np.load(os.path.join(FLOAT_MODEL_DIR, "y_scaler.npz"))
    y_mean, y_std = scaler["mean"], scaler["std"]

    akida_model = convert(q_model)
    print(akida_model.summary())

    X_test, y_test = load_split_int8("test")
    X_test = X_test.astype(np.int8)

    pred_raw = akida_model.predict(X_test, batch_size=1024)
    y_pred = pred_raw.reshape(len(X_test), -1) * y_std + y_mean
    akida_ang_err = angular_error_deg(y_test, y_pred)
    print(f"test (Akida-converted, full test set) mean angular error={akida_ang_err:.3f} deg")

    stats = {"layer_count": akida_model.layer_count, "macs": akida_model.macs}
    try:
        stats["statistics"] = str(akida_model.statistics)
    except Exception as e:
        stats["statistics_error"] = str(e)

    out_path = os.path.join(HERE, "model_mlp.fbz")
    akida_model.save(out_path)
    print(f"saved {out_path}")

    metrics = {
        "test_mean_angular_error_deg": akida_ang_err,
        "n_eval_windows": len(X_test),
        **stats,
    }
    with open(os.path.join(HERE, "metrics_mlp.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
