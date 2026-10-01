"""Convert the quantized Tiny Temporal CNN to an Akida model and evaluate it.

Usage:
    python convert_akida.py
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
FLOAT_MODEL_DIR = os.path.join(ROOT, "training", "runs", "cnn_akida")


def load_split_uint8(name):
    d = np.load(os.path.join(DATA_DIR, f"{name}.npz"))
    X = d["X"]
    X_u8 = np.clip(np.round(X * 255.0), 0, 255).astype(np.uint8)[:, :, None, :]
    return X_u8, d["y"]


def angular_error_deg(y_true, y_pred):
    return float(np.mean(np.linalg.norm(y_true - y_pred, axis=1)))


def main():
    q_model = keras.models.load_model(os.path.join(QUANT_DIR, "model_quantized.h5"))
    scaler = np.load(os.path.join(FLOAT_MODEL_DIR, "y_scaler.npz"))
    y_mean, y_std = scaler["mean"], scaler["std"]

    akida_model = convert(q_model)
    print(akida_model.summary())

    X_test, y_test = load_split_uint8("test")
    X_test = X_test.astype(np.uint8)

    pred_raw = akida_model.predict(X_test, batch_size=1024)
    y_pred = pred_raw.reshape(len(X_test), -1) * y_std + y_mean
    akida_ang_err = angular_error_deg(y_test, y_pred)
    print(f"test (Akida-converted) mean angular error={akida_ang_err:.3f} deg")

    stats = {
        "layer_count": akida_model.layer_count,
        "macs": akida_model.macs,
    }
    try:
        stats["statistics"] = str(akida_model.statistics)
    except Exception as e:  # pragma: no cover - statistics may require a mapped device
        stats["statistics_error"] = str(e)

    out_path = os.path.join(HERE, "model.fbz")
    akida_model.save(out_path)
    print(f"saved {out_path}")

    metrics = {
        "test_mean_angular_error_deg": akida_ang_err,
        **stats,
    }
    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
