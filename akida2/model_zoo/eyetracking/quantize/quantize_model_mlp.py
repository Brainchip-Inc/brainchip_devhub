"""Quantize the Akida-compatible Tiny MLP with QuantizeML.

Simpler than the CNN case: no Conv3D/BufferTempConv bufferization needed,
just Flatten (folded into the following Dense by sanitize()) + QuantizedDense
layers throughout.

Usage:
    python quantize_model_mlp.py
"""

import json
import os

import numpy as np
import tf_keras as keras
from quantizeml.models import quantize, QuantizationParams

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data", "processed")
FLOAT_MODEL_DIR = os.path.join(ROOT, "training", "runs", "mlp_akida")


def load_split_int8(name):
    d = np.load(os.path.join(DATA_DIR, f"{name}.npz"))
    X = d["X"]
    X_u8 = np.clip(np.round(X * 255.0), 0, 255).astype(np.int16)
    X_i8 = (X_u8 - 128).astype(np.int8).reshape(len(X_u8), 1, 1, -1)  # (N, 1, 1, 512) - matches model input
    return X_i8, d["y"]


def angular_error_deg(y_true, y_pred):
    return float(np.mean(np.linalg.norm(y_true - y_pred, axis=1)))


def main():
    model = keras.models.load_model(os.path.join(FLOAT_MODEL_DIR, "model.h5"))
    scaler = np.load(os.path.join(FLOAT_MODEL_DIR, "y_scaler.npz"))
    y_mean, y_std = scaler["mean"], scaler["std"]

    X_train, _ = load_split_int8("train")
    X_test, y_test = load_split_int8("test")

    rng = np.random.default_rng(0)
    calib_idx = rng.choice(len(X_train), size=1024, replace=False)
    calib_samples = X_train[calib_idx]

    qparams = QuantizationParams(input_weight_bits=8, weight_bits=8, activation_bits=8)
    q_model = quantize(model, qparams=qparams, samples=calib_samples, num_samples=1024, batch_size=128)
    q_model.summary()

    y_pred = q_model.predict(X_test, batch_size=1024, verbose=0).reshape(len(X_test), 2) * y_std + y_mean
    q_ang_err = angular_error_deg(y_test, y_pred)
    print(f"test (quantized int8, pre-Akida-conversion) mean angular error={q_ang_err:.3f} deg")

    os.makedirs(HERE, exist_ok=True)
    q_model.save(os.path.join(HERE, "model_mlp_quantized.h5"))
    metrics = {"test_mean_angular_error_deg": q_ang_err, "qparams": "w8_a8"}
    with open(os.path.join(HERE, "metrics_mlp.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
