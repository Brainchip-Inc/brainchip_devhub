"""Quantize the trained Akida-compatible Tiny Temporal CNN with QuantizeML.

Loads the float tf_keras model trained by training/train_akida.py, quantizes
it to 8-bit weights/activations (Akida's standard config) using real
training-set samples for calibration, and evaluates the quantized model's
accuracy before handing off to cnn2snn for Akida conversion.

Usage:
    python quantize_model.py
"""

import json
import os

import numpy as np
import tf_keras as keras
from quantizeml.models import quantize, QuantizationParams

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data", "processed")
FLOAT_MODEL_DIR = os.path.join(ROOT, "training", "runs", "cnn_akida")


def load_split_uint8(name):
    d = np.load(os.path.join(DATA_DIR, f"{name}.npz"))
    X = d["X"]
    X_u8 = np.clip(np.round(X * 255.0), 0, 255).astype(np.uint8)[:, :, None, :]
    return X_u8, d["y"]


def angular_error_deg(y_true, y_pred):
    return float(np.mean(np.linalg.norm(y_true - y_pred, axis=1)))


def main():
    model = keras.models.load_model(os.path.join(FLOAT_MODEL_DIR, "model.h5"))
    scaler = np.load(os.path.join(FLOAT_MODEL_DIR, "y_scaler.npz"))
    y_mean, y_std = scaler["mean"], scaler["std"]

    X_train, _ = load_split_uint8("train")
    X_test, y_test = load_split_uint8("test")

    rng = np.random.default_rng(0)
    calib_idx = rng.choice(len(X_train), size=1024, replace=False)
    calib_samples = X_train[calib_idx]

    qparams = QuantizationParams(input_weight_bits=8, weight_bits=8, activation_bits=8)
    q_model = quantize(model, qparams=qparams, samples=calib_samples, num_samples=1024, batch_size=128)
    q_model.summary()

    y_pred = q_model.predict(X_test, batch_size=1024, verbose=0) * y_std + y_mean
    q_ang_err = angular_error_deg(y_test, y_pred)
    print(f"test (quantized int8, pre-Akida-conversion) mean angular error={q_ang_err:.3f} deg")

    os.makedirs(HERE, exist_ok=True)
    q_model.save(os.path.join(HERE, "model_quantized.h5"))
    metrics = {"test_mean_angular_error_deg": q_ang_err, "qparams": "w8_a8"}
    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
