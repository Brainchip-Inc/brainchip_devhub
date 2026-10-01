"""Quantize the causal/streaming Tiny Temporal CNN with QuantizeML.

quantize() calls sanitize() internally, which calls replace_conv3d() -
this automatically bufferizes the Conv3D layers into BufferTempConv before
quantizing, no manual transform call needed.

Usage:
    python quantize_model_streaming.py
"""

import json
import os
import sys

import numpy as np
import tf_keras as keras
from quantizeml.models import quantize, QuantizationParams, reset_buffers

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data", "processed")
FLOAT_MODEL_DIR = os.path.join(ROOT, "training", "runs", "cnn_streaming")

sys.path.insert(0, os.path.join(ROOT, "training"))
from streaming_eval_utils import predict_streaming  # noqa: E402


def load_split_int8(name):
    d = np.load(os.path.join(DATA_DIR, f"{name}_seq.npz"))
    X = d["X"]
    X_u8 = np.clip(np.round(X * 255.0), 0, 255).astype(np.int16)
    X_i8 = (X_u8 - 128).astype(np.int8)[:, :, None, None, :]
    return X_i8, d["y"]  # y: (N, 64, 2), no trailing spatial dims


def angular_error_deg(y_true, y_pred):
    return float(np.mean(np.linalg.norm(y_true - y_pred, axis=-1)))


def main():
    model = keras.models.load_model(os.path.join(FLOAT_MODEL_DIR, "model.h5"))
    scaler = np.load(os.path.join(FLOAT_MODEL_DIR, "y_scaler.npz"))
    y_mean, y_std = scaler["mean"], scaler["std"]

    X_train, _ = load_split_int8("train")
    X_test, y_test = load_split_int8("test")

    # After bufferization the model consumes ONE timestep at a time
    # (None, 1, 1, C), so calibration samples must be individual frames, not
    # whole (64, 1, 1, C) windows - flatten the window axis away.
    rng = np.random.default_rng(0)
    frames = X_train.reshape(-1, 1, 1, X_train.shape[-1])
    calib_idx = rng.choice(len(frames), size=1024, replace=False)
    calib_samples = frames[calib_idx]

    qparams = QuantizationParams(input_weight_bits=8, weight_bits=8, activation_bits=8, input_dtype="int8")
    q_model = quantize(model, qparams=qparams, samples=calib_samples, num_samples=1024, batch_size=128)
    q_model.summary()

    # Evaluate on a subsample: the bufferized model must be fed one timestep
    # at a time (sequentially, per window), which is much slower per-sample
    # than a single batched call - a few thousand windows is enough to get a
    # reliable mean.
    n_eval = min(4000, len(X_test))
    eval_idx = rng.choice(len(X_test), size=n_eval, replace=False)
    X_eval, y_eval = X_test[eval_idx], y_test[eval_idx]

    def step_fn(frame):
        return q_model.predict(frame, batch_size=len(frame), verbose=0)

    y_pred = predict_streaming(step_fn, lambda: reset_buffers(q_model), X_eval, batch_size=512)
    y_pred = y_pred * y_std + y_mean
    last_step_err = angular_error_deg(y_eval[:, -1], y_pred[:, -1])
    overall_err = angular_error_deg(y_eval, y_pred)
    print(f"test (quantized, pre-conversion, n={n_eval}): overall={overall_err:.3f} deg, "
          f"last-timestep={last_step_err:.3f} deg")

    os.makedirs(HERE, exist_ok=True)
    q_model.save(os.path.join(HERE, "model_streaming_quantized.h5"))
    metrics = {
        "test_overall_mean_angular_error_deg": overall_err,
        "test_last_timestep_mean_angular_error_deg": last_step_err,
        "qparams": "w8_a8",
    }
    with open(os.path.join(HERE, "metrics_streaming.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
