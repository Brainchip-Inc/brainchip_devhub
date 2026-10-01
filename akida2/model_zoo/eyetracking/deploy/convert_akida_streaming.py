"""Convert the quantized causal/streaming Tiny Temporal CNN to an Akida model
and evaluate it (software backend - no physical Akida device attached in
this environment; see docs/benchmark_report.md for the honest limitation).

Usage:
    python convert_akida_streaming.py
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
FLOAT_MODEL_DIR = os.path.join(ROOT, "training", "runs", "cnn_streaming")

def load_split_int8(name):
    d = np.load(os.path.join(DATA_DIR, f"{name}_seq.npz"))
    X = d["X"]
    X_u8 = np.clip(np.round(X * 255.0), 0, 255).astype(np.int16)
    X_i8 = (X_u8 - 128).astype(np.int8)[:, :, None, None, :]
    return X_i8, d["y"]


def angular_error_deg(y_true, y_pred):
    return float(np.mean(np.linalg.norm(y_true - y_pred, axis=-1)))


def main():
    q_model = keras.models.load_model(os.path.join(QUANT_DIR, "model_streaming_quantized.h5"))
    scaler = np.load(os.path.join(FLOAT_MODEL_DIR, "y_scaler.npz"))
    y_mean, y_std = scaler["mean"], scaler["std"]

    akida_model = convert(q_model)
    print(akida_model.summary())

    X_test, y_test = load_split_int8("test")
    rng = np.random.default_rng(1)
    # Akida's buffered/FIFO layers only correctly maintain state for
    # batch_size=1 - confirmed by direct test: batching multiple independent
    # windows together (sharing one FIFO across the batch axis) silently
    # produces garbage (all windows' errors stayed elevated instead of
    # settling). One continuous stream at a time is also how a real device
    # would actually be used. This makes per-window evaluation expensive
    # (64 sequential predict() calls each), so the eval set is small.
    n_eval = min(300, len(X_test))
    eval_idx = rng.choice(len(X_test), size=n_eval, replace=False)
    X_eval, y_eval = X_test[eval_idx].astype(np.int8), y_test[eval_idx]

    y_pred = np.zeros((n_eval, 64, 2), dtype=np.float32)
    for i in range(n_eval):
        akida_model = type(akida_model)(akida_model.layers)  # fresh FIFO state per window
        for t in range(64):
            frame = X_eval[i, t : t + 1]  # (1, 1, 1, 8)
            y_pred[i, t] = akida_model.predict(frame).reshape(2)
        if (i + 1) % 50 == 0:
            print(f"  evaluated {i + 1}/{n_eval} windows")
    y_pred = y_pred * y_std + y_mean
    last_step_err = angular_error_deg(y_eval[:, -1], y_pred[:, -1])
    overall_err = angular_error_deg(y_eval, y_pred)
    print(f"test (Akida-converted, n={n_eval}): overall={overall_err:.3f} deg, "
          f"last-timestep={last_step_err:.3f} deg")

    stats = {"layer_count": akida_model.layer_count, "macs": akida_model.macs}
    try:
        stats["statistics"] = str(akida_model.statistics)
    except Exception as e:
        stats["statistics_error"] = str(e)

    out_path = os.path.join(HERE, "model_streaming.fbz")
    akida_model.save(out_path)
    print(f"saved {out_path}")

    metrics = {
        "test_overall_mean_angular_error_deg": overall_err,
        "test_last_timestep_mean_angular_error_deg": last_step_err,
        "n_eval_windows": n_eval,
        **stats,
    }
    with open(os.path.join(HERE, "metrics_streaming.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
