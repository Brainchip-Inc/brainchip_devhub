"""Build a sensor-shift stress-test evaluation set from the TEST subjects.

Each (subject, session) recording gets ONE random sensor-array shift vector
(magnitude uniform in [0, MAX_SHIFT_DEG], random direction), applied
consistently to every window from that recording - modeling a headset that
was put on with some misalignment for that sitting, rather than shift
jittering window-to-window (which isn't physically how slippage works and
would be trivial for a model to average out).

Train/val are untouched - this only produces a new evaluation set, to check
how the ALREADY-TRAINED models (clean, zero-shift) degrade under shift,
before deciding whether shift-augmented training is worth adding.

Usage:
    python build_shift_eval.py [path/to/GazeBase_v2_0.zip] [max_shift_deg]
"""

import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

from psog_simulate import simulate_photodiode_array, make_windows
from build_dataset import (
    DEFAULT_ZIP,
    GAZE_RANGE_X,
    GAZE_RANGE_Y,
    SESSIONS,
    TASK_DIR_MATCH,
    WINDOW,
    STRIDE,
    split_subjects,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "processed")

MAX_SHIFT_DEG = 4.0  # see psog_simulate.simulate_photodiode_array docstring


def load_task_csv(subject_zip_bytes, session):
    zs = zipfile.ZipFile(io.BytesIO(subject_zip_bytes))
    matches = [
        n for n in zs.namelist()
        if n.startswith(f"{session}/") and TASK_DIR_MATCH in n and n.endswith(".csv")
    ]
    if not matches:
        return None
    df = pd.read_csv(io.BytesIO(zs.read(matches[0])))
    df = df.loc[df["val"] == 0]
    df = df.loc[df["x"].between(*GAZE_RANGE_X) & df["y"].between(*GAZE_RANGE_Y)]
    return df[["x", "y"]].to_numpy(dtype=np.float64)


def random_shift(rng, max_deg):
    angle = rng.uniform(0, 2 * np.pi)
    mag = rng.uniform(0, max_deg)
    return np.array([mag * np.cos(angle), mag * np.sin(angle)]), mag


def main():
    zip_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ZIP
    max_shift = float(sys.argv[2]) if len(sys.argv) > 2 else MAX_SHIFT_DEG

    z = zipfile.ZipFile(zip_path)
    subject_entries = sorted(
        n for n in z.namelist() if n.startswith("Round_1/Subject_") and n.endswith(".zip")
    )
    subject_ids = [int(n.split("Subject_")[1].split(".zip")[0]) for n in subject_entries]
    split_map = split_subjects(subject_ids)
    test_ids = split_map["test"]

    rng = np.random.default_rng(12345)
    Xs, ys, sids, shift_mags = [], [], [], []

    for entry, sid in zip(subject_entries, subject_ids):
        if sid not in test_ids:
            continue
        subj_bytes = z.read(entry)
        for session in SESSIONS:
            xy = load_task_csv(subj_bytes, session)
            if xy is None or len(xy) < WINDOW:
                continue
            shift_vec, shift_mag = random_shift(rng, max_shift)
            signal, _ = simulate_photodiode_array(
                xy, n_channels=8, sensor_shift=shift_vec, seed=sid
            )
            X, y = make_windows(signal, xy, window=WINDOW, stride=STRIDE)
            Xs.append(X)
            ys.append(y)
            sids.append(np.full(len(X), sid, dtype=np.int32))
            shift_mags.append(np.full(len(X), shift_mag, dtype=np.float32))
            print(f"subject {sid} {session}: shift={shift_mag:.2f} deg, {len(X)} windows")

    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.float32)
    subject_id = np.concatenate(sids)
    shift_deg = np.concatenate(shift_mags)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "test_shifted.npz")
    np.savez_compressed(out_path, X=X, y=y, subject_id=subject_id, shift_deg=shift_deg)
    print(f"wrote {out_path}: X={X.shape}")

    with open(os.path.join(OUT_DIR, "shift_eval_manifest.json"), "w") as f:
        json.dump(
            {"max_shift_deg": max_shift, "n_windows": int(len(X)), "n_subjects": len(test_ids)},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
