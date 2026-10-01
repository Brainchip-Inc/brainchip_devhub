"""Build windowed (photodiode-signal, gaze) training data from GazeBase.

Scope for this first pass: Round_1 (all 322 subjects have data here; later
rounds are a longitudinal subset), both sessions, the Random_Saccades task —
the closest paradigm to what PSOG/photodiode hardware papers use for
calibration (discrete step saccades to known targets), and simple to reason
about before adding naturalistic tasks (reading, video free-viewing).

Splits by SUBJECT (not by window) so no subject appears in more than one
split - required to get a meaningful validation signal, since windows from
the same recording are highly correlated.

Usage:
    python build_dataset.py [path/to/GazeBase_v2_0.zip]
"""

import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

from psog_simulate import simulate_photodiode_array, make_windows

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ZIP = os.path.join(HERE, "raw", "GazeBase_v2_0.zip")
OUT_DIR = os.path.join(HERE, "processed")

TASK_DIR_MATCH = "Random_Saccades"
SESSIONS = ["S1", "S2"]
WINDOW = 64
STRIDE = 32
TRAIN_FRAC, VAL_FRAC = 0.7, 0.15  # remainder -> test

# The Random_Saccades task's actual stimulus grid spans x in [-14.7, 15.3],
# y in [-9.4, 8.6] degrees (measured directly from the xT/yT target columns,
# consistent across a 15-subject sample). Recorded gaze well outside that -
# some subjects reach +-40-51 deg - is measurement artifact (overshoot
# dynamics or track loss the `val` flag didn't catch), not real signal.
# Small margin added beyond the target extent to keep legitimate overshoot.
GAZE_RANGE_X = (-16.0, 16.0)
GAZE_RANGE_Y = (-11.0, 11.0)


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
    df = df.loc[
        df["x"].between(*GAZE_RANGE_X) & df["y"].between(*GAZE_RANGE_Y)
    ]
    return df[["x", "y"]].to_numpy(dtype=np.float64)


def split_subjects(subject_ids):
    ids = sorted(subject_ids)
    n = len(ids)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    return {
        "train": set(ids[:n_train]),
        "val": set(ids[n_train : n_train + n_val]),
        "test": set(ids[n_train + n_val :]),
    }


def main():
    zip_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ZIP
    z = zipfile.ZipFile(zip_path)
    subject_entries = sorted(
        n for n in z.namelist() if n.startswith("Round_1/Subject_") and n.endswith(".zip")
    )
    subject_ids = [int(n.split("Subject_")[1].split(".zip")[0]) for n in subject_entries]
    split_map = split_subjects(subject_ids)
    id_to_split = {sid: split for split, ids in split_map.items() for sid in ids}

    buckets = {"train": [], "val": [], "test": []}
    skipped = []

    for entry, sid in zip(subject_entries, subject_ids):
        subj_split = id_to_split[sid]
        subj_bytes = z.read(entry)
        n_windows_this_subject = 0
        for session in SESSIONS:
            xy = load_task_csv(subj_bytes, session)
            if xy is None or len(xy) < WINDOW:
                skipped.append((sid, session))
                continue
            signal, _ = simulate_photodiode_array(xy, n_channels=8, seed=sid)
            X, y = make_windows(signal, xy, window=WINDOW, stride=STRIDE)
            sid_col = np.full(len(X), sid, dtype=np.int32)
            buckets[subj_split].append((X, y, sid_col))
            n_windows_this_subject += len(X)
        print(f"subject {sid} ({subj_split}): {n_windows_this_subject} windows")

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {"window": WINDOW, "stride": STRIDE, "task": TASK_DIR_MATCH, "sessions": SESSIONS}
    for split, parts in buckets.items():
        if not parts:
            continue
        X = np.concatenate([p[0] for p in parts])
        y = np.concatenate([p[1] for p in parts])
        subject_id = np.concatenate([p[2] for p in parts])
        out_path = os.path.join(OUT_DIR, f"{split}.npz")
        np.savez_compressed(out_path, X=X.astype(np.float32), y=y.astype(np.float32), subject_id=subject_id)
        manifest[split] = {
            "n_windows": int(len(X)),
            "n_subjects": len(split_map[split]),
            "path": out_path,
        }
        print(f"{split}: X={X.shape} y={y.shape} -> {out_path}")

    manifest["skipped"] = skipped
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"skipped {len(skipped)} (subject, session) pairs with missing/short data")


if __name__ == "__main__":
    main()
