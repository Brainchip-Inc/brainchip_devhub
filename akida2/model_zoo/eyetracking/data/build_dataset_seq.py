"""Build the per-timestep ("sequence") variant of the windowed dataset for
the causal/streaming Akida model: same X windows as build_dataset.py, but y
is the full (window, 2) gaze trajectory instead of just the last sample.

Usage:
    python build_dataset_seq.py [path/to/GazeBase_v2_0.zip]
"""

import json
import os
import sys
import zipfile

import numpy as np

from psog_simulate import simulate_photodiode_array, make_windows_seq
from build_dataset import DEFAULT_ZIP, SESSIONS, WINDOW, STRIDE, split_subjects, load_task_csv

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "processed")


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

    for entry, sid in zip(subject_entries, subject_ids):
        subj_split = id_to_split[sid]
        subj_bytes = z.read(entry)
        n_windows_this_subject = 0
        for session in SESSIONS:
            xy = load_task_csv(subj_bytes, session)
            if xy is None or len(xy) < WINDOW:
                continue
            signal, _ = simulate_photodiode_array(xy, n_channels=8, seed=sid)
            X, y = make_windows_seq(signal, xy, window=WINDOW, stride=STRIDE)
            buckets[subj_split].append((X, y))
            n_windows_this_subject += len(X)
        print(f"subject {sid} ({subj_split}): {n_windows_this_subject} windows")

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = {"window": WINDOW, "stride": STRIDE, "target": "per_timestep_sequence"}
    for split, parts in buckets.items():
        if not parts:
            continue
        X = np.concatenate([p[0] for p in parts]).astype(np.float32)
        y = np.concatenate([p[1] for p in parts]).astype(np.float32)
        out_path = os.path.join(OUT_DIR, f"{split}_seq.npz")
        np.savez_compressed(out_path, X=X, y=y)
        manifest[split] = {"n_windows": int(len(X)), "path": out_path}
        print(f"{split}_seq: X={X.shape} y={y.shape} -> {out_path}")

    with open(os.path.join(OUT_DIR, "manifest_seq.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
