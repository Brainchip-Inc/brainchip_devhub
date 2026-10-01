"""Build shift-AUGMENTED train/val sets: same subjects/windows as
build_dataset.py, but each (subject, session) recording gets one random
sensor-array shift (magnitude uniform in [0, MAX_SHIFT_DEG], random
direction) applied throughout - so the model sees many different shift
conditions during training instead of only the idealized zero-shift case.

Test subjects are untouched here; use build_shift_eval.py's test_shifted.npz
(and the existing test.npz) for evaluation, so shift-augmented and
zero-shift-trained models are compared on identical held-out data.

Usage:
    python build_dataset_shift_aug.py [path/to/GazeBase_v2_0.zip] [max_shift_deg]
"""

import os
import sys
import zipfile

import numpy as np

from psog_simulate import simulate_photodiode_array, make_windows
from build_dataset import DEFAULT_ZIP, SESSIONS, WINDOW, STRIDE, split_subjects, load_task_csv
from build_shift_eval import random_shift, MAX_SHIFT_DEG

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "processed")


def build_split(z, ids, rng):
    Xs, ys, sids = [], [], []
    for sid in sorted(ids):
        entry = f"Round_1/Subject_{sid}.zip"
        subj_bytes = z.read(entry)
        for session in SESSIONS:
            xy = load_task_csv(subj_bytes, session)
            if xy is None or len(xy) < WINDOW:
                continue
            shift_vec, shift_mag = random_shift(rng, MAX_SHIFT_DEG)
            signal, _ = simulate_photodiode_array(xy, n_channels=8, sensor_shift=shift_vec, seed=sid)
            X, y = make_windows(signal, xy, window=WINDOW, stride=STRIDE)
            Xs.append(X)
            ys.append(y)
            sids.append(np.full(len(X), sid, dtype=np.int32))
        print(f"subject {sid}: done")
    return (
        np.concatenate(Xs).astype(np.float32),
        np.concatenate(ys).astype(np.float32),
        np.concatenate(sids),
    )


def main():
    zip_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ZIP
    z = zipfile.ZipFile(zip_path)
    subject_entries = sorted(
        n for n in z.namelist() if n.startswith("Round_1/Subject_") and n.endswith(".zip")
    )
    subject_ids = [int(n.split("Subject_")[1].split(".zip")[0]) for n in subject_entries]
    split_map = split_subjects(subject_ids)

    os.makedirs(OUT_DIR, exist_ok=True)
    for split_name, seed in [("train", 777), ("val", 778)]:
        rng = np.random.default_rng(seed)
        X, y, subject_id = build_split(z, split_map[split_name], rng)
        out_path = os.path.join(OUT_DIR, f"{split_name}_shift.npz")
        np.savez_compressed(out_path, X=X, y=y, subject_id=subject_id)
        print(f"{split_name}_shift: X={X.shape} -> {out_path}")


if __name__ == "__main__":
    main()
