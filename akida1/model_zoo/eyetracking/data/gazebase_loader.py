"""Load a single GazeBase recording CSV.

GazeBase (Griffith, Lohr, Abdulin & Komogortsev, Scientific Data 2021) ships
one CSV per recording with columns:
  n    - sample index / timestamp (ms, 1000 Hz)
  x, y - gaze position, degrees of visual angle
  val  - validity flag (0 = valid, nonzero = blink/off-screen/loss of track)
  xT, yT - target position (stimulus-dependent, may be blank)
  dP   - uncalibrated pupil area
  lab  - eye-movement label (fixation/saccade/blink/...)
"""

import numpy as np
import pandas as pd


def load_recording(path):
    """Return (t_ms, xy) for one GazeBase CSV, with invalid (blink/lost-track)
    samples dropped."""
    df = pd.read_csv(path)
    valid = df["val"] == 0
    df = df.loc[valid]
    t_ms = df["n"].to_numpy(dtype=np.float64)
    xy = df[["x", "y"]].to_numpy(dtype=np.float64)
    return t_ms, xy
