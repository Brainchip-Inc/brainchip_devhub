# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""Shared float<->uint8 input scaling for the ECG Arrhythmia model.

Each preprocessed sample stacks two very different value ranges into one image:
the CWT scalogram rows (min-max normalized to [0, 1]) and the appended
RR-interval feature rows (z-scored, mean 0 std 1, with real tails). The combined
range is NOT [-1, 1] -- that was the pre-existing eval.py's assumption
(inherited from a template) and silently corrupts accuracy on converted Akida
models, since most of the uint8 dynamic range gets wasted / values saturate.

Bounds below are the training split's actual min/max (data/processed/X_train.npy,
measured directly -- see git history for the one-off computation). Frozen from
train, like any other train-time normalization statistic; a few test-set values
fall outside this range (one extreme RR interval reaches 24.9) and get clipped,
same as real deployment would do.

cnn2snn's `input_scaling=(scale, shift)` convention: input_akida = scale *
input_float + shift. Every script that converts to or evaluates a converted
Akida model must use these same two constants, or the float<->uint8 mapping
goes inconsistent between conversion and inference.
"""
import numpy as np

MIN_V = -5.592949441790575
MAX_V = 14.829335506403263

SCALE = 255.0 / (MAX_V - MIN_V)
SHIFT = -MIN_V * SCALE


def to_uint8(x):
    """Map a float array in the model's input range to uint8 [0, 255]."""
    return np.clip(np.round(x * SCALE + SHIFT), 0, 255).astype(np.uint8)
