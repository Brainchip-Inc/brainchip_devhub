"""Shared helper for evaluating a bufferized (BufferTempConv-based) streaming
model - one timestep fed in at a time, with internal FIFO state persisting
across the loop for a batch of independent windows, reset before the next
batch. Used for both the quantized Keras model and the final Akida model.
"""

import numpy as np


def predict_streaming(step_fn, reset_fn, X_windows, batch_size=2048):
    """Run a bufferized model over (N, T, 1, 1, C) windows, one timestep at a
    time, returning (N, T, 2) predictions.

    Args:
        step_fn: callable(frame) -> prediction, frame is (b, 1, 1, C)
        reset_fn: callable() that resets the model's FIFO state
        X_windows (np.ndarray): (N, T, 1, 1, C) uint8 input windows
        batch_size (int): number of independent windows processed per batch
    """
    n, T = X_windows.shape[0], X_windows.shape[1]
    outputs = np.zeros((n, T, 2), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        reset_fn()
        for t in range(T):
            frame = X_windows[start:end, t]
            pred = step_fn(frame)
            outputs[start:end, t] = np.asarray(pred).reshape(end - start, 2)
    return outputs
