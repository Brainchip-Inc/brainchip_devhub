#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Float-model activation sparsity measurement for the ECG Arrhythmia
Classification model.

arrhythmia_sparsity_only.py measures sparsity via
akida_models.sparsity.compute_sparsity, which requires a quantized/Akida
model and runs inference through a software backend (see
SPARSITY_EXPERIMENT.md's note on why those numbers aren't directly
comparable to hardware-measured sparsity). The joint activation+structured
sparsity experiment stays float-only end to end, so this instead measures
the same "mean fraction of zero activations across ReLU layers" statistic
directly on a float tf_keras model's ReLU outputs -- no quantization/
conversion involved. Adapted from ../../vww/vww_float_sparsity.py.
"""
import numpy as np
from tf_keras import Model
from tf_keras.layers import ReLU


def compute_float_sparsity(model, samples, batch_size=100):
    """Mean fraction of exactly-zero activations across every ReLU layer.

    Args:
        model (tf_keras.Model): float model to probe.
        samples (np.ndarray): input samples, matching model.input_shape.
        batch_size (int): batch size for the forward pass.

    Returns:
        dict, float: per-layer sparsity keyed by layer name, and the mean across layers.
    """
    relu_layers = [layer for layer in model.layers if isinstance(layer, ReLU)]
    probe_model = Model(model.input, [layer.output for layer in relu_layers])

    sums = np.zeros(len(relu_layers))
    counts = np.zeros(len(relu_layers))
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        outputs = probe_model.predict(batch, verbose=0)
        if len(relu_layers) == 1:
            outputs = [outputs]
        for i, out in enumerate(outputs):
            sums[i] += np.sum(out == 0)
            counts[i] += out.size

    per_layer = {layer.name: sums[i] / counts[i] for i, layer in enumerate(relu_layers)}
    mean_sparsity = sum(per_layer.values()) / len(per_layer)
    return per_layer, mean_sparsity
