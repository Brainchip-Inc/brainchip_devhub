#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Float-model activation sparsity measurement for PlantVillage.

plant_village_sparsity_only.py measures sparsity via
akida_models.sparsity.compute_sparsity, which requires a quantized/Akida
model and runs inference through a software backend (see
SPARSITY_EXPERIMENT.md's note on why those numbers aren't directly
comparable to hardware-measured sparsity). The joint activation+structured
sparsity experiment stays float-only end to end, so this instead measures
the same "mean fraction of zero activations across ReLU layers" statistic
directly on a float tf_keras model's ReLU outputs -- no quantization/
conversion involved. Ported unchanged from ../vww/vww_float_sparsity.py.

Example
-------
    python plant_village_float_sparsity.py -l models/akidanet_plant_village.h5 \\
        -d ./data/plant_village
"""
import argparse

import numpy as np
from tf_keras import Model
from tf_keras.layers import ReLU

from plant_village_data import get_samples


def compute_float_sparsity(model, samples, batch_size=100):
    """Mean fraction of exactly-zero activations across every ReLU layer.

    Args:
        model (tf_keras.Model): float model to probe.
        samples (np.ndarray): input images, matching model.input_shape.
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Float model to load (.h5 tf_keras)')
    parser.add_argument('-d', '--data', default='./data/plant_village',
                        help='PlantVillage tfds data directory')
    parser.add_argument('-n', '--num_samples', type=int, default=1000)
    args = parser.parse_args()

    from cnn2snn import load_quantized_model
    model = load_quantized_model(args.loadmodel)

    samples = get_samples(args.data, model.input_shape[1:], num_samples=args.num_samples)
    per_layer, mean_sparsity = compute_float_sparsity(model, samples)

    for layer, sparsity in per_layer.items():
        print(f'{layer} : {sparsity:.4f}')
    print(f'Mean sparsity: {mean_sparsity:.4f}')
