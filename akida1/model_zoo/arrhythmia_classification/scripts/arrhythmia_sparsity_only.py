#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Software-only sparsity measurement for an ECG Arrhythmia Classification model
(float .h5, quantized .h5, or Akida .fbz). Does not require Akida hardware --
compute_sparsity runs inference through the software backend.

Use this in place of a hardware benchmark when no Akida device is connected;
it gives the accuracy/sparsity half of the trade-off, without hardware
latency/power numbers.

Example
-------
    python arrhythmia_sparsity_only.py -l model/3class_arrhythmia_classification/arrhythmia_classification.fbz \\
        -d data/processed
"""
import argparse

import numpy as np

from akida_models.sparsity import compute_sparsity

from data import ECGDatasetLoader
from input_scaling import to_uint8

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras or .fbz akida model)')
    parser.add_argument('-d', '--data_dir', default='./data/processed',
                        help='Path to preprocessed .npy arrays (ECGDatasetBuilder output)')
    parser.add_argument('-n', '--num_samples', type=int, default=1000)
    args = parser.parse_args()

    _, _, _, _, X_test, _ = ECGDatasetLoader(data_dir=args.data_dir).load_dataset()

    rng = np.random.default_rng(42)
    num_samples = min(args.num_samples, len(X_test))
    idx = rng.choice(len(X_test), size=num_samples, replace=False)
    samples = X_test[idx]

    if args.loadmodel.endswith('.fbz'):
        import akida
        model = akida.Model(args.loadmodel)
        samples = to_uint8(samples)
    else:
        from cnn2snn import load_quantized_model
        model = load_quantized_model(args.loadmodel)

    sparsity_dict = compute_sparsity(model, samples=samples, batch_size=num_samples)

    for layer, sparsity in sparsity_dict.items():
        print(f'{layer} : {sparsity:.4f}')
    mean_sparsity = sum(sparsity_dict.values()) / len(sparsity_dict)
    print(f'Mean sparsity: {mean_sparsity:.4f}')
