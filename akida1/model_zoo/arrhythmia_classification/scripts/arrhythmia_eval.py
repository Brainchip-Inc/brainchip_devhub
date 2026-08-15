#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Lightweight ECG Arrhythmia Classification evaluation for tf_keras (float/quantized
.h5) or Akida (.fbz) models, against the already-preprocessed test split. Unlike
eval.py (the full hardware benchmarking pipeline), this doesn't re-run raw-signal
preprocessing or require Akida hardware -- it's meant for quick accuracy checks in
a sparsity sweep (see arrhythmia_sparsity_sweep.py).

Example
-------
    python arrhythmia_eval.py -l model/3class_arrhythmia_classification/arrhythmia_classification.fbz \\
        -d data/processed
"""
import argparse

import numpy as np

from data import ECGDatasetLoader
from input_scaling import to_uint8


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras or .fbz akida model)')
    parser.add_argument('-d', '--data_dir', default='./data/processed',
                        help='Path to preprocessed .npy arrays (ECGDatasetBuilder output)')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    _, _, _, _, X_test, y_test = ECGDatasetLoader(data_dir=args.data_dir).load_dataset()

    if args.loadmodel.endswith('.fbz'):
        import akida
        model = akida.Model(args.loadmodel)
        X_test = to_uint8(X_test)

        outputs = []
        for idx in range(0, len(X_test), args.batch_size):
            batch_out = model.predict(X_test[idx:idx + args.batch_size])
            outputs.append(batch_out.squeeze(axis=(1, 2)))
        logits = np.concatenate(outputs, axis=0)
        preds = np.argmax(logits, axis=1)
        accuracy = float(np.mean(preds == y_test))
        print(f'Akida accuracy: {accuracy:.4f}')
    else:
        from cnn2snn import load_quantized_model
        model = load_quantized_model(args.loadmodel)
        preds = np.argmax(model.predict(X_test, batch_size=args.batch_size), axis=1)
        accuracy = float(np.mean(preds == y_test))
        print(f'Test accuracy: {accuracy:.4f}')
