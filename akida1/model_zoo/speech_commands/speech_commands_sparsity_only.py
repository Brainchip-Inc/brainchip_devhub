#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Software-only sparsity measurement for a Speech Commands model (float .h5,
quantized .h5, or Akida .fbz). Does not require Akida hardware --
compute_sparsity runs inference through the software backend.

Use this in place of speech_commands_benchmark.py when no Akida device is
connected; it gives the accuracy/sparsity half of the trade-off, without
hardware latency/power numbers.

Example
-------
    python speech_commands_sparsity_only.py -l models/speech_commands_qat.fbz
"""
import argparse

from akida_models.sparsity import compute_sparsity

from speech_commands_data_loader import compute_mfcc_range, get_samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras or .fbz akida model)')
    parser.add_argument('-d', '--data', default='./data/sc10',
                        help='Speech Commands tfds data directory')
    parser.add_argument('-n', '--num_samples', type=int, default=1000)
    args = parser.parse_args()

    if args.loadmodel.endswith('.fbz'):
        import akida
        model = akida.Model(args.loadmodel)
    else:
        from cnn2snn import load_quantized_model
        model = load_quantized_model(args.loadmodel)

    data_transform = compute_mfcc_range(data_dir=args.data)
    samples = get_samples(args.data, data_transform=data_transform, num_samples=args.num_samples)
    sparsity_dict = compute_sparsity(model, samples=samples, batch_size=args.num_samples)

    for layer, sparsity in sparsity_dict.items():
        print(f'{layer} : {sparsity:.4f}')
    mean_sparsity = sum(sparsity_dict.values()) / len(sparsity_dict)
    print(f'Mean sparsity: {mean_sparsity:.4f}')
