#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Software-only sparsity measurement for a PlantVillage model (float .h5,
quantized .h5, or Akida .fbz). Does not require Akida hardware --
compute_sparsity runs inference through the software backend.

Use this in place of plant_village_benchmark.py when no Akida device is
connected; it gives the accuracy/sparsity half of the trade-off, without
hardware latency/power numbers.

Example
-------
    python plant_village_sparsity_only.py -l models/akidanet_plant_village_qat.fbz
"""
import argparse

from akida_models.sparsity import compute_sparsity

from plant_village_data import get_samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras or .fbz akida model)')
    parser.add_argument('-d', '--data', default='./data/plant_village',
                        help='PlantVillage tfds data directory')
    parser.add_argument('-n', '--num_samples', type=int, default=1000)
    args = parser.parse_args()

    if args.loadmodel.endswith('.fbz'):
        import akida
        model = akida.Model(args.loadmodel)
        imsize = tuple(model.input_shape)
    else:
        from cnn2snn import load_quantized_model
        model = load_quantized_model(args.loadmodel)
        imsize = model.input_shape[1:]

    samples = get_samples(args.data, imsize, num_samples=args.num_samples)
    sparsity_dict = compute_sparsity(model, samples=samples, batch_size=args.num_samples)

    for layer, sparsity in sparsity_dict.items():
        print(f'{layer} : {sparsity:.4f}')
    mean_sparsity = sum(sparsity_dict.values()) / len(sparsity_dict)
    print(f'Mean sparsity: {mean_sparsity:.4f}')
