#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
VWW evaluation for tf_keras or akida models.
Example
-------
    python eval.py -d /data/vww_coco2014_96/ -l akidanet_vww.h5
"""
import argparse
import numpy as np

import akida

from cnn2snn import load_quantized_model
from akida_models.sparsity import compute_sparsity

from vww_data import get_data, get_samples


def pretty_print_sparsity(sparsity_dict):
    col_w = max(len(k) for k in sparsity_dict) + 2
    print(f"\n{'Layer':<{col_w}} {'Sparsity':>10}")
    print("-" * (col_w + 11))
    for layer, sparsity in sparsity_dict.items():
        print(f"{layer:<{col_w}} {sparsity:>9.2%}")
    print("-" * (col_w + 11))
    mean_sparsity = np.mean(list(sparsity_dict.values()))
    print(f"{'Mean':<{col_w}} {mean_sparsity:>9.2%}")


# ---------------------------------------------------------------------------
# Evaluation on Akida
# ---------------------------------------------------------------------------
def evaluate_akida_model(akida_model, val_dataset):
    """Run inference with an Akida model and return (predictions, labels)."""
    labels_all = None
    logits_all = None

    # Akida can't directly digest the tensorflow dataset, we need to 
    # manually iterate over the dataset to deliver inputs as numpy arrays
    for batch, label_batch in val_dataset:
        if not isinstance(batch, np.ndarray):
            batch = batch.numpy()

        # Inference on Akida
        logits_batch = akida_model.predict(batch)
        logits_batch = logits_batch.squeeze(axis=(1, 2))  # (B, 1, 1, C) -> (B, C)

        if labels_all is None:
            labels_all = label_batch
            logits_all = logits_batch
        else:
            labels_all = np.concatenate([labels_all, label_batch])
            logits_all = np.concatenate([logits_all, logits_batch])

    preds = np.argmax(logits_all, axis=1)
    accuracy = np.mean(np.equal(np.array(preds), np.array(labels_all)))
    print(f'Akida accuracy: {accuracy:.4f}')
    return preds, labels_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras or .fbz akida model)')
    parser.add_argument('-d', '--data', default='./data/vw_coco2014_96',
                        help='VWW dataset root (contains train/ and val/ subdirs)')
    args = parser.parse_args()


    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    if args.loadmodel.endswith('.h5'):
        model = load_quantized_model(args.loadmodel)
        model.compile(metrics=['accuracy'])
        isakida = False
        imsize = model.input_shape[1:]
    elif args.loadmodel.endswith('.fbz'):
        model = akida.Model(args.loadmodel)
        isakida = True
        imsize = tuple(model.input_shape)

    # ---------------------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------------------
    train_ds, val_ds = get_data(args.data, imsize, batch_size=32)

    # ---------------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------------
    if isakida:
        preds, labels = evaluate_akida_model(model, val_ds)
    else:
        _, accuracy = model.evaluate(val_ds, verbose=0)
        print(f'Validation accuracy: {accuracy:.4f}')


    try:
        samples = get_samples(args.data, imsize, num_samples=1024)
        sparsity_dict = compute_sparsity(model, samples=samples)
        pretty_print_sparsity(sparsity_dict)
    except:
        pass
