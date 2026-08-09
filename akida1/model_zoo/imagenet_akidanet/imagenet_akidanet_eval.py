#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Evaluation of the published AkidaNet ImageNet models.

Selects one model by width multiplier and input resolution, and evaluates one of
its three variants:

  float  full-precision Keras model
  qat    quantization-aware-trained Keras model (8-bit input, 4-bit weights and
         activations)
  akida  the converted Akida model, run on hardware if a device is present and
         on the software backend otherwise

Reports ImageNet top-1 and top-5 accuracy. For the Akida variant it also reports
mean activation sparsity, which is what makes the model cheap to run on Akida --
that measurement needs no hardware, so it is done here rather than in the
benchmark script.

Examples
--------
    # Full validation set (requires the ImageNet dataset setup)
    python imagenet_akidanet_eval.py -a 1.0 -i 224 --variant float

    # No dataset setup needed: 10-image smoke test of the whole pipeline
    python imagenet_akidanet_eval.py -a 1.0 -i 224 --variant float --samples
"""
import argparse
import json
import pathlib

import numpy as np
from tqdm import tqdm

import akida
from akida_models.sparsity import compute_sparsity
from cnn2snn import load_quantized_model
from tf_keras.metrics import SparseTopKCategoricalAccuracy

from imagenet_akidanet_data import (get_data, get_labelled_samples, get_samples,
                                    index_to_label)
from imagenet_akidanet_model import (ALPHAS, RESOLUTIONS, VARIANTS, model_path,
                                     metrics_prefix)
from brainchip_utils.hardware_utils import get_akida_device

# Number of images used to measure activation sparsity
SPARSITY_SAMPLES = 100


def evaluate_keras_model(model, dataset, steps=None):
    """Evaluates a Keras model, returning (top-1, top-5) accuracy."""
    model.compile(metrics=['accuracy', SparseTopKCategoricalAccuracy(k=5, name='top5')])
    results = model.evaluate(dataset, steps=steps, verbose=1, return_dict=True)
    return results['accuracy'], results['top5']


def evaluate_akida_model(akida_model, dataset, num_samples=None):
    """Runs inference with an Akida model, returning (top-1, top-5) accuracy.

    Akida cannot consume a tf.data pipeline directly, so batches are pulled out
    as numpy arrays and fed one at a time.
    """
    device = get_akida_device(target_version=akida_model.ip_version)
    if device is not None:
        akida_model.map(device, mode=akida.MapMode.Minimal)
        print('Running inference on Akida hardware device')
        akida_model.summary()
    else:
        print('No Akida device found - running on the software backend')

    correct_1 = 0
    correct_5 = 0
    seen = 0

    for batch, labels in tqdm(dataset, desc='Evaluating on Akida'):
        batch = batch.numpy() if hasattr(batch, 'numpy') else batch
        labels = labels.numpy() if hasattr(labels, 'numpy') else labels

        logits = akida_model.predict(batch)
        logits = logits.squeeze(axis=(1, 2))  # (B, 1, 1, C) -> (B, C)

        correct_1 += int(np.sum(np.argmax(logits, axis=-1) == labels))
        # Top-5: the 5 highest logits per row, order within them is irrelevant
        top5 = np.argpartition(logits, -5, axis=-1)[:, -5:]
        correct_5 += int(np.sum([labels[i] in top5[i] for i in range(len(labels))]))
        seen += len(labels)

        if num_samples is not None and seen >= num_samples:
            break

    return correct_1 / seen, correct_5 / seen, seen


def run_smoke_test(model, isakida, input_shape):
    """Evaluates the 10-image sample pack and prints per-image predictions."""
    images, labels = get_labelled_samples(input_shape)

    if isakida:
        logits = model.predict(images).squeeze(axis=(1, 2))
    else:
        logits = model.predict(images, verbose=0)

    preds = np.argmax(logits, axis=-1)
    top5 = np.argsort(logits, axis=-1)[:, -5:][:, ::-1]

    print('\nSample predictions:')
    for i in range(len(images)):
        mark = 'OK  ' if preds[i] == labels[i] else 'MISS'
        print(f'  {mark} true={index_to_label(labels[i])[:36]:<36} '
              f'pred={index_to_label(preds[i])[:36]}')

    top1_acc = float(np.mean(preds == labels))
    top5_acc = float(np.mean([labels[i] in top5[i] for i in range(len(labels))]))
    return top1_acc, top5_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a published AkidaNet ImageNet model')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, choices=ALPHAS,
                        help='Width multiplier. Defaults to %(default)s.')
    parser.add_argument('-i', '--input-resolution', type=int, default=224,
                        choices=RESOLUTIONS, dest='resolution',
                        help='Input resolution. Defaults to %(default)s.')
    parser.add_argument('--variant', default='float', choices=VARIANTS,
                        help='Model variant to evaluate. Defaults to %(default)s.')
    parser.add_argument('-d', '--data', default='./data/imagenet_tfds',
                        help='ImageNet directory (containing the tfds store)')
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-n', '--num-samples', type=int, default=None,
                        help='Evaluate only the first N validation images')
    parser.add_argument('--samples', action='store_true',
                        help='Use the 10-image sample pack instead of the full '
                             'validation set. A pipeline smoke test, not an '
                             'accuracy measurement.')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Write accuracy (and sparsity/params) to docs/metrics.json')
    args = parser.parse_args()

    loadmodel = model_path(args.alpha, args.resolution, args.variant)
    if not loadmodel.exists():
        raise FileNotFoundError(
            f'{loadmodel} not found. These weights are tracked with Git LFS - '
            'run `git lfs pull` to fetch them.')
    print(f'Evaluating {loadmodel.name}  '
          f'(alpha={args.alpha}, {args.resolution}x{args.resolution}, {args.variant})')

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    isakida = args.variant == 'akida'
    if isakida:
        model = akida.Model(str(loadmodel))
        input_shape = tuple(model.input_shape)
    else:
        model = load_quantized_model(str(loadmodel))
        # The 224 checkpoints carry a stale internal name saying '160' (they were
        # produced by rescaling the 160 models), so trust input_shape, not name.
        input_shape = model.input_shape[1:]

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    if args.samples:
        top1, top5 = run_smoke_test(model, isakida, input_shape)
        num_evaluated = 10
        print(f'\nSmoke test ({num_evaluated} images): '
              f'top-1 {top1 * 100:.1f}%, top-5 {top5 * 100:.1f}%')
        print('This is a pipeline check, not an accuracy measurement.')
    else:
        dataset, num_examples = get_data(args.data, input_shape, args.batch_size)
        num_evaluated = min(args.num_samples or num_examples, num_examples)

        if isakida:
            top1, top5, num_evaluated = evaluate_akida_model(
                model, dataset, num_samples=args.num_samples)
        else:
            steps = None
            if args.num_samples is not None:
                steps = int(np.ceil(args.num_samples / args.batch_size))
                num_evaluated = steps * args.batch_size
            top1, top5 = evaluate_keras_model(model, dataset, steps=steps)

        print(f'\n{args.variant} accuracy over {num_evaluated} images: '
              f'top-1 {top1 * 100:.2f}%, top-5 {top5 * 100:.2f}%')

    # -------------------------------------------------------------------------
    # Activation sparsity (Akida variant only)
    # -------------------------------------------------------------------------
    sparsity = None
    if isakida:
        samples = get_samples(input_shape, num_samples=SPARSITY_SAMPLES,
                              data_path=None if args.samples else args.data)
        sparsity_dict = compute_sparsity(model, samples=samples)
        sparsity = float(np.mean(list(sparsity_dict.values())))
        print(f'Mean activation sparsity: {sparsity * 100:.2f}%')

    # -------------------------------------------------------------------------
    # Persist metrics
    # -------------------------------------------------------------------------
    if args.save_metrics:
        # Used to update the stored metrics behind the README performance tables.
        # This is a maintenance step, run when the models or pipeline change.
        if args.samples:
            raise SystemExit(
                '--save-metrics refuses to record a 10-image smoke test. '
                'Run against the validation set instead.')

        metrics_path = pathlib.Path(__file__).parent / 'docs' / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        prefix = metrics_prefix(args.alpha, args.resolution)

        metrics[f'{prefix}{args.variant}_t1'] = f'{top1 * 100:.2f}%'
        metrics[f'{prefix}{args.variant}_t5'] = f'{top5 * 100:.2f}%'
        if sparsity is not None:
            metrics[f'{prefix}sparsity'] = f'{sparsity * 100:.2f}%'
        if args.variant == 'float':
            metrics[f'{prefix}params'] = f'{model.count_params():,}'
        # Record how many images each variant was scored on, so the README can
        # be honest about any run that did not cover the full split.
        metrics[f'{prefix}{args.variant}_n'] = f'{num_evaluated}'

        metrics_path.write_text(json.dumps(metrics, indent=4, sort_keys=True) + '\n')
        print(f'Metrics saved to {metrics_path}')
