#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Evaluation of the MobileNetV1 ImageNet models, at every step of the pipeline.

Selects one model by width multiplier and evaluates one of its four variants:

  float      the timm PyTorch model (on GPU if available)
  onnx       the float model exported to ONNX, run with onnxruntime
  quantized  the 8-bit quantizeml model, run with onnxruntime
  akida      the converted Akida model, run on hardware if a device is present
             and on the software backend otherwise

Reports ImageNet top-1 and top-5 accuracy. For the Akida variant it also reports
mean activation sparsity over the ReLU layers, which is what makes the model cheap
to run on Akida -- that measurement needs no hardware, so it is done here rather
than in the benchmark script.

The float variant, and any run on the full validation set, need PyTorch (see
Requirements in the README). The 10-image smoke test of the onnx, quantized and akida variants does not.

Examples
--------
    # Full validation set (requires the ImageNet dataset setup)
    python imagenet_mobilenet_eval.py -a 1.0 --variant akida -d /path/to/imagenet

    # No dataset setup needed: 10-image smoke test of the whole pipeline
    python imagenet_mobilenet_eval.py -a 1.0 --variant akida --samples
"""
import os
# akida_models imports TensorFlow, which by default reserves almost all GPU memory
# as soon as it starts, leaving none for PyTorch. TensorFlow does no GPU work in
# this example, so let it allocate only what it uses. Set before TensorFlow loads.
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

import argparse
import json
import pathlib

import akida
import numpy as np
import onnx
import onnxruntime
from akida_models.sparsity import compute_sparsity
from tqdm import tqdm

from imagenet_mobilenet_data import (get_data, get_labelled_samples, get_samples,
                                     index_to_label)
from imagenet_mobilenet_model import (ALPHAS, VARIANTS, TIMM_NAMES, TIMM_REFERENCE,
                                      load_model, metrics_prefix)
from brainchip_utils.hardware_utils import get_akida_device

# Number of images used to measure activation sparsity
SPARSITY_SAMPLES = 100


def get_predictor(model):
    """Returns a function mapping a numpy batch (B, 3, H, W) to logits (B, 1000).

    Hides the differences between the three runtimes: onnxruntime (onnx and
    quantized), Akida and PyTorch (float). Each is set up once, here, rather than
    per batch.
    """
    if isinstance(model, onnx.ModelProto):
        session = onnxruntime.InferenceSession(model.SerializeToString())
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        def predict(images):
            return session.run([output_name], {input_name: images})[0]
        return predict

    if isinstance(model, akida.Model):
        device = get_akida_device(target_version=model.ip_version)
        if device is not None:
            model.map(device, mode=akida.MapMode.Minimal)
            print('Running inference on Akida hardware device')
        else:
            print('No Akida device found - running on the software backend')

        def predict(images):
            outputs = model.predict(images.astype(np.uint8))
            return outputs.reshape(outputs.shape[0], -1)  # (B, 1, 1, C) -> (B, C)
        return predict

    # Otherwise the float timm model, a torch.nn.Module. Only this variant needs
    # PyTorch, so it is imported here.
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()

    def predict(images):
        with torch.inference_mode():
            return model(torch.from_numpy(images).to(device)).cpu().numpy()
    return predict


def evaluate_model(model, loader):
    """Evaluates a model over a DataLoader, returning (top-1, top-5, num_images)."""
    predict = get_predictor(model)
    correct_1, correct_5, seen = 0, 0, 0
    for images, labels in tqdm(loader, desc='ImageNet validation'):
        logits = predict(images.numpy())
        labels = labels.numpy()
        correct_1 += int(np.sum(np.argmax(logits, axis=-1) == labels))
        # Top-5: the 5 highest logits per row, order within them is irrelevant
        top5 = np.argpartition(logits, -5, axis=-1)[:, -5:]
        correct_5 += int(np.sum(top5 == labels[:, None]))
        seen += len(labels)
    return correct_1 / seen, correct_5 / seen, seen


def run_smoke_test(model, alpha, variant):
    """Evaluates the 10-image sample pack and prints per-image predictions."""
    images, labels = get_labelled_samples(alpha, normalize=variant != 'akida')
    logits = get_predictor(model)(images)

    preds = np.argmax(logits, axis=-1)
    top5 = np.argsort(logits, axis=-1)[:, -5:]

    print('\nSample predictions:')
    for i in range(len(images)):
        mark = 'OK  ' if preds[i] == labels[i] else 'MISS'
        print(f'  {mark} true={index_to_label(labels[i])[:36]:<36} '
              f'pred={index_to_label(preds[i])[:36]}')

    top1_acc = float(np.mean(preds == labels))
    top5_acc = float(np.mean(np.any(top5 == labels[:, None], axis=-1)))
    return top1_acc, top5_acc


def activation_layer_names(akida_model):
    """Names of the Akida layers that end in an activation (ReLU)."""
    names = []
    for layer in akida_model.layers:
        try:
            if layer.parameters.activation != 0:
                names.append(layer.name)
        except ValueError:
            # Layer types without an activation parameter, e.g. Quantizer
            pass
    return names


def mean_activation_sparsity(akida_model, sparsity_dict):
    """Mean sparsity over the layers that end in a ReLU.

    In MobileNetV1 that is every convolution: the stem and each depthwise and
    pointwise convolution end in a ReLU6. The classifier has no activation, so
    its signed outputs are almost never exactly zero and would only pull the
    mean down. Keep the full ``sparsity_dict`` for per-layer plots.
    """
    return float(np.mean([sparsity_dict[name] for name in activation_layer_names(akida_model)
                          if name in sparsity_dict]))


def count_params(model):
    """Number of parameters of the float PyTorch model."""
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a MobileNetV1 ImageNet model')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, choices=ALPHAS,
                        help='Width multiplier. Defaults to %(default)s.')
    parser.add_argument('--variant', default='akida', choices=VARIANTS,
                        help='Model variant to evaluate. Defaults to %(default)s.')
    parser.add_argument('-d', '--data', default='./data/imagenet',
                        help='ImageNet root, containing a validation/ folder with '
                             'one subfolder per class')
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-n', '--num-samples', type=int, default=None,
                        help='Evaluate only N validation images, drawn at random '
                             '(with a fixed seed) so that every class is represented')
    parser.add_argument('-m', '--models-dir', default=None,
                        help='Directory holding the models. Defaults to pretrained_models/')
    parser.add_argument('--samples', action='store_true',
                        help='Use the 10-image sample pack instead of the full '
                             'validation set. A pipeline smoke test, not an '
                             'accuracy measurement.')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Write accuracy (and sparsity/params) to docs/metrics.json')
    args = parser.parse_args()

    if args.save_metrics and (args.samples or args.num_samples is not None):
        # Checked up front, so a long evaluation is not wasted
        raise SystemExit('--save-metrics only records full validation set runs '
                         '(without --samples or --num-samples).')

    print(f'Evaluating {TIMM_NAMES[args.alpha]} (alpha={args.alpha}), {args.variant} variant')
    model = load_model(args.alpha, args.variant, args.models_dir)

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    if args.samples:
        top1, top5 = run_smoke_test(model, args.alpha, args.variant)
        print(f'\nSmoke test (10 images): top-1 {top1 * 100:.1f}%, top-5 {top5 * 100:.1f}%')
        print('This is a pipeline check, not an accuracy measurement.')
    else:
        loader, _ = get_data(args.data, args.alpha, batch_size=args.batch_size,
                             normalize=args.variant != 'akida',
                             num_samples=args.num_samples,
                             seed=None if args.num_samples is None else 0)
        top1, top5, num_evaluated = evaluate_model(model, loader)
        print(f'\n{args.variant} accuracy over {num_evaluated} images: '
              f'top-1 {top1 * 100:.2f}%, top-5 {top5 * 100:.2f}%')
        ref_top1, ref_top5 = TIMM_REFERENCE[args.alpha]
        print(f'timm reference (float): top-1 {ref_top1:.2f}%, top-5 {ref_top5:.2f}%')

    # -------------------------------------------------------------------------
    # Activation sparsity (Akida variant only)
    # -------------------------------------------------------------------------
    sparsity = None
    if args.variant == 'akida':
        samples = get_samples(args.alpha, num_samples=SPARSITY_SAMPLES,
                              data_path=None if args.samples else args.data)
        sparsity_dict = compute_sparsity(model, samples=samples)
        sparsity = mean_activation_sparsity(model, sparsity_dict)
        print(f'Mean activation sparsity (ReLU layers): {sparsity * 100:.2f}%')

    # -------------------------------------------------------------------------
    # Persist metrics
    # -------------------------------------------------------------------------
    if args.save_metrics:
        # Used to update the stored metrics behind the README performance tables.
        # This is a maintenance step, run when the models or pipeline change.
        metrics_path = pathlib.Path(__file__).parent / 'docs' / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        prefix = metrics_prefix(args.alpha)

        metrics[f'{prefix}{args.variant}_t1'] = f'{top1 * 100:.2f}%'
        metrics[f'{prefix}{args.variant}_t5'] = f'{top5 * 100:.2f}%'
        if sparsity is not None:
            metrics[f'{prefix}sparsity'] = f'{sparsity * 100:.2f}%'
        if args.variant == 'float':
            metrics[f'{prefix}params'] = f'{count_params(model):,}'

        metrics_path.write_text(json.dumps(metrics, indent=4, sort_keys=True) + '\n')
        print(f'Metrics saved to {metrics_path}')
