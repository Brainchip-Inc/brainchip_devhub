#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Model selection and creation for the MobileNetV1/ImageNet example.

This module does two jobs.

**1. Resolve a model from its width multiplier and variant.**
Two timm models are covered, selected by width multiplier (alpha):

  1.0   mobilenetv1_100
  1.25  mobilenetv1_125

and each exists in four variants:

  float      the timm PyTorch model, downloaded from the Hugging Face hub
  onnx       the same model exported to ONNX (float)
  quantized  the ONNX model quantized by quantizeml (8-bit weights and activations)
  akida      the quantized model converted to Akida by cnn2snn

Files in ``pretrained_models/`` are named after the timm model, so
:func:`model_path` is a single f-string.

**2. Create the Akida model from the timm model.**
:func:`create_float_model`, :func:`export_onnx`, :func:`quantize_model` and
:func:`convert_model` are the four steps of the PyTorch -> Akida pipeline.
Running this file chains them and writes the three model files. Quantization
needs calibration images from the ImageNet train split, so the dataset is
required (see the README).

Only the first two steps need PyTorch (see Requirements in the README). Loading the
ONNX, quantized and Akida models does not.

Usage:
    python imagenet_mobilenet_model.py -a 1.0 -d /path/to/imagenet
"""
import os
# akida_models imports TensorFlow, which by default reserves almost all GPU memory
# as soon as it starts, leaving none for PyTorch. TensorFlow does no GPU work in
# this example, so let it allocate only what it uses. Set before TensorFlow loads.
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

import argparse
import pathlib

import akida
import onnx
from cnn2snn import convert
from quantizeml.models import quantize

__all__ = ["ALPHAS", "VARIANTS", "INPUT_SIZE", "TIMM_NAMES", "TIMM_REFERENCE",
           "PRETRAINED_DIR", "model_path", "metrics_prefix", "load_model",
           "create_float_model", "export_onnx", "quantize_model", "convert_model"]

ALPHAS = (1.0, 1.25)
VARIANTS = ("float", "onnx", "quantized", "akida")
INPUT_SIZE = 224

PRETRAINED_DIR = pathlib.Path(__file__).parent / 'pretrained_models'

# timm model name for each width multiplier
TIMM_NAMES = {1.0: 'mobilenetv1_100', 1.25: 'mobilenetv1_125'}

# ImageNet (top-1, top-5) accuracy published by timm for the default pretrained
# weights of each model at 224 x 224 (mobilenetv1_100.ra4_e3600_r224_in1k and
# mobilenetv1_125.ra4_e3600_r224_in1k).
TIMM_REFERENCE = {1.0: (75.382, 92.312), 1.25: (76.924, 93.234)}

# Filename ending for each variant stored on disk. The float variant has no
# file: timm downloads its weights.
_VARIANT_SUFFIX = {'onnx': '.onnx', 'quantized': '_quantized.onnx', 'akida': '.fbz'}


def _check(alpha, variant=None):
    if alpha not in ALPHAS:
        raise ValueError(f'alpha must be one of {ALPHAS}, received {alpha}')
    if variant is not None and variant not in VARIANTS:
        raise ValueError(f'variant must be one of {VARIANTS}, received {variant}')


def model_path(alpha, variant, models_dir=None):
    """Returns the path of a MobileNetV1 model file.

    Args:
        alpha (float): width multiplier, 1.0 or 1.25.
        variant (str): 'onnx', 'quantized' or 'akida'.
        models_dir (str, optional): directory holding the models. Defaults to
            this example's ``pretrained_models/``.

    Returns:
        pathlib.Path: path to the model file.
    """
    _check(alpha, variant)
    if variant == 'float':
        raise ValueError('The float variant has no model file: it is downloaded '
                         'by timm, see create_float_model().')
    directory = pathlib.Path(models_dir) if models_dir else PRETRAINED_DIR
    return directory / f'{TIMM_NAMES[alpha]}{_VARIANT_SUFFIX[variant]}'


def metrics_prefix(alpha):
    """Returns the ``docs/metrics.json`` key prefix for a model.

    Both models share one metrics file, so every key is namespaced by width and
    resolution, e.g. ``a125_224_akida_t1``.

    The alpha is scaled to an integer because these keys are substituted into
    the README template with ``str.format_map``, which reads a dot in a field
    name as attribute access: ``{a1.25_224_akida_t1}`` would fail to render.
    """
    _check(alpha)
    return f'a{round(alpha * 100)}_{INPUT_SIZE}_'


def create_float_model(alpha):
    """Returns the pretrained timm model, in eval mode. Needs PyTorch and timm."""
    import timm

    _check(alpha)
    return timm.create_model(TIMM_NAMES[alpha], pretrained=True).eval()


def export_onnx(pt_model, path):
    """Exports a timm model to ONNX with a dynamic batch axis. Needs PyTorch.

    Args:
        pt_model (torch.nn.Module): the float timm model.
        path (str or pathlib.Path): where to write the ``.onnx`` file.

    Returns:
        onnx.ModelProto: the exported model, re-loaded from ``path``.
    """
    import torch

    # The export traces the model, so the dummy input only has to have the
    # right shape: its values do not matter.
    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(pt_model.to('cpu').eval(), dummy, f=str(path),
                      input_names=['inputs'], output_names=['outputs'],
                      dynamic_axes={'inputs': {0: 'batch_size'},
                                    'outputs': {0: 'batch_size'}})
    return onnx.load_model(str(path))


def quantize_model(model_onnx, samples, batch_size=128):
    """Quantizes a float ONNX model with quantizeml (8-bit, post-training).

    Args:
        model_onnx (onnx.ModelProto): the float ONNX model.
        samples (np.ndarray): calibration images, preprocessed exactly as for
            the float model (float32, normalised, NCHW). They should come from
            the train split, so the validation set stays unseen.
        batch_size (int, optional): calibration batch size. Defaults to 128.

    Returns:
        onnx.ModelProto: the quantized model.
    """
    # Default quantization parameters: 8-bit weights and activations.
    # QuantizationParams(per_tensor_activations=True), as used in the quantizeml
    # MobileNetV2 off-the-shelf example, was tried and did not improve accuracy.
    return quantize(model_onnx, samples=samples, batch_size=batch_size)


def convert_model(model_quantized):
    """Converts a quantized ONNX model to an Akida model.

    The input normalisation is folded into the first layer, so the Akida model
    takes raw uint8 images (see ``imagenet_mobilenet_preprocessing``).
    """
    return convert(model_quantized)


def load_model(alpha, variant, models_dir=None):
    """Loads one variant of a model.

    Returns:
        a ``torch.nn.Module`` (float), an ``onnx.ModelProto`` (onnx, quantized)
        or an ``akida.Model`` (akida).
    """
    _check(alpha, variant)
    if variant == 'float':
        return create_float_model(alpha)

    path = model_path(alpha, variant, models_dir)
    if not path.exists():
        raise FileNotFoundError(
            f'{path} not found. The published models are tracked with Git LFS - '
            'run `git lfs pull` to fetch them, or create them with '
            'imagenet_mobilenet_model.py.')
    if variant == 'akida':
        return akida.Model(str(path))
    return onnx.load_model(str(path))


if __name__ == '__main__':
    from imagenet_mobilenet_data import get_calibration_samples

    parser = argparse.ArgumentParser(
        description='Create the ONNX, quantized and Akida MobileNetV1 models')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, choices=ALPHAS,
                        help='Width multiplier. Defaults to %(default)s.')
    parser.add_argument('-d', '--data', required=True,
                        help='ImageNet root, containing a train/ folder with one '
                             'subfolder per class (used for calibration)')
    parser.add_argument('-n', '--num-calib-samples', type=int, default=8192,
                        help='Number of calibration images. Defaults to %(default)s.')
    parser.add_argument('-b', '--batch_size', type=int, default=128,
                        help='Calibration batch size. Defaults to %(default)s.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for drawing the calibration images. '
                             'Defaults to %(default)s.')
    parser.add_argument('-s', '--savedir', default=None,
                        help='Directory to write the models to. Defaults to '
                             'pretrained_models/, overwriting the published models.')
    args = parser.parse_args()

    savedir = pathlib.Path(args.savedir) if args.savedir else PRETRAINED_DIR
    savedir.mkdir(parents=True, exist_ok=True)

    # 1. Float model
    pt_model = create_float_model(args.alpha)
    print(f'{TIMM_NAMES[args.alpha]}: {sum(p.numel() for p in pt_model.parameters()):,} '
          f'parameters, pretrained weights {pt_model.pretrained_cfg["hf_hub_id"]}')

    # 2. Export to ONNX
    onnx_path = model_path(args.alpha, 'onnx', savedir)
    model_onnx = export_onnx(pt_model, onnx_path)
    print(f'ONNX model saved to {onnx_path}')

    # 3. Quantize, calibrating on images from the train split
    samples = get_calibration_samples(args.data, args.alpha,
                                      num_samples=args.num_calib_samples,
                                      batch_size=args.batch_size, seed=args.seed)
    print(f'Calibration samples: {samples.shape}')
    model_quantized = quantize_model(model_onnx, samples, batch_size=args.batch_size)
    quantized_path = model_path(args.alpha, 'quantized', savedir)
    onnx.save_model(model_quantized, str(quantized_path))
    print(f'Quantized model saved to {quantized_path}')

    # 4. Convert to Akida
    model_akida = convert_model(model_quantized)
    model_akida.summary()
    akida_path = model_path(args.alpha, 'akida', savedir)
    model_akida.save(str(akida_path))
    print(f'Akida model saved to {akida_path}')
