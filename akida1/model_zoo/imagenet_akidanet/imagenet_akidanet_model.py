#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Model selection and backbone loading for the AkidaNet/ImageNet example.

This module does two jobs.

**1. Resolve a model file from (alpha, resolution, variant).**
Files in ``pretrained_models/`` are named uniformly as
``akidanet_imagenet_<RES>_alpha_<A>[_qat].<ext>``, so :func:`model_path` is a
single f-string. Note this differs from the names published on
data.brainchip.com, which omit the alpha entirely for 1.0 and write the others
as integer percentages (``_alpha_50``); ``model_fetch.sh`` at the repository root
performs the renaming. The uniform scheme is worth the small divergence: it
sorts predictably and needs no lookup table.

**2. Load an ImageNet-pretrained backbone for transfer learning.**
:func:`load_akidanet_backbone` returns a headless AkidaNet with the ImageNet
weights loaded, ready to have a task-specific head attached. This is the
starting point used by the other examples in this zoo (see ``plant_village``
and ``vww``).

Why this loader exists rather than just calling ``akidanet_imagenet_pretrained``:
that helper only knows about the **224** checkpoints. The 160-resolution weights
are published, but are not reachable through it. Loading from this folder covers
both resolutions uniformly.

Usage:
    python imagenet_akidanet_model.py -a 0.5 -i 224
"""

import argparse
import pathlib

from akida_models import akidanet_imagenet
from cnn2snn import set_akida_version, AkidaVersion

__all__ = ["ALPHAS", "RESOLUTIONS", "VARIANTS", "model_path", "metrics_prefix",
           "load_akidanet_backbone", "PRETRAINED_DIR"]

ALPHAS = (0.25, 0.5, 1.0)
RESOLUTIONS = (160, 224)
VARIANTS = ("float", "qat", "akida")

PRETRAINED_DIR = pathlib.Path(__file__).parent / 'pretrained_models'

# Filename ending for each variant of a given model.
#   float -> full-precision Keras model (the transfer-learning backbone)
#   qat   -> quantization-aware-trained Keras model, 8-bit input / 4-bit weights
#            and activations (published upstream as `_iq8_wq4_aq4`)
#   akida -> the same model converted to Akida format by `cnn2snn convert`, which
#            names its output after the model it converted -- hence `_qat.fbz`
_VARIANT_SUFFIX = {'float': '.h5', 'qat': '_qat.h5', 'akida': '_qat.fbz'}


def _check(alpha, resolution):
    if alpha not in ALPHAS:
        raise ValueError(f'alpha must be one of {ALPHAS}, received {alpha}')
    if resolution not in RESOLUTIONS:
        raise ValueError(
            f'resolution must be one of {RESOLUTIONS}, received {resolution}')


def model_path(alpha, resolution, variant='float', models_dir=None):
    """Returns the path of a published AkidaNet ImageNet model.

    Args:
        alpha (float): width multiplier, one of 0.25, 0.5, 1.0.
        resolution (int): input resolution, one of 160, 224.
        variant (str, optional): 'float', 'qat' or 'akida'. Defaults to 'float'.
        models_dir (str, optional): directory holding the models. Defaults to
            this example's ``pretrained_models/``.

    Returns:
        pathlib.Path: path to the model file.
    """
    _check(alpha, resolution)
    if variant not in VARIANTS:
        raise ValueError(f'variant must be one of {VARIANTS}, received {variant}')

    directory = pathlib.Path(models_dir) if models_dir else PRETRAINED_DIR
    # float() so an int alpha still renders with its decimal place: 1 -> '1.0'
    name = (f'akidanet_imagenet_{resolution}_alpha_{float(alpha)}'
            f'{_VARIANT_SUFFIX[variant]}')
    return directory / name


def metrics_prefix(alpha, resolution):
    """Returns the ``docs/metrics.json`` key prefix for a model.

    Six models share one metrics file, so every key is namespaced by width and
    resolution, e.g. ``a50_224_float_t1``.

    Note the alpha is scaled to an integer here, unlike in the filenames. These
    keys are substituted into the README template with ``str.format_map``, which
    reads a dot in a field name as attribute access -- ``{{a0.5_224_float_t1}}``
    would be parsed as ``a0`` dot ``5_224_float_t1`` and fail. So the two
    schemes differ deliberately.
    """
    _check(alpha, resolution)
    return f'a{int(alpha * 100)}_{resolution}_'


def load_akidanet_backbone(alpha=1.0, resolution=224, pooling='avg', models_dir=None):
    """Loads an ImageNet-pretrained AkidaNet backbone, without the classifier.

    The returned model is Akida 1 compatible and expects **uint8** inputs in the
    [0, 255] range -- rescaling to [-1, 1] happens inside the model, so the data
    pipeline must not normalise (see ``imagenet_akidanet_preprocessing``).

    Attach your own head to ``backbone.output`` to fine-tune on a new task; see
    ``akida1/model_zoo/plant_village/plant_village_model.py`` for a worked example.

    Args:
        alpha (float, optional): width multiplier. Defaults to 1.0.
        resolution (int, optional): input resolution. Defaults to 224.
        pooling (str, optional): 'avg' to end with global average pooling, or
            None to keep the final feature map. Defaults to 'avg'.
        models_dir (str, optional): directory holding the models. Defaults to
            this example's ``pretrained_models/``.

    Returns:
        keras.Model: the pretrained backbone.
    """
    _check(alpha, resolution)
    weights = model_path(alpha, resolution, 'float', models_dir)
    if not weights.exists():
        raise FileNotFoundError(
            f'{weights} not found. These weights are tracked with Git LFS -- run '
            '`git lfs pull` to fetch them.')

    # The published weights are Akida 1 models. The default Akida context is v2,
    # which would build the unfused variant (separate depthwise + pointwise
    # convolutions, ReLU7.5, post-ReLU global average pooling) and fail to match
    # these weights. Forcing v1 selects the fused SeparableConv2D form instead.
    with set_akida_version(AkidaVersion.v1):
        backbone = akidanet_imagenet(input_shape=(resolution, resolution, 3),
                                     alpha=alpha,
                                     include_top=False,
                                     pooling=pooling,
                                     input_scaling=(128, -1))

    backbone.load_weights(str(weights), by_name=True)
    return backbone


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Load an ImageNet-pretrained AkidaNet backbone')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, choices=ALPHAS,
                        help='Width multiplier. Defaults to %(default)s.')
    parser.add_argument('-i', '--input-resolution', type=int, default=224,
                        choices=RESOLUTIONS, dest='resolution',
                        help='Input resolution. Defaults to %(default)s.')
    parser.add_argument('-p', '--pooling', default='avg', choices=['avg', 'none'],
                        help='Final pooling. Defaults to %(default)s.')
    parser.add_argument('-s', '--savepath', default=None,
                        help='Optionally save the backbone to this path')
    args = parser.parse_args()

    model = load_akidanet_backbone(alpha=args.alpha,
                                   resolution=args.resolution,
                                   pooling=None if args.pooling == 'none' else args.pooling)
    model.summary()
    print(f'\nBackbone: alpha={args.alpha}, {args.resolution}x{args.resolution}, '
          f'{model.count_params():,} parameters')
    print(f'Output shape: {model.output_shape}')

    if args.savepath:
        model.save(args.savepath, include_optimizer=False)
        print(f'Backbone saved to {args.savepath}')
