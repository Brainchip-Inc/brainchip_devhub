#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Model definitions and model file selection for the MobileNet/ImageNet example.

This module does two jobs.

**1. Build an Akida 2 ready MobileNet from the Keras ImageNet weights.**
:func:`build_mobilenet` takes MobileNetV1 or MobileNetV2 from
``tf_keras.applications``, with its ImageNet weights, and makes two changes so
that it quantizes and converts cleanly. Neither changes the predictions.

* **A Rescaling layer up front.** The Keras MobileNets expect inputs in [-1, 1].
  ``quantizeml``'s ``insert_rescaling`` adds that scaling to the model itself, so
  the model takes uint8 pixels in [0, 255], as the Akida model will. The data
  pipeline then never normalises (see ``imagenet_mobilenet_preprocessing``).

* **No softmax.** The classifier outputs logits. Quantization and conversion
  would drop a final softmax anyway (Akida does not run it), with a warning;
  removing it here avoids the warning. The ranking of the logits is the same as
  that of the probabilities, so top-1 and top-5 accuracy are unaffected. To get
  class probabilities, apply a softmax to the model output on the host.

The classifier head is the part worth knowing about when you reuse these models
for another task:

* **MobileNetV2** keeps its own head: global average pooling, then a ``Dense``
  classifier.

* **MobileNetV1** ships with a different head: global average pooling (keeping
  the 1 x 1 spatial dimensions), dropout, a 1 x 1 ``Conv2D`` classifier, and a
  ``Reshape`` that drops the singleton dimensions. Here the convolution is
  replaced with the equivalent ``Dense`` layer (the same weights, reshaped), and
  the ``Reshape`` goes with it. The reason is mapping: on Akida 2 a 1000-class
  dense layer runs on a single dedicated dense-layer NP (FNP), whereas a 1 x 1
  convolution with 1000 filters is spread over many convolutional NPs (CNPs).
  On the 6-node Akida 2 FPGA that convolution needs more CNPs than a single
  pass has from alpha 0.75 up, so those models don't fit in a single hardware
  sequence. A classifier
  with few classes is a different case: see below.

* **Dropout is kept** in the V1 head. It has no effect at inference and
  conversion removes it, so you can keep using dropout when you fine-tune these
  models on another task and still convert them as they are.

**Which models fit the 6-node Akida 2 FPGA.** MobileNetV1 at alpha 0.25, 0.5
and 0.75, and MobileNetV2 at 0.35 and 0.5, map onto it in a single hardware
sequence. The other three fit in one sequence on a 12-node Akida 2 device; on
the FPGA they only run split into several sequences (``hw_only=False``), with
the values between sequences leaving the chip. In each, a single layer needs
more NPs than one pass on the FPGA offers (NP counts from the akida 2.19 mapper).

* MobileNetV2 0.75 and 1.0: ``block_1_depthwise`` needs 21 NPs. That is driven
  by its large input volume, 112 x 112 x 96 (the block has 96 channels at both
  widths).
* MobileNetV1 1.0: ``conv_pw_13`` needs 22 NPs, driven by the filter memory for
  its 1024 x 1024 weights. Those include the NPs that host the FNPs, and the
  classifier must share a pass with it (a layer writes its outputs into the next
  layer's input buffers, so every pass holds at least two layers), so no dense
  classifier can follow it. Without the 1000-class ImageNet classifier the
  model does fit in one sequence, though: as a backbone (for object detection,
  say), or with a small classifier for an edge task, written as a 1 x 1
  convolution after pooling. A 30-class one fits (it needs one more NP); a
  100-class one needs three and doesn't. So MobileNetV1 1.0 remains a good starting point for
  transfer learning on the FPGA.

**2. Resolve a model file from (version, alpha, variant).**
Files are named uniformly as ``mobilenet_<VERSION>_imagenet_224_alpha_<A>``,
with ``_i8_w8_a8`` for the 8-bit quantized model (``.h5``) and its Akida
conversion (``.fbz``). The float models are not stored in ``pretrained_models/``:
:func:`build_mobilenet` rebuilds them from the Keras weights, which
``tf_keras`` downloads on first use.

Usage:
    python imagenet_mobilenet_model.py -v v1 -a 0.5 -s models/mobilenet_v1_imagenet_224_alpha_0.5.h5
"""

import argparse
import pathlib

from tf_keras import Model
from tf_keras.applications.mobilenet import MobileNet
from tf_keras.applications.mobilenet_v2 import MobileNetV2
from tf_keras.layers import Dense, Dropout, GlobalAveragePooling2D
from quantizeml.models.transforms import insert_rescaling

__all__ = ["VERSIONS", "ALPHAS", "RESOLUTION", "VARIANTS", "build_mobilenet",
           "model_name", "model_path", "metrics_prefix", "PRETRAINED_DIR"]

VERSIONS = ("v1", "v2")
# The ImageNet-pretrained widths published for each version at 224 x 224.
# (MobileNetV2 also has 1.3 and 1.4; they are left out as the models get large.)
ALPHAS = {"v1": (0.25, 0.5, 0.75, 1.0),
          "v2": (0.35, 0.5, 0.75, 1.0)}
RESOLUTION = 224
VARIANTS = ("float", "quantized", "akida")

PRETRAINED_DIR = pathlib.Path(__file__).parent / 'pretrained_models'

# Filename ending for each variant of a given model.
#   float     -> full-precision Keras model (rebuilt by build_mobilenet, not stored)
#   quantized -> 8-bit post-training quantized Keras model (quantizeml)
#   akida     -> the same model converted to Akida format by `cnn2snn convert`,
#                which names its output after the model it converted
_VARIANT_SUFFIX = {'float': '.h5', 'quantized': '_i8_w8_a8.h5', 'akida': '_i8_w8_a8.fbz'}


def _check(version, alpha):
    if version not in VERSIONS:
        raise ValueError(f'version must be one of {VERSIONS}, received {version}')
    if alpha not in ALPHAS[version]:
        raise ValueError(
            f'alpha must be one of {ALPHAS[version]} for {version}, received {alpha}')


def build_mobilenet(version, alpha):
    """Builds an ImageNet-pretrained MobileNet ready for Akida 2.

    See the module docstring for the changes made to the Keras model.

    Args:
        version (str): 'v1' or 'v2'.
        alpha (float): width multiplier, one of ``ALPHAS[version]``.

    Returns:
        keras.Model: the model, taking uint8 224 x 224 x 3 inputs and returning
        1000 logits.
    """
    _check(version, alpha)
    input_shape = (RESOLUTION, RESOLUTION, 3)

    if version == 'v1':
        keras_model = MobileNet(input_shape=input_shape, alpha=alpha,
                                include_top=True, weights='imagenet',
                                classifier_activation=None)
        # Rebuild the head with a Dense classifier in place of the 1 x 1
        # convolution and reshape, keeping the dropout.
        features = keras_model.get_layer('conv_pw_13_relu').output
        x = GlobalAveragePooling2D(name='global_avg_pool')(features)
        x = Dropout(keras_model.get_layer('dropout').rate, name='dropout')(x)
        classifier = Dense(1000, name='predictions')
        x = classifier(x)
        kernel, bias = keras_model.get_layer('conv_preds').get_weights()
        classifier.set_weights([kernel[0, 0], bias])  # (1, 1, C, 1000) -> (C, 1000)
        model = Model(keras_model.input, x)
    else:
        model = MobileNetV2(input_shape=input_shape, alpha=alpha,
                            include_top=True, weights='imagenet',
                            classifier_activation=None)

    # The Keras MobileNets expect inputs in [-1, 1]: x / 127.5 - 1
    model = insert_rescaling(model, scale=1 / 127.5, offset=-1)
    model._name = model_name(version, alpha)
    return model


def model_name(version, alpha):
    """Returns the model's name, which is also the stem of its filenames."""
    _check(version, alpha)
    # float() so an int alpha still renders with its decimal place: 1 -> '1.0'
    return f'mobilenet_{version}_imagenet_{RESOLUTION}_alpha_{float(alpha)}'


def model_path(version, alpha, variant='quantized', models_dir=None):
    """Returns the path of a model file.

    Args:
        version (str): 'v1' or 'v2'.
        alpha (float): width multiplier, one of ``ALPHAS[version]``.
        variant (str, optional): 'float', 'quantized' or 'akida'. Defaults to
            'quantized'.
        models_dir (str, optional): directory holding the models. Defaults to
            this example's ``pretrained_models/``.

    Returns:
        pathlib.Path: path to the model file.
    """
    if variant not in VARIANTS:
        raise ValueError(f'variant must be one of {VARIANTS}, received {variant}')
    directory = pathlib.Path(models_dir) if models_dir else PRETRAINED_DIR
    return directory / (model_name(version, alpha) + _VARIANT_SUFFIX[variant])


def metrics_prefix(version, alpha):
    """Returns the ``docs/metrics.json`` key prefix for a model.

    Eight models share one metrics file, so every key is namespaced by version
    and width, e.g. ``v2_a35_float_t1``.

    Note the alpha is scaled to an integer here, unlike in the filenames. These
    keys are substituted into the README template with ``str.format_map``, which
    reads a dot in a field name as attribute access -- ``{v1_a0.5_float_t1}``
    would be parsed as ``v1_a0`` dot ``5_float_t1`` and fail. So the two
    schemes differ deliberately.
    """
    _check(version, alpha)
    return f'{version}_a{round(alpha * 100)}_'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build an ImageNet-pretrained MobileNet for Akida 2')
    parser.add_argument('-v', '--version', default='v1', choices=VERSIONS,
                        help='MobileNet version. Defaults to %(default)s.')
    parser.add_argument('-a', '--alpha', type=float, default=1.0,
                        help='Width multiplier. Defaults to %(default)s.')
    parser.add_argument('-s', '--savepath', default=None,
                        help='Optionally save the model to this path')
    args = parser.parse_args()

    model = build_mobilenet(args.version, args.alpha)
    model.summary()
    print(f'\n{model.name}: {model.count_params():,} parameters')

    if args.savepath:
        model.save(args.savepath, include_optimizer=False)
        print(f'Model saved to {args.savepath}')
