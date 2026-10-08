#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
ImageNet preprocessing for the MobileNetV1 (timm) evaluation.

This is a self-contained copy of the evaluation transform that timm builds for
the ``mobilenetv1_100`` and ``mobilenetv1_125`` models from their
``pretrained_cfg`` (``timm.data.resolve_data_config`` +
``timm.data.create_transform``). It is written out here so that this example
documents, in one readable place, exactly what is done to a JPEG before it
reaches the model, and it is written with PIL and numpy only, so that the Akida
side of the example runs without PyTorch.

The pipeline is the standard ImageNet "resize shorter side, then centre crop":

1. Aspect-preserving **bicubic** resize so the *shorter* side becomes
   ``floor(size / crop_pct)``. ``crop_pct`` differs between the two widths
   (see ``CROP_PCT``): 224 -> 256 for alpha=1.0, 224 -> 248 for alpha=1.25.
2. Centre crop to the target ``size x size``.

What happens next depends on which model the image is for:

* **PyTorch and ONNX models (float and quantized)** take a float32 CHW array
  scaled to [0, 1] and normalised with mean 0.5 and std 0.5, i.e. mapped to
  [-1, 1].
* **Akida models** take the raw **uint8** CHW pixels, with no scaling and no
  normalisation. ``cnn2snn.convert`` folds the input normalisation into the
  first layer, so normalising the data yourself on top of that is the most
  common cause of an Akida model scoring at chance.

The output is identical, value for value, to timm's own transform (the resize
and crop reproduce torchvision's ``Resize`` and ``CenterCrop`` on PIL images).
"""
import functools
import math

import numpy as np
from PIL import Image

__all__ = ["get_transform", "resize_and_crop", "CROP_PCT", "MEAN", "STD"]

# Ratio between the crop size and the resize target, per width multiplier, from
# the timm pretrained_cfg of each model
CROP_PCT = {1.0: 0.875, 1.25: 0.9}

# Normalisation, from the timm pretrained_cfg (the same for both widths)
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def resize_and_crop(image, alpha, input_size=224):
    """Bicubic resize of the shorter side, then centre crop.

    Args:
        image (PIL.Image.Image): an RGB image.
        alpha (float): width multiplier, 1.0 or 1.25. Selects the crop ratio.
        input_size (int, optional): output height and width. Defaults to 224.

    Returns:
        PIL.Image.Image: the ``input_size x input_size`` crop.
    """
    if alpha not in CROP_PCT:
        raise ValueError(f'alpha must be one of {tuple(CROP_PCT)}, received {alpha}')

    # Resize: shorter side to floor(input_size / crop_pct), long side scaled and
    # truncated, as torchvision.transforms.Resize does
    short_target = math.floor(input_size / CROP_PCT[alpha])
    width, height = image.size
    if width <= height:
        new_size = (short_target, int(short_target * height / width))
    else:
        new_size = (int(short_target * width / height), short_target)
    image = image.resize(new_size, Image.BICUBIC)

    # Centre crop, rounding the offsets as torchvision.transforms.CenterCrop does
    width, height = image.size
    top = int(round((height - input_size) / 2.0))
    left = int(round((width - input_size) / 2.0))
    return image.crop((left, top, left + input_size, top + input_size))


def _preprocess(image, alpha, input_size, normalize):
    pixels = np.asarray(resize_and_crop(image, alpha, input_size), dtype=np.uint8)
    pixels = pixels.transpose(2, 0, 1)  # HWC -> CHW, as the exported models expect
    if not normalize:
        return pixels.copy()
    mean = np.asarray(MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(STD, dtype=np.float32)[:, None, None]
    return (pixels.astype(np.float32) / np.float32(255) - mean) / std


def get_transform(alpha, input_size=224, normalize=True):
    """Returns the evaluation transform for a MobileNetV1 model.

    Args:
        alpha (float): width multiplier, 1.0 or 1.25. Selects the crop ratio.
        input_size (int, optional): output height and width. Defaults to 224.
        normalize (bool, optional): True for PyTorch/ONNX models (float32,
            normalised), False for Akida models (raw uint8). Defaults to True.

    Returns:
        callable: a function taking a PIL RGB image and returning a numpy
        array of shape (3, H, W).
    """
    if alpha not in CROP_PCT:
        raise ValueError(f'alpha must be one of {tuple(CROP_PCT)}, received {alpha}')
    # A partial rather than a closure, so DataLoader workers can pickle it
    return functools.partial(_preprocess, alpha=alpha, input_size=input_size,
                             normalize=normalize)
