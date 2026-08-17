#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
ImageNet preprocessing for AkidaNet evaluation.

This is a self-contained copy of the inference/validation preprocessing used by
``akida_models.imagenet.preprocessing``. It is reproduced here rather than
imported so that this example documents, in one readable place, exactly what is
done to a JPEG before it reaches the model.

The pipeline is the standard ImageNet "resize shorter side, then centre crop":

1. Aspect-preserving resize so the *shorter* side becomes ``round(size * 1.143)``
   (224 -> 256, 160 -> 183). The 1.143 factor is the conventional 256/224 ratio.
2. Centre crop to the target ``size x size``.

Two things are deliberately *not* done here, and both trip people up:

* **No mean/std normalisation.** There is no ImageNet mean subtraction anywhere
  in this pipeline.
* **No division by 255.** Pixels stay in the [0, 255] range.

That is because normalisation lives *inside* the model, as a Keras ``Rescaling``
layer built by ``akidanet_imagenet(input_scaling=(128, -1))`` -- i.e. the model
itself computes ``x / 128 - 1``. The data pipeline therefore hands the model
plain uint8 pixels. Adding your own normalisation on top is the single most
common cause of "the pretrained model scores 0.1%".
"""

import numpy as np
import tensorflow as tf

__all__ = ["preprocess_image", "RESIZE_RATIO"]

# Conventional ImageNet ratio between the resize target and the crop size
# (256/224). Applied to whichever input resolution is in use.
RESIZE_RATIO = 1.143


@tf.function
def preprocess_image(image, image_size):
    """ImageNet validation/inference preprocessing.

    Aspect-preserving resize followed by a central crop. No normalisation is
    applied -- see the module docstring.

    Args:
        image (tf.Tensor): input image as a 3-D uint8 tensor (H, W, C).
        image_size (tuple): target size as (height, width).

    Returns:
        :obj:`tensorflow.Tensor`: preprocessed image, float32 in the [0, 255]
        range. Cast to uint8 by the caller before feeding the model.
    """
    assert len(image_size) == 2, \
        f"image_size should have 2 elements (H, W), received {image_size}."

    shape = tf.shape(image)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)

    # Scale the image before cropping, keeping the aspect ratio: the shorter
    # side is resized to image_size * RESIZE_RATIO so that the subsequent
    # central crop keeps the informative middle of the frame.
    resize_min_h = np.round(image_size[0] * RESIZE_RATIO).astype(np.float32)
    resize_min_w = np.round(image_size[1] * RESIZE_RATIO).astype(np.float32)
    min_dim = tf.minimum(height, width)
    scale_ratio = (resize_min_h / min_dim, resize_min_w / min_dim)

    # Convert back to int for the TF resize op
    new_height = tf.cast(height * scale_ratio[0], tf.int32)
    new_width = tf.cast(width * scale_ratio[1], tf.int32)

    image = tf.image.resize(image, [new_height, new_width])

    # Central crop to the desired size
    image = tf.image.resize_with_crop_or_pad(image, image_size[0], image_size[1])

    return tf.cast(image, tf.float32)
