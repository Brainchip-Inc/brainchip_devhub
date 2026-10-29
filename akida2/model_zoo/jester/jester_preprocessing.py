#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Evaluation preprocessing for the Jester TENN example (Akida 2).

A self-contained copy of the validation path of
``akida_models.tenn_spatiotemporal.jester_train.get_data``, so that what happens
to a clip before it reaches the model is documented in one readable place.
``jester_data.check_preprocessing`` asserts that this copy is bit-identical to
the ``akida_models`` implementation.

Each Jester clip is stored in a GZIP TFRecord as a sequence of JPEG frames, 100
pixels high and of varying width. For evaluation, a clip becomes a
``(16, 100, 100, 3)`` uint8 tensor:

* **Frames:** 16 frames, taking every other frame from the first one (frames 0,
  2, 4, ..., 30). A clip shorter than 32 frames wraps around to its start.
  Training uses the same stride from a random starting frame.
* **Size:** each frame is center-cropped (or zero-padded, if it is smaller) to
  100 x 100. Frames are not resized.
* **Values:** raw uint8 pixels in [0, 255]. **Do not normalize.** The model's
  first layer is a ``Rescaling(1.)`` placeholder that only exists so the model
  can take uint8 input, because this model was trained on unscaled pixels.

The float model takes the whole 16-frame clip at once. The quantized and Akida
models take one ``(100, 100, 3)`` frame at a time, in order: see
``jester_eval.py``.
"""
import tensorflow as tf

__all__ = ["FRAMES_PER_CLIP", "FRAME_STEP", "FRAME_SIZE", "parse_sequence_example",
           "decode_clip"]

FRAMES_PER_CLIP = 16
FRAME_STEP = 2
FRAME_SIZE = (100, 100)


def parse_sequence_example(example):
    """Parses one serialized clip into (encoded frames, label, frame count).

    Args:
        example (tf.Tensor): a serialized ``tf.train.SequenceExample``.

    Returns:
        tuple: the JPEG-encoded frames (tf.string, shape (frame_count,)), the
        integer label, and the clip's frame count.
    """
    context, sequence = tf.io.parse_single_sequence_example(
        example,
        context_features={
            'filename': tf.io.FixedLenFeature([], tf.string),
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'frame_count': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
        },
        sequence_features={'frames': tf.io.FixedLenSequenceFeature([], dtype=tf.string)})
    return sequence['frames'], context['label'], context['frame_count']


def decode_clip(encoded_frames, frame_count, frames_per_clip=FRAMES_PER_CLIP,
                step=FRAME_STEP, frame_size=FRAME_SIZE):
    """Decodes the frames used for evaluation from one clip.

    Args:
        encoded_frames (tf.Tensor): the clip's JPEG-encoded frames.
        frame_count (tf.Tensor): the clip's number of frames.
        frames_per_clip (int, optional): number of frames to keep. Defaults to 16.
        step (int, optional): stride between kept frames. Defaults to 2.
        frame_size (tuple, optional): output (height, width). Defaults to (100, 100).

    Returns:
        tf.Tensor: uint8 tensor of shape (frames_per_clip, height, width, 3).
    """
    total = tf.cast(frame_count, tf.int32)
    # Every `step`-th frame from the first; a short clip wraps around to its start
    indices = tf.math.floormod(tf.range(0, frames_per_clip * step, step), total)
    frames = tf.TensorArray(tf.uint8, size=frames_per_clip)
    for i in tf.range(frames_per_clip):
        image = tf.io.decode_jpeg(tf.gather(encoded_frames, indices[i]))
        # Center crop (or pad) without resizing. The float round trip matches the
        # reference implementation, and is exact for uint8 pixels.
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_with_crop_or_pad(image, frame_size[0], frame_size[1])
        frames = frames.write(i, tf.image.convert_image_dtype(image, tf.uint8))
    return frames.stack()
