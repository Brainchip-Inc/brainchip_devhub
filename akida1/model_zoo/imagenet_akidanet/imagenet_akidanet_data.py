#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Data loading for the AkidaNet/ImageNet example.

Two sources are provided, for two different jobs:

* :func:`get_data` -- the full **ImageNet 2012 validation split** (50,000 images)
  via TensorFlow Datasets. This is what the accuracy figures in the README are
  measured on. It requires a one-off manual dataset setup (see the README), since
  ImageNet cannot be downloaded automatically.

* :func:`get_samples` / :func:`get_labelled_samples` -- a small **10-image
  ImageNet-like sample pack** hosted by BrainChip, fetched on demand. This needs
  no setup at all. It is used to drive the hardware benchmark (Akida latency and
  power are activity-dependent, so benchmarking on real images matters far more
  than the number of them) and as a smoke test of the preprocessing and label
  conventions. It is far too small to measure accuracy with.

Labels are integer class indices in [0, 999], matching the TFDS ``imagenet2012``
class order. :func:`index_to_label` maps one to its human-readable name.
"""

import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from akida_models.imagenet import index_to_label  # noqa: F401  (re-exported)
from akida_models.utils import fetch_file

from imagenet_akidanet_preprocessing import preprocess_image

__all__ = ["get_data", "get_samples", "get_labelled_samples", "index_to_label",
           "NUM_CLASSES", "NUM_VAL_IMAGES"]

NUM_CLASSES = 1000
NUM_VAL_IMAGES = 50000

# Small ImageNet-like sample pack (10 labelled JPEGs), mirrored by BrainChip.
# Note this is *not* ImageNet itself: it is a redistributable stand-in, which is
# why it can be fetched freely while ImageNet requires manual setup.
SAMPLES_URL = "https://data.brainchip.com/dataset-mirror/imagenet_like/imagenet_like.zip"
NUM_SAMPLE_IMAGES = 10


def get_data(data_path, input_shape, batch_size=32, dtype=tf.uint8):
    """Loads the ImageNet 2012 validation split via tensorflow_datasets.

    The dataset must already have been prepared (see 'Dataset setup' in the
    README). ``data_path`` is the directory *containing* the ImageNet tar files;
    the TFDS store is expected at ``<data_path>/tfds/data``.

    Args:
        data_path (str): directory holding the ImageNet tar files and the
            ``tfds/`` store.
        input_shape (tuple): input image shape (height, width, channels).
        batch_size (int, optional): the batch size. Defaults to 32.
        dtype (tf.dtypes.DType, optional): input data type. Defaults to tf.uint8,
            which is what the model expects (it rescales internally).

    Returns:
        tf.data.Dataset, int: the validation dataset and its number of examples.
    """
    image_size = tuple(input_shape[:2])

    # Layout expected by the akida_models ImageNet tooling: the tar files sit in
    # data_path, and TFDS reads/writes under data_path/tfds.
    write_dir = os.path.join(data_path, 'tfds')
    download_and_prepare_kwargs = {
        'download_dir': os.path.join(write_dir, 'downloaded'),
        'download_config': tfds.download.DownloadConfig(manual_dir=data_path),
    }

    tfds.disable_progress_bar()
    dataset, infos = tfds.load(
        'imagenet2012',
        data_dir=os.path.join(write_dir, 'data'),
        split='validation',
        shuffle_files=False,
        download=True,
        as_supervised=True,
        download_and_prepare_kwargs=download_and_prepare_kwargs,
        with_info=True)

    def _preprocess(image, label):
        image = preprocess_image(image, image_size)
        return tf.cast(image, dtype), tf.cast(label, tf.int32)

    dataset = (dataset
               .map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
               .batch(batch_size)
               .prefetch(tf.data.AUTOTUNE))

    # Silences a TensorFlow warning about the auto shard policy
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = \
        tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)

    return dataset, infos.splits['validation'].num_examples


def get_labelled_samples(input_shape):
    """Loads the 10-image ImageNet-like sample pack, with labels.

    Downloaded on first use and cached under ``~/.keras/datasets``. No ImageNet
    setup is required.

    Args:
        input_shape (tuple): input image shape (height, width, channels).

    Returns:
        np.ndarray, np.ndarray: images of shape (10, H, W, C) as uint8, and
        their integer class labels.
    """
    image_size = tuple(input_shape[:2])
    num_channels = input_shape[2] if len(input_shape) > 2 else 3

    file_path = fetch_file(fname="imagenet_like.zip",
                           origin=SAMPLES_URL,
                           cache_subdir='datasets/imagenet_like',
                           extract=True)
    data_folder = os.path.dirname(file_path)

    # Labels file maps image filename -> class index
    labels = {}
    with open(os.path.join(data_folder, 'labels_validation.txt')) as f:
        for line in f:
            if line.strip():
                name, index = line.split()
                labels[name] = int(index)

    images = np.zeros((NUM_SAMPLE_IMAGES, *image_size, num_channels), dtype=np.uint8)
    targets = np.zeros(NUM_SAMPLE_IMAGES, dtype=np.int32)

    for idx in range(NUM_SAMPLE_IMAGES):
        fname = f'image_{str(idx + 1).zfill(2)}.jpg'
        raw = tf.io.read_file(os.path.join(data_folder, fname))
        image = tf.io.decode_jpeg(raw, channels=num_channels)
        images[idx] = preprocess_image(image, image_size).numpy().astype(np.uint8)
        targets[idx] = labels[fname]

    return images, targets


def get_samples(input_shape, num_samples=100, data_path=None):
    """Returns uint8 image samples for benchmarking and sparsity measurement.

    By default this uses the 10-image ImageNet-like pack, cycled up to
    ``num_samples``. Akida's latency and power depend on activation sparsity,
    which in turn depends on the input, so benchmarking must use real images
    rather than random noise -- but a handful of real images captures the
    activity statistics well enough for that purpose.

    Pass ``data_path`` to draw the samples from the ImageNet validation split
    instead, which gives a more representative spread at the cost of requiring
    the full dataset setup.

    Args:
        input_shape (tuple): input image shape (height, width, channels).
        num_samples (int, optional): number of samples to return. Defaults to 100.
        data_path (str, optional): if given, take samples from the ImageNet
            validation split at this path instead of the sample pack.

    Returns:
        np.ndarray: array of shape (num_samples, H, W, C), dtype uint8.
    """
    if data_path is not None:
        dataset, _ = get_data(data_path, input_shape, batch_size=num_samples)
        for images, _ in dataset.take(1):
            return images.numpy().astype(np.uint8)

    images, _ = get_labelled_samples(input_shape)

    # Cycle the 10 images up to the requested count
    repeats = int(np.ceil(num_samples / len(images)))
    return np.tile(images, (repeats, 1, 1, 1))[:num_samples]
