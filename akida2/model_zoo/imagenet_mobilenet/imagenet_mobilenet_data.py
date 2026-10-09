#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Data loading for the MobileNetV1/ImageNet example.

Two sources are provided, for two different jobs:

* :func:`get_data` / :func:`get_calibration_samples` -- the full **ImageNet
  2012** dataset, read from disk with ``torchvision.datasets.ImageFolder``. The
  validation split (50,000 images) is what the accuracy figures in the README
  are measured on, and the train split supplies the quantization calibration
  images. ImageNet cannot be downloaded automatically, so this needs a one-off
  manual setup (see the README). These two functions need PyTorch.

* :func:`get_samples` / :func:`get_labelled_samples` -- a small **10-image
  ImageNet-like sample pack** hosted by BrainChip, fetched on demand (the same
  one used by the ``imagenet_akidanet`` example). This needs no setup and no
  PyTorch. It drives the hardware benchmark (Akida latency depends on
  activation sparsity, so benchmarking on real images matters far more than the
  number of them) and serves as a smoke test of the preprocessing and label
  conventions. It is far too small to measure accuracy with.

Labels are integer class indices in [0, 999] in sorted-synset order, which is
both the ``ImageFolder`` order and the order the timm models were trained with.
:func:`index_to_label` maps one to its human-readable name.
"""
import os
# akida_models imports TensorFlow, which by default reserves almost all GPU memory
# as soon as it starts, leaving none for PyTorch. TensorFlow does no GPU work in
# this example, so let it allocate only what it uses. Set before TensorFlow loads.
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')

import numpy as np
from PIL import Image

from akida_models.imagenet import index_to_label  # noqa: F401  (re-exported)
from akida_models.utils import fetch_file

from imagenet_mobilenet_preprocessing import get_transform

__all__ = ["get_data", "get_calibration_samples", "get_samples",
           "get_labelled_samples", "index_to_label", "NUM_CLASSES"]

NUM_CLASSES = 1000

# Small ImageNet-like sample pack (10 labelled JPEGs), mirrored by BrainChip.
# Note this is *not* ImageNet itself: it is a redistributable stand-in, which is
# why it can be fetched freely while ImageNet requires manual setup.
SAMPLES_URL = "https://data.brainchip.com/dataset-mirror/imagenet_like/imagenet_like.zip"
NUM_SAMPLE_IMAGES = 10


def get_data(data_path, alpha, split='validation', batch_size=128, normalize=True,
             num_samples=None, seed=None, num_workers=8, resolution=224):
    """Loads one ImageNet split from an ``ImageFolder`` layout.

    ``data_path`` must contain one folder per split, each with one subfolder
    per class (synset id), e.g. ``<data_path>/validation/n01440764/*.JPEG``.
    Needs PyTorch (see Requirements in the README).

    Args:
        data_path (str): ImageNet root directory.
        alpha (float): width multiplier of the model the data is for (selects
            the crop ratio).
        split (str, optional): 'validation' or 'train'. Defaults to 'validation'.
        batch_size (int, optional): the batch size. Defaults to 128.
        normalize (bool, optional): True for PyTorch/ONNX models, False for
            Akida models (raw uint8). Defaults to True.
        num_samples (int, optional): keep only this many images. Defaults to all.
        seed (int, optional): if set, draw the images in a random order fixed by
            this seed rather than in class order. Defaults to None.
        num_workers (int, optional): DataLoader workers. Defaults to 8.
        resolution (int, optional): input resolution of the model the data is
            for, 224 or 256. Defaults to 224.

    Returns:
        torch.utils.data.DataLoader, int: batches of (images, labels) with
        images of shape (B, 3, H, W), and the number of images.
    """
    # PyTorch is only needed to read the full dataset, so it is imported here
    # rather than at module level: the sample pack and the benchmark run without it.
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(os.path.join(data_path, split),
                          transform=get_transform(alpha, input_size=resolution,
                                                  normalize=normalize))
    indices = np.arange(len(dataset))
    if seed is not None:
        indices = np.random.default_rng(seed).permutation(indices)
    if num_samples is not None:
        indices = indices[:num_samples]
    dataset = Subset(dataset, indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers,
                        pin_memory=torch.cuda.is_available())
    return loader, len(dataset)


def get_calibration_samples(data_path, alpha, num_samples=8192, batch_size=128, seed=0,
                            resolution=224):
    """Draws quantization calibration images at random from the train split.

    The draw is seeded, so the same seed always gives the same images.

    Returns:
        np.ndarray: float32 normalised images, shape (num_samples, 3, H, W).
    """
    loader, _ = get_data(data_path, alpha, split='train', batch_size=batch_size,
                         num_samples=num_samples, seed=seed, resolution=resolution)
    return np.concatenate([images.numpy() for images, _ in loader])


def get_labelled_samples(alpha, normalize=False, resolution=224):
    """Loads the 10-image ImageNet-like sample pack, with labels.

    Downloaded on first use and cached under ``~/.keras/datasets``. No ImageNet
    setup is required.

    Args:
        alpha (float): width multiplier of the model the data is for.
        normalize (bool, optional): True for PyTorch/ONNX models, False for
            Akida models (raw uint8). Defaults to False.
        resolution (int, optional): input resolution of the model the data is
            for, 224 or 256. Defaults to 224.

    Returns:
        np.ndarray, np.ndarray: images of shape (10, 3, H, W) and their integer
        class labels.
    """
    file_path = fetch_file(fname="imagenet_like.zip", origin=SAMPLES_URL,
                           cache_subdir='datasets/imagenet_like', extract=True)
    data_folder = os.path.dirname(file_path)

    # Labels file maps image filename -> class index
    labels = {}
    with open(os.path.join(data_folder, 'labels_validation.txt')) as f:
        for line in f:
            if line.strip():
                name, index = line.split()
                labels[name] = int(index)

    transform = get_transform(alpha, input_size=resolution, normalize=normalize)
    images, targets = [], []
    for idx in range(NUM_SAMPLE_IMAGES):
        fname = f'image_{idx + 1:02d}.jpg'
        image = Image.open(os.path.join(data_folder, fname)).convert('RGB')
        images.append(transform(image))
        targets.append(labels[fname])
    return np.stack(images), np.array(targets, dtype=np.int64)


def get_samples(alpha, num_samples=100, data_path=None, resolution=224):
    """Returns uint8 Akida input samples for benchmarking and sparsity measurement.

    By default this uses the 10-image sample pack, cycled up to
    ``num_samples``. Akida's latency depends on activation sparsity, which in
    turn depends on the input, so benchmarking must use real images rather than
    random noise -- but a handful of real images captures the activity
    statistics well enough for that purpose.

    Pass ``data_path`` to draw the samples from the ImageNet validation split
    instead, which gives a more representative spread (needs PyTorch).

    Args:
        alpha (float): width multiplier of the model the samples are for. The
            two widths crop differently, so the raw images differ too.
        num_samples (int, optional): number of samples. Defaults to 100.
        data_path (str, optional): if given, take samples from the ImageNet
            validation split at this path instead of the sample pack.
        resolution (int, optional): input resolution of the model the samples
            are for, 224 or 256. Defaults to 224.

    Returns:
        np.ndarray: uint8 array of shape (num_samples, 3, H, W).
    """
    if data_path is not None:
        loader, _ = get_data(data_path, alpha, batch_size=num_samples, normalize=False,
                             num_samples=num_samples, seed=0, resolution=resolution)
        images, _ = next(iter(loader))
        return images.numpy()

    images, _ = get_labelled_samples(alpha, normalize=False, resolution=resolution)
    repeats = int(np.ceil(num_samples / len(images)))
    return np.tile(images, (repeats, 1, 1, 1))[:num_samples]
