#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Data loading for the Portrait128 person-segmentation dataset.

The dataset is two NumPy arrays in one folder:
  * img_uint8.npy: RGB portraits, shape (N, 128, 128, 3), uint8
  * msk_uint8.npy: person masks, shape (N, 128, 128, 1), uint8, 255 = person,
    0 = background

For training, both arrays are loaded into RAM (about 1.2 GB); calibration
samples are read straight from disk.

The dataset ships as a single pool, so it is split here: 80 % training and 20 %
validation, by a shuffle with a fixed seed (SPLIT_SEED). That seed is
independent of the training seed, so every script and notebook sees the same
validation images whatever --seed is used for training.

Images are delivered as uint8 in [0, 255]: the model rescales them itself. Masks
are delivered as int32 0/1 labels with the same spatial shape as the image.

Run as a script, it writes a calibration-sample pack for `quantizeml quantize`:

    python portrait128_data.py --extract data/portrait128_batch1024.npz
"""

import argparse
import os

import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras.utils import set_random_seed

IMAGES_FILE = 'img_uint8.npy'
MASKS_FILE = 'msk_uint8.npy'

VALIDATION_SPLIT = 0.2
# Fixed seed for the train/validation split, independent of the training seed
SPLIT_SEED = 1


def _load_arrays(data_path):
    """Memory-map the image and mask arrays."""
    x = np.load(os.path.join(data_path, IMAGES_FILE), mmap_mode='r')
    y = np.load(os.path.join(data_path, MASKS_FILE), mmap_mode='r')
    return x, y


def _check_shape(x, input_shape):
    if tuple(x.shape[1:]) != tuple(input_shape):
        raise ValueError(f'Portrait128 images have shape {tuple(x.shape[1:])}, '
                         f'but the model expects {tuple(input_shape)}.')


def _split_indices(num_samples, validation_split=VALIDATION_SPLIT, random_state=SPLIT_SEED):
    """Shuffle sample indices and split them into training and validation sets.

    Args:
        num_samples (int): number of samples in the dataset.
        validation_split (float, optional): fraction used for validation. Defaults to 0.2.
        random_state (int, optional): seed of the shuffle. Defaults to SPLIT_SEED.

    Returns:
        np.ndarray, np.ndarray: training indices, validation indices
    """
    rng = np.random.default_rng(random_state)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    validation_size = int(validation_split * num_samples)
    return indices[validation_size:], indices[:validation_size]


def get_data(data_path, input_shape, batch_size, dtype=tf.uint8, seed=42):
    """ Loads Portrait128 data.

    Training images are augmented with random horizontal flips, translations
    (up to 10 % in each direction) and zooms (up to 20 %). Each image and its
    mask are stacked into one 4-channel tensor before augmentation, so both
    receive exactly the same geometric transform.

    Args:
        data_path (str): folder containing img_uint8.npy and msk_uint8.npy
        input_shape (tuple): input image shape (height, width, channels)
        batch_size (int): the batch size
        dtype (tf.dtypes.DType, optional): input data type. Defaults to tf.uint8.
        seed (int, optional): seed for shuffling and augmentation. Defaults to 42.

    Returns:
        tf.data.Dataset, tf.data.Dataset: training dataset, validation dataset
    """
    set_random_seed(seed)

    # Load both arrays into RAM. Masks are stored as 0/255; integer division
    # turns them into 0/1 labels once, here.
    x, y = _load_arrays(data_path)
    _check_shape(x, input_shape)
    x = np.asarray(x)
    y = np.asarray(y) // 255
    train_indices, val_indices = _split_indices(len(x))

    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=seed),
        keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, seed=seed),
        keras.layers.RandomZoom(0.2, seed=seed),
    ])

    def cast_data(image, label):
        image = tf.cast(image, dtype)
        label = tf.cast(label, tf.int32)
        return image, label

    def data_generator(indices):
        for i in indices:
            yield x[i], y[i]

    def apply_data_augmentation(image, mask):
        # Concatenate image and mask along the last axis
        combined = tf.concat([image, mask], axis=-1)
        # Apply the same augmentation to both image and mask
        augmented_combined = data_augmentation(combined)
        # Split the augmented combined tensor back into image and mask
        augmented_image, augmented_mask = tf.split(augmented_combined, [3, 1], axis=-1)

        return augmented_image, augmented_mask

    output_signature = (tf.TensorSpec(shape=x.shape[1:], dtype=tf.uint8),
                        tf.TensorSpec(shape=y.shape[1:], dtype=tf.uint8))

    train_dataset = tf.data.Dataset.from_generator(
        generator=lambda: data_generator(train_indices),
        output_signature=output_signature)

    validation_dataset = tf.data.Dataset.from_generator(
        generator=lambda: data_generator(val_indices),
        output_signature=output_signature)

    SHUFFLE_BUFFER_SIZE = 1000
    train_batches = (train_dataset
                     .shuffle(SHUFFLE_BUFFER_SIZE, seed=seed)
                     .map(apply_data_augmentation)
                     .map(cast_data, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
                     .batch(batch_size)
                     .prefetch(buffer_size=tf.data.AUTOTUNE))

    validation_batches = (validation_dataset
                          .map(cast_data, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
                          .batch(batch_size)
                          .prefetch(buffer_size=tf.data.AUTOTUNE))

    # The datasets come from Python generators, so their length is unknown to
    # tf.data. Declaring it lets Keras show per-epoch progress and lets callers
    # use len().
    train_batches = train_batches.apply(
        tf.data.experimental.assert_cardinality(-(-len(train_indices) // batch_size)))
    validation_batches = validation_batches.apply(
        tf.data.experimental.assert_cardinality(-(-len(val_indices) // batch_size)))

    return train_batches, validation_batches


def get_samples(data_path, input_shape, num_samples=1024):
    """ Loads image samples from the training split as a numpy array.

    No augmentation is applied. Suitable for model calibration and testing.

    Args:
        data_path (str): folder containing img_uint8.npy
        input_shape (tuple): input image shape (height, width, channels)
        num_samples (int): number of samples to return. Defaults to 1024.

    Returns:
        np.ndarray: array of shape (num_samples, height, width, channels), dtype uint8
    """
    x = np.load(os.path.join(data_path, IMAGES_FILE), mmap_mode='r')
    _check_shape(x, input_shape)
    train_indices, _ = _split_indices(len(x))

    # The training indices are already in shuffled order; sort the selection so
    # the memory-mapped reads are sequential.
    selection = np.sort(train_indices[:num_samples])
    return np.asarray(x[selection], dtype=np.uint8)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Write Portrait128 calibration samples to an .npz file')
    parser.add_argument('--extract', required=True,
                        help='Output .npz path')
    parser.add_argument('-d', '--data', default='./data/portrait128',
                        help='Portrait128 folder (contains img_uint8.npy and msk_uint8.npy)')
    parser.add_argument('-n', '--num_samples', type=int, default=1024,
                        help='Number of training-split images to save')
    args = parser.parse_args()

    samples = get_samples(args.data, (128, 128, 3), num_samples=args.num_samples)
    # Saved as a single 'data' array, the layout `quantizeml quantize --samples` reads
    np.savez(args.extract, data=samples)
    print(f'{len(samples)} samples saved to {args.extract}')
