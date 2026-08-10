#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tf_keras.utils import set_random_seed
from tensorflow_datasets.datasets.plant_village import plant_village_dataset_builder

plant_village_dataset_builder._URL = "https://data.brainchip.com/dataset-mirror/plantvillage/Plant_leaf_diseases_dataset_without_augmentation.zip"

def get_data(data_path, input_shape, batch_size, dtype=tf.uint8, seed=42):
    """ Loads PlantVillage data via tensorflow_datasets.

    Args:
        data_path (str): directory used as the tfds data_dir (dataset is
            downloaded here on first call)
        input_shape (tuple): input image shape (height, width, channels)
        batch_size (int): the batch size
        dtype (tf.dtypes.DType, optional): input data type. Defaults to tf.uint8.

    Returns:
        tf.data.Dataset, tf.data.Dataset: training dataset, validation dataset
    """
    set_random_seed(seed)

    h, w = input_shape[:2]

    def resize_and_cast(image, label):
        image = tf.image.resize(image, (h, w))
        image = tf.cast(image, dtype)
        return image, tf.cast(label, tf.float32)

    def augment(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        image = tf.clip_by_value(image, 0, 255)
        image = tf.cast(image, dtype)
        return image, label

    tfds.disable_progress_bar()
    # raw_train, raw_val = tfds.load(
    #     'plant_village',
    #     split=['train[:80%]', 'train[80%:90%]'],
    #     as_supervised=True,
    #     data_dir=data_path)
    raw_train, raw_val, raw_test = tfds.load(
        'plant_village',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        as_supervised=True,
        data_dir=data_path)

    train_dataset = (raw_train
                     .map(resize_and_cast, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
                     .map(augment, num_parallel_calls=1)
                     .shuffle(1000, seed=seed)
                     .batch(batch_size)
                     .prefetch(tf.data.AUTOTUNE))

    val_dataset = (raw_val
                   .map(resize_and_cast, num_parallel_calls=tf.data.AUTOTUNE)
                   .batch(batch_size)
                   .prefetch(tf.data.AUTOTUNE))
    
    test_dataset = (raw_test
                   .map(resize_and_cast, num_parallel_calls=tf.data.AUTOTUNE)
                   .batch(batch_size)
                   .prefetch(tf.data.AUTOTUNE))

    return train_dataset, val_dataset, test_dataset


def get_samples(data_path, input_shape, num_samples=1024):
    """ Loads image samples from the train split as a numpy array.

    No augmentation is applied; images are only resized to input_shape.
    Suitable for model calibration and testing.

    Args:
        data_path (str): directory used as the tfds data_dir
        input_shape (tuple): input image shape (height, width, channels)
        num_samples (int): number of samples to return. Defaults to 1024.

    Returns:
        np.ndarray: array of shape (num_samples, height, width, channels), dtype uint8
    """
    h, w = input_shape[:2]

    tfds.disable_progress_bar()
    ds = tfds.load(
        'plant_village',
        split='train[:80%]',
        as_supervised=True,
        data_dir=data_path)

    samples = []
    for image, _ in ds.take(num_samples):
        image = tf.image.resize(image, (h, w))
        samples.append(image.numpy().astype(np.uint8))

    return np.array(samples[:num_samples])
