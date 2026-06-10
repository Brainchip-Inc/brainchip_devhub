#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License

import os
import numpy as np
import tensorflow as tf
from tf_keras.preprocessing.image import ImageDataGenerator


def get_data(data_path, input_shape, batch_size, dtype=tf.uint8):
    """ Loads VWW data.

    Args:
        data_path (str): path to data
        input_shape (tuple): input image shape (height, width, channels)
        batch_size (int): the batch size
        dtype (tf.dtypes.DType, optional): input data type. Defaults to tf.uint8.

    Returns:
        tf_keras.data.Dataset, tf_keras.data.Dataset: training dataset, validation dataset
    """

    def cast_data(image, label):
        image = tf.cast(image, dtype)
        return image, label

    # Create a data generator with data augmentation and load files from
    # directory
    datagen = ImageDataGenerator(rotation_range=10,
                                 width_shift_range=0.05,
                                 height_shift_range=0.05,
                                 zoom_range=.1,
                                 horizontal_flip=True)

    train_generator = datagen.flow_from_directory(
        os.path.join(data_path, 'train'),
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='sparse')

    train_dataset = tf.data.Dataset.from_generator(
        lambda: iter(train_generator),
        output_signature=(
            tf.TensorSpec(shape=(None,) + input_shape, dtype=dtype),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),)
    ).take(len(train_generator)).map(
        cast_data, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    val_generator = ImageDataGenerator().flow_from_directory(
        os.path.join(data_path, 'val'),
        target_size=input_shape[:2],
        batch_size=batch_size,
        shuffle=False,
        class_mode='sparse')

    val_dataset = tf.data.Dataset.from_generator(
        lambda: iter(val_generator),
        output_signature=(
            tf.TensorSpec(shape=(None,) + input_shape, dtype=dtype),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),)
    ).take(len(val_generator)).map(
        cast_data, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset



def get_samples(data_path, input_shape, num_samples=1024):
    """ Loads image samples from the train split as a numpy array.

    No augmentation is applied; images are only resized to input_shape.
    Suitable for model calibration and testing.

    Args:
        data_path (str): path to data
        input_shape (tuple): input image shape (height, width, channels)
        num_samples (int): number of samples to return. Defaults to 1024.

    Returns:
        np.ndarray: array of shape (num_samples, height, width, channels), dtype uint8
    """
    generator = ImageDataGenerator().flow_from_directory(
        os.path.join(data_path, 'train'),
        target_size=input_shape[:2],
        batch_size=num_samples,
        shuffle=False)

    images, _ = next(generator)
    return images[:num_samples].astype(np.uint8)
