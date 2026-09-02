#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License

import numpy as np
import tensorflow as tf
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.utils import set_random_seed

# Define the base directory for the VWW dataset

VALIDATION_SPLIT= 0.1

def get_data(data_path, input_shape, batch_size, seed=42):
    """ Loads VWW data.

    Args:
        data_path (str): path to data
        input_shape (tuple): input image shape (height, width, channels)
        batch_size (int): the batch size
        dtype (tf.dtypes.DType, optional): input data type. Defaults to tf.uint8.

    Returns:
        tf_keras.data.Dataset, tf_keras.data.Dataset: training dataset, validation dataset
    """
    set_random_seed(seed)

    # Set aside .1 split for validation
    validation_split = VALIDATION_SPLIT

    # Create a data generator with data augmentation and load files from
    # directory
    train_datagen = ImageDataGenerator(
        rotation_range = 10,
        width_shift_range = 0.05,
        height_shift_range = 0.05,
        zoom_range = 0.1,
        horizontal_flip = True,
        validation_split = validation_split)
    
    train_generator = train_datagen.flow_from_directory(
        data_path,
        target_size=input_shape[:2],
        batch_size=batch_size,
        subset = 'training',
        color_mode='rgb',
        class_mode = 'sparse',
        shuffle=True)

    val_datagen = ImageDataGenerator(
        validation_split = validation_split)
    
    val_generator = val_datagen.flow_from_directory(
        data_path,
        target_size=input_shape[:2],
        batch_size=batch_size,
        subset = 'validation',
        color_mode='rgb',
        class_mode = 'sparse',
        shuffle=False)

    return train_generator, val_generator 

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
    # Set aside .1 split for validation
    validation_split = VALIDATION_SPLIT
    # Create a data generator without data augmentation, but taking
    # from the same training split as the main dataset generator
    train_generator = ImageDataGenerator(validation_split = validation_split).flow_from_directory(
        data_path,
        target_size=input_shape[:2],
        batch_size=num_samples,
        subset = 'training',
        color_mode='rgb',
        class_mode = 'sparse',
        shuffle=True)

    images, _ = next(train_generator)
    return images[:num_samples].astype(np.uint8)
