#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License

"""PlantVillage data loading.

Loads PlantVillage directly from the authors' official GitHub repository
(spMohanty/PlantVillage-Dataset) instead of tensorflow_datasets. The TFDS
builder's download URL (hosted on Mendeley) began returning HTTP 403 to
automated clients, which broke `tfds.load('plant_village')` everywhere.

The images are pulled once from a plain GitHub archive zip (no auth, no bot
blocking), extracted to a class-per-folder layout, and read with pure
TensorFlow (`image_dataset_from_directory`). Folder names are the class
labels in the familiar "Apple___Black_rot" scheme, so the 38-class setup
matches what the model expects.

The returned tf.data.Dataset objects and the (image, label) element spec are
the same as before, so the rest of the pipeline is unchanged.
"""

import os
import zipfile
import urllib.request

import numpy as np
import tensorflow as tf
from tf_keras.utils import set_random_seed

# Plain, reliable, wget-able archive of the official dataset repo.
_ARCHIVE_URL = "https://github.com/spMohanty/PlantVillage-Dataset/archive/refs/heads/master.zip"

# After extraction, color images live here (class-per-subfolder).
_COLOR_SUBDIR = os.path.join("PlantVillage-Dataset-master", "raw", "color")

# The 38 classes, in the exact order used by the original TFDS builder, so
# integer label indices match what the model was trained/evaluated against.
_LABELS = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def _ensure_data(data_path):
    """Download + extract the color image tree once. Returns the color dir path.

    Idempotent: if the color directory already exists under data_path, does
    nothing and returns immediately.
    """
    os.makedirs(data_path, exist_ok=True)
    color_dir = os.path.join(data_path, _COLOR_SUBDIR)

    if os.path.isdir(color_dir):
        return color_dir

    zip_path = os.path.join(data_path, "plantvillage_master.zip")
    if not os.path.exists(zip_path):
        print("Downloading PlantVillage images from GitHub...")
        urllib.request.urlretrieve(_ARCHIVE_URL, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_path)

    if not os.path.isdir(color_dir):
        raise RuntimeError(
            f"Expected color images at {color_dir} after extraction, but the "
            f"directory was not found. Check the archive layout."
        )
    return color_dir


def get_data(data_path, input_shape, batch_size, dtype=tf.uint8, seed=42):
    """Loads PlantVillage data from the official GitHub image repository.

    Args:
        data_path (str): directory where the dataset is downloaded/extracted.
        input_shape (tuple): input image shape (height, width, channels).
        batch_size (int): the batch size.
        dtype (tf.dtypes.DType, optional): input data type. Defaults to tf.uint8.
        seed (int): random seed.

    Returns:
        tf.data.Dataset, tf.data.Dataset, tf.data.Dataset:
            training, validation, and test datasets
    """
    set_random_seed(seed)
    h, w = input_shape[:2]
    color_dir = _ensure_data(data_path)

    # Read the whole dataset once, unbatched, with labels derived from folder
    # names. class_names is pinned to _LABELS so label indices match the model.
    #
    # IMPORTANT: shuffle=False here. image_dataset_from_directory's shuffle uses
    # reshuffle_each_iteration=True, which reshuffles on every pass. Because we
    # iterate the dataset multiple times (take/skip for the three splits), that
    # caused images to leak across train/val/test. Instead we load in a fixed
    # order and apply a single, fixed shuffle below, then slice deterministically
    # so the three splits are guaranteed disjoint and reproducible.
    full = tf.keras.utils.image_dataset_from_directory(
        color_dir,
        labels="inferred",
        label_mode="int",
        class_names=_LABELS,          # enforce the exact 38 classes and order
        color_mode="rgb",
        batch_size=None,              # unbatched; we split then batch below
        image_size=(h, w),
        shuffle=False,                # fixed order; we shuffle once, deterministically, below
    )

    # Single fixed shuffle (no reshuffle across iterations), then deterministic
    # slicing into disjoint train/val/test.
    full = full.shuffle(buffer_size=full.cardinality(), seed=seed,
                        reshuffle_each_iteration=False)

    def cast_img(image, label):
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

    # 80/10/10 split over the shuffled dataset, matching prior behavior.
    n = tf.data.experimental.cardinality(full).numpy()
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    raw_train = full.take(n_train)
    raw_val = full.skip(n_train).take(n_val)
    raw_test = full.skip(n_train + n_val)

    train_dataset = (raw_train
                     .map(cast_img, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
                     .map(augment, num_parallel_calls=1)
                     .shuffle(1000, seed=seed)
                     .batch(batch_size)
                     .prefetch(tf.data.AUTOTUNE))

    val_dataset = (raw_val
                   .map(cast_img, num_parallel_calls=tf.data.AUTOTUNE)
                   .batch(batch_size)
                   .prefetch(tf.data.AUTOTUNE))

    test_dataset = (raw_test
                    .map(cast_img, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(batch_size)
                    .prefetch(tf.data.AUTOTUNE))

    return train_dataset, val_dataset, test_dataset


def get_samples(data_path, input_shape, num_samples=1024):
    """Loads image samples from the train portion as a numpy array.

    No augmentation is applied; images are only resized to input_shape.
    Suitable for model calibration and testing.

    Args:
        data_path (str): directory where the dataset is downloaded/extracted.
        input_shape (tuple): input image shape (height, width, channels).
        num_samples (int): number of samples to return. Defaults to 1024.

    Returns:
        np.ndarray: array of shape (num_samples, height, width, channels), uint8
    """
    h, w = input_shape[:2]
    color_dir = _ensure_data(data_path)

    full = tf.keras.utils.image_dataset_from_directory(
        color_dir,
        labels="inferred",
        label_mode="int",
        class_names=_LABELS,
        color_mode="rgb",
        batch_size=None,
        image_size=(h, w),
        shuffle=False,
    )
    full = full.shuffle(buffer_size=full.cardinality(), seed=42,
                        reshuffle_each_iteration=False)

    samples = []
    for image, _ in full.take(num_samples):
        samples.append(tf.cast(image, tf.uint8).numpy())

    return np.array(samples[:num_samples])
