#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Loads the PASCAL VOC (2007 + 2012) dataset, restricted to the 'car' and
'person' classes, and prepares it for YOLOv2 training/evaluation.

VOC images and annotations are read through tensorflow_datasets, which
downloads and prepares the archives itself into data_path on first use
(see -d/--data in the scripts that use this module). Anchors are the
fixed set used for the released Akida 1 VOC detector and must be reused
as-is (not regenerated), so that decoded model outputs stay interpretable
against the grid/anchor layout the model and loss were built with.
"""

import pickle

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from akida_models.utils import fetch_file
from akida_models.detection.data_augmentation import build_yolo_aug_pipeline
from akida_models.detection.preprocess_data import preprocess_dataset
from akida_models.detection.processing import create_yolo_targets

LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                                  'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                                  'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                                  'train', 'tvmonitor']
GRID_SIZE = (7, 7)

ANCHORS_URL = 'https://data.brainchip.com/dataset-mirror/voc/voc_anchors_v1.pkl'
ANCHORS_HASH = 'b1fe1ed12691e100646cf52b1320f05abd17b2f546d3e12cdee87758cc9ed0ba'


def get_voc_dataset(data_path, labels=LABELS, training=False):
    """ Loads the VOC dataset through tfds, downloading it if needed.

    Args:
        data_path (str): directory tfds should use to download/cache VOC
        labels (list[str], optional): labels of interest. Defaults to LABELS.
        training (bool, optional): True for the train+validation split,
            False for the test split. Defaults to False.

    Returns:
        tf.data.Dataset, list[str], int: the requested dataset, the list of
        labels and the dataset size. Each sample is a dict with 'image' and
        'objects' (with 'bbox' and 'label', remapped to indices into
        `labels`), restricted to objects of interest and excluding
        difficult ones.
    """
    split = 'train+validation' if training else 'test'
    load_kwargs = dict(split=split, with_info=True, shuffle_files=training, data_dir=data_path)

    if training:
        dataset_2012, info = tfds.load('voc/2012', **load_kwargs)
        dataset_2007, _ = tfds.load('voc/2007', **load_kwargs)
        dataset = dataset_2012.concatenate(dataset_2007)
    else:
        dataset, info = tfds.load('voc/2007', **load_kwargs)

    voc_labels = info.features['labels'].names
    dataset = (dataset
               .map(lambda sample: _select_labels(sample, labels, voc_labels))
               .filter(_has_objects))

    num_samples = sum(1 for _ in dataset)

    return dataset, labels, num_samples


def _select_labels(sample, labels_of_interest, voc_labels):
    """ Restricts a VOC sample's objects to labels_of_interest, remapped to
    their index within that list, and drops difficult objects. """
    label_indices = tf.constant([voc_labels.index(label) for label in labels_of_interest],
                               dtype=tf.int64)

    objects = sample['objects']
    mask = tf.logical_and(
        tf.reduce_any(tf.equal(objects['label'][:, tf.newaxis], label_indices), axis=1),
        tf.logical_not(objects['is_difficult']))

    remapped_labels = tf.argmax(
        tf.equal(objects['label'][mask][:, tf.newaxis], label_indices), axis=1)

    return {
        'image': sample['image'],
        'objects': {
            'bbox': objects['bbox'][mask],
            'label': remapped_labels,
        },
    }


def _has_objects(sample):
    return tf.shape(sample['objects']['bbox'])[0] > 0


def get_anchors():
    """ Fetches the fixed anchor boxes used by the Akida 1 VOC YOLO model.

    Returns:
        list: list of [width, height] anchor boxes, in grid-cell units.
    """
    anchors_path = fetch_file(ANCHORS_URL,
                              fname='voc_anchors_v1.pkl',
                              file_hash=ANCHORS_HASH,
                              cache_subdir='datasets/voc')
    with open(anchors_path, 'rb') as handle:
        return pickle.load(handle)


def get_data(data_path, input_shape, batch_size, seed=42):
    """ Loads VOC data, ready for YOLOv2 training.

    Args:
        data_path (str): path to the folder containing the VOC tar archives
        input_shape (tuple): input image shape (height, width, channels)
        batch_size (int): the batch size
        seed (int, optional): random seed for reproducibility. Defaults to 42.

    Returns:
        tf.data.Dataset, tf.data.Dataset, int: training dataset, validation
        dataset and the number of training images. Each dataset yields
        (image, targets) batches, where targets are already encoded in the
        grid/anchor format expected by YoloLoss. The training dataset
        repeats indefinitely, so `num_train` is needed to size
        `steps_per_epoch`.
    """
    tf.random.set_seed(seed)

    anchors = get_anchors()
    aug_pipe = build_yolo_aug_pipeline()

    train_data, labels, num_train = get_voc_dataset(data_path, labels=LABELS, training=True)
    val_data, _, _ = get_voc_dataset(data_path, labels=LABELS, training=False)

    train_dataset = preprocess_dataset(train_data, input_shape=input_shape, grid_size=GRID_SIZE,
                                       labels=labels, batch_size=batch_size, aug_pipe=aug_pipe,
                                       create_targets_fn=create_yolo_targets, training=True,
                                       anchors=anchors)
    val_dataset = preprocess_dataset(val_data, input_shape=input_shape, grid_size=GRID_SIZE,
                                     labels=labels, batch_size=batch_size, aug_pipe=aug_pipe,
                                     create_targets_fn=create_yolo_targets, training=False,
                                     anchors=anchors)

    return train_dataset, val_dataset, num_train


def get_samples(data_path, input_shape, num_samples=1024):
    """ Loads image samples from the validation split as a numpy array.

    No augmentation is applied; images are only resized to input_shape.
    Suitable for model calibration, sparsity analysis and benchmarking.

    Args:
        data_path (str): path to the folder containing the VOC tar archives
        input_shape (tuple): input image shape (height, width, channels)
        num_samples (int): number of samples to return. Defaults to 1024.

    Returns:
        np.ndarray: array of shape (num_samples, height, width, channels), dtype uint8
    """
    anchors = get_anchors()
    aug_pipe = build_yolo_aug_pipeline()

    data, labels, num_available = get_voc_dataset(data_path, labels=LABELS, training=False)
    num_samples = min(num_samples, num_available)

    dataset = preprocess_dataset(data, input_shape=input_shape, grid_size=GRID_SIZE,
                                 labels=labels, batch_size=num_samples, aug_pipe=aug_pipe,
                                 create_targets_fn=create_yolo_targets, training=False,
                                 anchors=anchors)
    images, _ = next(iter(dataset))
    return images.numpy()[:num_samples].astype(np.uint8)
