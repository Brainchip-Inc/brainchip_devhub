#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Data loading for the Jester TENN example (Akida 2).

Two sources, both mirrored by BrainChip and downloaded on first use into
``data/`` (git-ignored):

* **The BrainChip Jester subset** (:data:`SUBSET_URL`, about 100 MB): one TFRecord
  shard of the Jester validation split, 541 clips covering all 27 classes, plus the
  label and split CSV files. The accuracy figures in the README are measured on it;
  it is not the full validation split. :func:`get_data` reads it, or any directory
  with the same layout, so the full validation set can be dropped in with ``-d``::

      <dir>/jester-v1-labels.csv
      <dir>/tfrecords/*validation*     GZIP TFRecords, one SequenceExample per clip

* **The calibration pack** (:data:`CALIBRATION_URL`): 100 clips of 16 frames from
  the Jester train split, stored as one ``(1600, 100, 100, 3)`` uint8 array in
  clip order. It is the file BrainChip calibrated the published 8-bit model on.

Labels are integer class indices in [0, 26], in the order of
``jester-v1-labels.csv``.

Run as a script to check that ``jester_preprocessing`` is bit-identical to the
``akida_models`` pipeline over every clip of a dataset directory::

    python jester_data.py --check-preprocessing
"""
import argparse
import csv
import os
import pathlib
from glob import glob

import numpy as np
import tensorflow as tf
from akida_models.utils import fetch_file

from jester_preprocessing import (FRAMES_PER_CLIP, FRAME_SIZE, parse_sequence_example,
                                  decode_clip)

__all__ = ["NUM_CLASSES", "SUBSET_URL", "CALIBRATION_URL", "get_data_dir", "read_labels",
           "count_clips", "get_data", "get_clips", "get_stream", "get_calibration_path",
           "get_calibration_samples", "check_preprocessing"]

NUM_CLASSES = 27

SUBSET_URL = "https://data.brainchip.com/dataset-mirror/jester/jester_subset.tar.gz"
SUBSET_SHA256 = "af55f22ead4e2ca086038043457400b97a447f3796fb9c78ef4c497250e6155b"
CALIBRATION_URL = ("https://data.brainchip.com/dataset-mirror/samples/jester_video/"
                   "jester_video_bs100.npz")
CALIBRATION_SHA256 = "d0ba505971075dc8bd2e0b0fecf4c301e775cdcf1a06e2e7713513303575309a"
# The calibration pack holds this many clips; quantization must be told so (batch_size)
CALIBRATION_CLIPS = 100

DATA_DIR = pathlib.Path(__file__).parent / 'data'


def get_data_dir(data_path=None):
    """Returns the dataset directory, downloading the BrainChip subset if none is given.

    Args:
        data_path (str, optional): a Jester directory (see the module docstring).
            If None, the BrainChip Jester subset is downloaded into ``data/`` on
            first use.

    Returns:
        str: the dataset directory.
    """
    if data_path:
        return str(data_path)
    data_dir = DATA_DIR.resolve() / 'jester_subset'
    # fetch_file re-extracts the archive on every call, so only call it when the
    # extracted subset is missing.
    if not (data_dir / '.extracted').exists():
        fetch_file(SUBSET_URL, fname='jester_subset.tar.gz', file_hash=SUBSET_SHA256,
                   cache_subdir=str(DATA_DIR.resolve()), extract=True)
        (data_dir / '.extracted').touch()
    return str(data_dir)


def read_labels(data_dir):
    """Returns the 27 class names, in label-index order."""
    with open(os.path.join(data_dir, 'jester-v1-labels.csv')) as f:
        return [row[0] for row in csv.reader(f)]


def _records(data_dir):
    filenames = sorted(glob(os.path.join(data_dir, 'tfrecords', '*validation*')))
    if not filenames:
        raise FileNotFoundError(f'No validation TFRecords under {data_dir}/tfrecords')
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def count_clips(data_dir):
    """Counts the clips in the validation TFRecords.

    The count comes from the records themselves: ``jester-v1-validation.csv``
    lists the whole validation split, even in the subset.
    """
    return int(_records(data_dir).reduce(np.int64(0), lambda n, _: n + 1))


def _decode(example):
    frames, label, frame_count = parse_sequence_example(example)
    return decode_clip(frames, frame_count), tf.cast(label, tf.int32)


def get_data(data_dir, batch_size=32):
    """Loads the validation clips as a batched tf.data pipeline.

    Args:
        data_dir (str): dataset directory (see :func:`get_data_dir`).
        batch_size (int, optional): clips per batch. Defaults to 32.

    Returns:
        tuple: the dataset, yielding (clips, labels) with clips of shape
        (B, 16, 100, 100, 3) uint8 and labels of shape (B,) int32, and the number
        of clips.
    """
    dataset = _records(data_dir).map(_decode, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE), count_clips(data_dir)


def get_clips(data_dir, num_clips, one_per_class=False):
    """Returns clips as numpy arrays.

    Args:
        data_dir (str): dataset directory.
        num_clips (int): number of clips to return.
        one_per_class (bool, optional): take the first clip of each class in turn
            rather than the first ``num_clips`` clips. Defaults to False.

    Returns:
        tuple: clips (N, 16, 100, 100, 3) uint8 and labels (N,) int.
    """
    clips, labels, seen = [], [], set()
    for clip, label in _records(data_dir).map(_decode):
        label = int(label)
        if one_per_class and label in seen:
            continue
        seen.add(label)
        clips.append(clip.numpy())
        labels.append(label)
        if len(clips) == num_clips:
            break
    return np.stack(clips), np.array(labels)


def get_stream(data_dir, num_clips):
    """Returns consecutive clips as one frame stream, for the hardware benchmark.

    Returns:
        np.ndarray: (num_clips * 16, 100, 100, 3) uint8 frames, clip after clip.
    """
    clips, _ = get_clips(data_dir, num_clips)
    return clips.reshape(-1, *clips.shape[2:])


def get_calibration_path():
    """Returns the path of the calibration pack, downloading it into ``data/`` on first use."""
    return fetch_file(CALIBRATION_URL, fname='jester_video_bs100.npz',
                      file_hash=CALIBRATION_SHA256, cache_subdir=str(DATA_DIR.resolve()))


def get_calibration_samples():
    """Returns the calibration pack, downloading it into ``data/`` on first use.

    Returns:
        np.ndarray: (1600, 100, 100, 3) uint8 frames: 100 train clips of 16 frames,
        clip after clip.
    """
    with np.load(get_calibration_path()) as data:
        return np.concatenate([data[k] for k in data.files])


def check_preprocessing(data_dir):
    """Asserts that jester_preprocessing matches akida_models over every clip.

    Returns:
        int: the number of clips compared.
    """
    from akida_models.tenn_spatiotemporal.jester_train import get_data as reference_get_data

    reference, _ = reference_get_data('val', data_dir, FRAMES_PER_CLIP, FRAME_SIZE,
                                      batch_size=1)
    ours = _records(data_dir).map(_decode)
    n = 0
    for (ref_clip, ref_onehot), (clip, label) in zip(reference, ours):
        assert np.array_equal(ref_clip.numpy()[0], clip.numpy()), f'clip {n} differs'
        assert int(np.argmax(ref_onehot[0])) == int(label), f'label {n} differs'
        n += 1
    assert n == count_clips(data_dir), 'the pipelines yield different clip counts'
    return n


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Jester data utilities')
    parser.add_argument('-d', '--data', default=None,
                        help='Jester directory. Defaults to the BrainChip subset, '
                             'downloaded into data/.')
    parser.add_argument('--check-preprocessing', action='store_true',
                        help='Check jester_preprocessing against akida_models')
    parser.add_argument('--fetch-calibration', action='store_true',
                        help='Download the calibration pack and print its path')
    args = parser.parse_args()

    if args.fetch_calibration:
        print(get_calibration_path())
        raise SystemExit

    data_dir = get_data_dir(args.data)
    print(f'{data_dir}: {count_clips(data_dir)} clips, {len(read_labels(data_dir))} classes')
    if args.check_preprocessing:
        print(f'Preprocessing bit-identical to akida_models over '
              f'{check_preprocessing(data_dir)} clips')
