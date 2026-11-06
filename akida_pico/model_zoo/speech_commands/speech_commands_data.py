#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Speech Commands data for the Akida Pico keyword spotting example.

The model works on raw audio, not on MFCC features: each 1 s clip (16 kHz, int16) is
padded or cropped to 16384 samples (2^14) and fed to the network as a sequence.

Data comes from the TFDS `speech_commands` dataset (v0.0.3): 10 keywords plus
`_silence_` and `_unknown_`, 12 classes in all. TFDS downloads and prepares it on
first use (several GB) into the directory passed as `data_dir`.

Splits:
    - train: class-balanced, re-drawn every epoch (3100 clips per class), because
      `_unknown_` is far larger than the other classes and `_silence_` far smaller.
    - validation: used to pick the best epoch during training.
    - test: used for every reported number.

Two views of a clip are used:
    - full length, shape (16384, 1), for the float model as trained;
    - split into consecutive chunks of `timestep` samples, for the stateful model,
      which processes the clip chunk by chunk and keeps its state between chunks.

Usage (sanity check):
    python speech_commands_data.py [-d DATA_DIR]
"""
import argparse

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from akida_models.tenn_recurrent.sc12_training_utils import build_balanced_indices

CLASS_NAMES = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes',
               '_silence_', '_unknown_']
NUM_CLASSES = len(CLASS_NAMES)
SIGNAL_LENGTH = 16384           # samples per clip fed to the model (~1.02 s at 16 kHz)
TARGET_PER_CLASS = 3100         # balanced training: clips per class per epoch
SAMPLES_URL = ('https://data.brainchip.com/dataset-mirror/samples/sc12/'
               'sc12_batch100_1024samples.npz')


def _pad_or_crop(audio, length=SIGNAL_LENGTH):
    """Zero-pads a clip at the end, or keeps its last `length` samples."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.shape[-1] < length:
        return np.pad(audio, (0, length - audio.shape[-1]))
    return audio[-length:]


def load_split(data_dir, split, length=SIGNAL_LENGTH):
    """Loads one TFDS split as arrays.

    Returns:
        (np.ndarray, np.ndarray): clips of shape (N, length, 1), float32 holding int16
        values, and integer labels of shape (N,).
    """
    ds = tfds.load('speech_commands', split=split, data_dir=data_dir, shuffle_files=False,
                   download=False)
    clips, labels = [], []
    for example in ds:
        clips.append(_pad_or_crop(example['audio'].numpy(), length))
        labels.append(example['label'].numpy())
    return np.stack(clips)[..., np.newaxis], np.array(labels, dtype=np.int64)


def _to_chunks(timestep):
    """Splits a batch of clips into consecutive chunks, time-major.

    (B, L, 1) -> (L // timestep, B, timestep, 1), with the label repeated per chunk.
    After `unbatch()`, the stateful model receives every chunk of a batch of clips in
    order, so it can carry its state from one chunk to the next.
    """
    def split(x, y):
        segments = SIGNAL_LENGTH // timestep
        x = tf.reshape(x, (-1, segments, timestep, 1))
        x = tf.transpose(x, (1, 0, 2, 3))
        y = tf.tile([y], multiples=[segments, 1])
        return x, y
    return split


def get_data(data_dir, batch_size=128, timestep=SIGNAL_LENGTH, seed=42):
    """Builds the training and validation datasets.

    Args:
        data_dir (str): TFDS data_dir holding `speech_commands`.
        batch_size (int): batch size.
        timestep (int): chunk length; SIGNAL_LENGTH (the default) keeps clips whole.
        seed (int): seed for the balanced sampling.

    Returns:
        train_ds, val_ds, train_steps, val_steps
    """
    x_train, y_train = load_split(data_dir, 'train')
    x_val, y_val = load_split(data_dir, 'validation')
    assert SIGNAL_LENGTH % timestep == 0, f'timestep must divide {SIGNAL_LENGTH}'
    segments = SIGNAL_LENGTH // timestep

    rng = np.random.default_rng(seed)
    n_epoch = NUM_CLASSES * TARGET_PER_CLASS

    def balanced_indices():
        while True:
            np.random.seed(rng.integers(2**31))
            yield build_balanced_indices(y_train, target_per_class=TARGET_PER_CLASS)

    x_train, y_train = tf.convert_to_tensor(x_train), tf.convert_to_tensor(y_train)
    train_ds = tf.data.Dataset.from_generator(
        balanced_indices, output_signature=tf.TensorSpec(shape=(n_epoch,), dtype=tf.int64))
    train_ds = train_ds.flat_map(
        lambda idx: tf.data.Dataset.from_tensor_slices(idx).shuffle(n_epoch, seed=seed))
    train_ds = train_ds.map(lambda i: (x_train[i], y_train[i]),
                            num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.map(_to_chunks(timestep)).unbatch().prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
    val_ds = val_ds.map(_to_chunks(timestep)).unbatch().prefetch(tf.data.AUTOTUNE)

    train_steps = segments * (n_epoch // batch_size)
    val_steps = segments * int(np.ceil(len(y_val) / batch_size))
    return train_ds, val_ds, train_steps, val_steps


def get_test_data(data_dir):
    """Test split as arrays: clips (N, SIGNAL_LENGTH, 1) and labels (N,)."""
    return load_split(data_dir, 'test')


def get_samples(path='./data/sc12_batch100_1024samples.npz'):
    """Calibration samples for quantization: int16 chunks from the training set.

    The file is BrainChip's published sample pack (SAMPLES_URL), downloaded by
    speech_commands_train.sh.
    """
    return np.load(path)['data']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Speech Commands data sanity check')
    parser.add_argument('-d', '--data', default='./data/speech_commands',
                        help='TFDS data_dir for the speech_commands dataset')
    args = parser.parse_args()

    for split in ('validation', 'test'):
        x, y = load_split(args.data, split)
        print(f'{split}: {x.shape} clips, value range [{x.min():.0f}, {x.max():.0f}], '
              f'class counts {np.bincount(y, minlength=NUM_CLASSES).tolist()}')
