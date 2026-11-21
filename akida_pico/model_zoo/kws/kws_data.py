# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Speech Commands (12-class) data helpers for the Akida Pico KWS example.

The dataset comes from TensorFlow Datasets (about 2 GB on the first run, cached
afterwards in ``~/tensorflow_datasets`` or in the directory passed as ``data_path``).
Clips are kept as **raw 16 kHz audio in the int16 range**: Pico has no FFT/MFCC front
end, so there is no feature extraction here, only padding to a fixed window.

Window length: 16,384 samples (2^14, about 1.02 s). Streaming chunk: 256 samples, so one
decision every 16 ms. The model downsamples time by 256x, so 256 samples is the smallest
chunk that completes an output step.
"""
import numpy as np
import tensorflow_datasets as tfds

# TFDS label order for the 12-class task: 10 keywords, then silence and unknown.
LABEL_NAMES = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes',
               'silence', 'unknown']
NUM_CLASSES = len(LABEL_NAMES)
SAMPLE_RATE = 16000
SAMPLE_LENGTH = 16384
CHUNK_SIZE = 256


def _pad(audio, length):
    """Zero-pad (at the end) or crop (keeping the last samples) to ``length``."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.shape[-1] < length:
        return np.pad(audio, (0, length - audio.shape[-1]))
    return audio[-length:]


def load_split(split='test', length=SAMPLE_LENGTH, data_path=None):
    """Load one Speech Commands split as ``(X, y)``.

    Args:
        split (str): ``'train'``, ``'validation'`` or ``'test'``.
        length (int): window length in samples.
        data_path (str, optional): TFDS data directory. Defaults to ``~/tensorflow_datasets``.

    Returns:
        (np.ndarray, np.ndarray): ``X`` of shape ``(N, length, 1)`` with int16-range values
        and integer labels ``y`` of shape ``(N,)``.
    """
    ds = tfds.load('speech_commands', split=split, data_dir=data_path, shuffle_files=False)
    clips, labels = [], []
    for sample in ds:
        clips.append(_pad(sample['audio'].numpy(), length))
        labels.append(int(sample['label'].numpy()))
    return np.stack(clips)[..., np.newaxis], np.array(labels)


def get_calibration_samples(path):
    """Load the quantization calibration set: ``(100, 1024, 1)`` int16 segments.

    Each segment is 1,024 samples cut from the middle of a training clip. Recreate it from
    your own training data with ``tenn_sc12_train extract`` (see the README).
    """
    return np.load(path)['data']


def get_train_data(batch_size=128, data_path=None, length=SAMPLE_LENGTH):
    """Datasets for float training, from the ``akida_models`` SC12 recipe.

    Each epoch draws 3,100 clips per class: ``unknown`` is downsampled and ``silence`` is
    upsampled, so the classes are balanced.

    Returns:
        (train_ds, val_ds, train_steps, val_steps, segments_per_sample)
    """
    from akida_models.tenn_recurrent.sc12_train import get_data
    return get_data(length, data_path, batch_size, timestep=length, target_per_class=3100)
