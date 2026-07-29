#!/usr/bin/env python
"""Simplified data loader for keyword spotting (DS-CNN + MFCC).

Loads the speech_commands dataset via tensorflow_datasets and applies
MFCC feature extraction with parameters matching the MLPerf Tiny benchmark.

Output shape per sample: [SPECTROGRAM_LENGTH, NUM_MFCC, 1] = [49, 10, 1]
"""
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

# ---------------------------------------------------------------------------
# MFCC configuration — MLPerf Tiny defaults
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
CLIP_MS = 1000

WINDOW_SIZE_MS = 30.0
WINDOW_STRIDE_MS = 20.0

NUM_MEL_BINS = 40
NUM_MFCC = 10
MEL_LOWER_HZ = 20.0
MEL_UPPER_HZ = 4000.0

DESIRED_SAMPLES = int(SAMPLE_RATE * CLIP_MS / 1000)              # 16000
WINDOW_SIZE_SAMPLES = int(SAMPLE_RATE * WINDOW_SIZE_MS / 1000)   # 480
WINDOW_STRIDE_SAMPLES = int(SAMPLE_RATE * WINDOW_STRIDE_MS / 1000)  # 320
SPECTROGRAM_LENGTH = 1 + (DESIRED_SAMPLES - WINDOW_SIZE_SAMPLES) // WINDOW_STRIDE_SAMPLES  # 49

NUM_LABELS = 12
LABEL_NAMES = ["Down", "Go", "Left", "No", "Off", "On", "Right",
               "Stop", "Up", "Yes", "Silence", "Unknown"]

# For rescaling to uint8 range
# Max value in MFCC Spectrograms, Train split
MFCC_MAX = 31.0

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess(sample_dict):
    """Pad, normalise and compute MFCC for one audio sample.

    Args:
        sample_dict: dict with keys 'audio' (int16 tensor) and 'label' (int).

    Returns:
        (mfccs, label) where mfccs has shape [SPECTROGRAM_LENGTH, NUM_MFCC, 1].
    """
    audio = tf.cast(sample_dict['audio'], tf.float32)
    label = sample_dict['label']

    # Pad to exactly DESIRED_SAMPLES with trailing zeros
    audio = tf.pad(audio, [[0, DESIRED_SAMPLES - tf.shape(audio)[0]]])

    # Normalise to [-1, 1] by peak absolute amplitude
    audio = audio / (tf.reduce_max(tf.abs(audio)) + 1e-9)

    # STFT (fft_length=None → next power-of-2 >= frame_length, i.e. 512)
    stfts = tf.signal.stft(
        audio,
        frame_length=WINDOW_SIZE_SAMPLES,
        frame_step=WINDOW_STRIDE_SAMPLES,
        fft_length=None,
        window_fn=tf.signal.hann_window,
    )
    spectrogram = tf.abs(stfts)  # [SPECTROGRAM_LENGTH, fft_bins]
    num_fft_bins = stfts.shape[-1]

    # Mel filterbank
    linear_to_mel = tf.signal.linear_to_mel_weight_matrix(
        NUM_MEL_BINS, num_fft_bins, SAMPLE_RATE, MEL_LOWER_HZ, MEL_UPPER_HZ
    )
    mel = tf.tensordot(spectrogram, linear_to_mel, 1)
    mel.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel.shape[-1:]))

    # Log-mel → MFCC (take first NUM_MFCC coefficients)
    log_mel = tf.math.log(mel + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :NUM_MFCC]

    # Add channel dimension for Conv2D: [SPECTROGRAM_LENGTH, NUM_MFCC, 1]
    mfccs = tf.reshape(mfccs, [SPECTROGRAM_LENGTH, NUM_MFCC, 1])

    return mfccs, label


# Copied from tfio v0.37.1 (tensorflow_io/python/ops/audio_ops.py). Modified to accept
# a stateless seed so augmentation is deterministic across parallel AUTOTUNE map workers.
def _freq_mask(input, param, seed):
    """Apply a frequency mask. seed: shape-[2] int64 tensor."""
    input = tf.convert_to_tensor(input)
    freq_max = tf.shape(input)[1]
    sub_seeds = tf.random.experimental.stateless_split(seed, num=2)  # [2, 2]
    f = tf.random.stateless_uniform(
        shape=(), seed=sub_seeds[0], minval=0, maxval=param, dtype=tf.dtypes.int32
    )
    f0 = tf.random.stateless_uniform(
        shape=(), seed=sub_seeds[1], minval=0, maxval=freq_max - f, dtype=tf.dtypes.int32
    )
    indices = tf.reshape(tf.range(freq_max), (1, -1))
    condition = tf.math.logical_and(
        tf.math.greater_equal(indices, f0), tf.math.less(indices, f0 + f)
    )
    return tf.where(condition, tf.cast(0, input.dtype), input)


# Copied from tfio v0.37.1 (tensorflow_io/python/ops/audio_ops.py). Modified to accept
# a stateless seed so augmentation is deterministic across parallel AUTOTUNE map workers.
def _time_mask(input, param, seed):
    """Apply a time mask. seed: shape-[2] int64 tensor."""
    input = tf.convert_to_tensor(input)
    time_max = tf.shape(input)[0]
    sub_seeds = tf.random.experimental.stateless_split(seed, num=2)  # [2, 2]
    t = tf.random.stateless_uniform(
        shape=(), seed=sub_seeds[0], minval=0, maxval=param, dtype=tf.dtypes.int32
    )
    t0 = tf.random.stateless_uniform(
        shape=(), seed=sub_seeds[1], minval=0, maxval=time_max - t, dtype=tf.dtypes.int32
    )
    indices = tf.reshape(tf.range(time_max), (-1, 1))
    condition = tf.math.logical_and(
        tf.math.greater_equal(indices, t0), tf.math.less(indices, t0 + t)
    )
    return tf.where(condition, tf.cast(0, input.dtype), input)


def _preprocess_with_idx(idx, sample_dict):
    """Variant of _preprocess that threads the enumerate index through the pipeline."""
    mfcc, label = _preprocess(sample_dict)
    return idx, mfcc, label


def _make_waveform_augmenter(aug_seed, max_shift_samples):
    """Return a map fn that deterministically time-shifts the raw audio waveform.

    Accepts and returns (idx, sample_dict) so the enumerate index is threaded
    through the pipeline. seed = [aug_seed, idx] is unique per (trial, sample
    position in epoch), giving deterministic but varied augmentation.
    """
    aug_seed_val = aug_seed if aug_seed is not None else 0

    def augment(idx, sample_dict):
        seed = tf.stack([tf.cast(aug_seed_val, tf.int64), tf.cast(idx, tf.int64)])
        shift = tf.random.stateless_uniform(
            [], seed=seed, minval=-max_shift_samples,
            maxval=max_shift_samples + 1, dtype=tf.int32
        )
        audio = tf.roll(sample_dict["audio"], shift, axis=0)
        return idx, {**sample_dict, "audio": audio}

    return augment


def _make_specaugment_fn(aug_seed, freq_param, time_param):
    """Return a map fn that applies SpecAugment to a float32 MFCC tensor.

    Accepts and returns (idx, mfcc, label). Seed scheme:
    base = [aug_seed+1, idx]; split into freq_seed and time_seed so freq and
    time masks are independent. Each mask splits its seed further for its two
    internal draws. Input mfcc shape: [SPECTROGRAM_LENGTH, NUM_MFCC, 1].
    """
    aug_seed_val = aug_seed if aug_seed is not None else 0

    def augment(idx, mfcc, label):
        base = tf.stack([tf.cast(aug_seed_val + 1, tf.int64), tf.cast(idx, tf.int64)])
        mask_seeds = tf.random.experimental.stateless_split(base, num=2)  # [2, 2]
        mfcc_2d = tf.squeeze(mfcc, axis=-1)             # [49, 10]
        mfcc_2d = _freq_mask(mfcc_2d, freq_param, mask_seeds[0])
        mfcc_2d = _time_mask(mfcc_2d, time_param, mask_seeds[1])
        return idx, tf.expand_dims(mfcc_2d, axis=-1), label  # [49, 10, 1]

    return augment


def compute_mfcc_range(data_dir=None, percentile_low=0.5, percentile_high=99.5):
    """Compute data range from the training split for uint8 scaling.

    Iterates the full train split after MFCC preprocessing and returns
    percentile-based bounds for use as data_transform in get_datasets().

    Args:
        data_dir: Directory to read tfds data from. Defaults to ~/tensorflow_datasets.
        percentile_low:  Lower percentile (default 0.5).
        percentile_high: Upper percentile (default 99.5).

    Returns:
        (data_min, data_max): floats suitable for passing as data_transform.
    """
    ds, _ = tfds.load(
        'speech_commands', split='train', data_dir=data_dir, with_info=True
    )
    ds = (ds
          .map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(256)
          .prefetch(tf.data.AUTOTUNE))

    all_features = []
    for features, _ in ds:
        all_features.append(features.numpy())

    flat = np.concatenate(all_features, axis=0).flatten()
    data_min = float(np.percentile(flat, percentile_low))
    data_max = float(np.percentile(flat, percentile_high))
    print(f"MFCC range [{percentile_low}–{percentile_high} pct]: {data_min:.4f} – {data_max:.4f}")
    return data_min, data_max


def _make_uint8_scaler(data_min, data_max):
    """Return a tf.data map function that clips and scales features to uint8."""
    MAX_UINT8 = 255.0

    def scale(features, label):
        features = tf.clip_by_value(features, data_min, data_max)
        features = (features - data_min) * MAX_UINT8 / (data_max - data_min)
        features = tf.cast(tf.round(features), tf.uint8)
        return features, label

    return scale


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def get_datasets(
    data_dir=None,
    batch_size=100,
    num_train_samples=-1,
    num_val_samples=-1,
    num_test_samples=-1,
    data_transform=None,
    aug_enabled=False,
    aug_time_shift_max_ms=100,
    aug_freq_mask_param=2,
    aug_time_mask_param=10,
    shuffle_seed=None,
    aug_seed=None,
):
    """Load speech_commands and return batched (train, test, val) tf.data.Datasets.

    Each element is a tuple (mfcc_features, label) where mfcc_features has
    shape [batch, SPECTROGRAM_LENGTH, NUM_MFCC, 1] = [batch, 49, 10, 1].

    Args:
        data_dir: Directory to download / read tfds data.
                  Defaults to ~/tensorflow_datasets.
        batch_size: Samples per batch.
        num_train_samples: Cap on training samples (-1 = all).
        num_val_samples:   Cap on validation samples (-1 = all).
        num_test_samples:  Cap on test samples (-1 = all).
        data_transform: Optional (data_min, data_max) tuple returned by
                        compute_mfcc_range(). When provided, features are
                        clipped, scaled to [0, 255] and cast to uint8.
                        When None (default), features remain float32.

    Returns:
        (ds_train, ds_test, ds_val) — batched and prefetched tf.data.Datasets.
    """
    (ds_train, ds_test, ds_val), _ = tfds.load(
        'speech_commands',
        split=['train', 'test', 'validation'],
        data_dir=data_dir,
        with_info=True,
    )

    if num_train_samples != -1:
        ds_train = ds_train.take(num_train_samples)
    if num_val_samples != -1:
        ds_val = ds_val.take(num_val_samples)
    if num_test_samples != -1:
        ds_test = ds_test.take(num_test_samples)

    autotune = tf.data.AUTOTUNE

    if data_transform is not None:
        data_min, data_max = data_transform
        scaler = _make_uint8_scaler(data_min, data_max)

    max_shift_samples = int(SAMPLE_RATE * aug_time_shift_max_ms / 1000)
    waveform_aug = _make_waveform_augmenter(aug_seed, max_shift_samples)
    specaugment = _make_specaugment_fn(aug_seed, aug_freq_mask_param, aug_time_mask_param)

    def _build(ds, shuffle=False, aug=False):
        # Option B: shuffle before augmentation so the post-shuffle position (idx
        # from enumerate) seeds each sample's augmentation. This gives different
        # augmentation per epoch per sample while remaining deterministic across runs.
        if shuffle:
            ds = ds.shuffle(4096, seed=shuffle_seed)
        if aug:
            ds = ds.enumerate()                                                          # → (idx, sample_dict)
            ds = ds.map(waveform_aug, num_parallel_calls=autotune)                      # → (idx, sample_dict)
            ds = ds.map(_preprocess_with_idx, num_parallel_calls=autotune)              # → (idx, mfcc, label)
            ds = ds.map(specaugment, num_parallel_calls=autotune)                       # → (idx, mfcc, label)
            ds = ds.map(lambda _, mfcc, label: (mfcc, label),
                        num_parallel_calls=autotune)                                    # drop idx
        else:
            ds = ds.map(_preprocess, num_parallel_calls=autotune)
        if data_transform is not None:
            ds = ds.map(scaler, num_parallel_calls=autotune)
        return ds.batch(batch_size).prefetch(autotune)

    ds_train = _build(ds_train, shuffle=True, aug=aug_enabled)
    ds_test  = _build(ds_test)
    ds_val   = _build(ds_val)

    return ds_train, ds_test, ds_val

def get_samples(data_path, data_transform=None, num_samples=1024):
    """ Loads samples from the validation split as a numpy array.

    No augmentation is applied.
    Suitable for model calibration and testing.

    Args:
        data_path (str): directory used as the tfds data_dir
        data_transform: Optional (data_min, data_max) tuple returned by
                        compute_mfcc_range(). When provided, features are
                        clipped, scaled to [0, 255] and cast to uint8.
                        When None (default), features remain float32.
        num_samples (int): number of samples to return. Defaults to 1024.

    Returns:
        np.ndarray: array of shape (num_samples, height, width, channels), dtype uint8
    """

    tfds.disable_progress_bar()
    ds = tfds.load(
        'speech_commands',
        split='validation',
        data_dir=data_path,
        with_info=False,
    )
    autotune = tf.data.AUTOTUNE

    if data_transform is not None:
        data_min, data_max = data_transform
        scaler = _make_uint8_scaler(data_min, data_max)
    
    ds = ds.map(_preprocess, num_parallel_calls=autotune)
    if data_transform is not None:
        ds = ds.map(scaler, num_parallel_calls=autotune)
    
    samples = [s for s, _ in ds.take(num_samples).as_numpy_iterator()]

    return np.array(samples)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    DATA_DIR = '/mnt/data/sc10/'
    data_transform = compute_mfcc_range(data_dir=DATA_DIR)
    ds_train, ds_test, ds_val = get_datasets(
        data_dir=DATA_DIR, batch_size=100, data_transform=data_transform
    )
    for features, labels in ds_train.take(1):
        print(f"Feature batch shape: {features.shape}, dtype: {features.dtype}")
        print(f"Label batch shape:   {labels.shape}")