#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
MIT-BIH arrhythmia data pipeline.

Turns raw MIT-BIH WFDB recordings into 36x32 uint8 "images" that an Akida
convolutional model can consume directly:

    raw ECG (360 Hz) -> beat window around the annotated R-peak
                     -> Morlet continuous wavelet transform (scalogram)
                     -> log1p + min-max normalise -> resize to 32x32
                     -> append 4 rows of RR-interval features -> (36, 32, 1)

The dataset is split inter-patient (patient-wise): the DS1 record list is used
for training, the disjoint DS2 record list for the reported test metrics. This
is deliberately harder, and more clinically honest, than a random beat-level
split: no beat from a test patient is ever seen during training.

get_naive_data() offers the random beat-level split as an opt-in comparison:
all the beats pooled and divided 60/20/20 without regard to which patient they
came from. It exists to make the cost of that shortcut measurable rather than
merely asserted - the same model trained the same way scores far higher on it,
because it has seen the test patients' other beats. It is never the default.

Everything is lazy and cached. The first call downloads the ~100 MB of raw
records if they are absent, runs the wavelet transform over all ~90k beats
(around a minute with all cores busy), and writes a single .npz cache next to
the record directory. Later calls just read the cache.

The cache holds the DS1 beats as one pool, not pre-split: the stratified
train/hold-out split is drawn in get_data() from the seed it is given, so
re-running the pipeline with a different seed varies the split without
re-running the (much more expensive, and entirely deterministic) wavelet
preprocessing. Every seed shares the one cache file.

Usage:
    from arrhythmia_data import get_data, get_test_data, get_samples

    train_ds, val_ds = get_data('./data/mitdb', (36, 32, 1), batch_size=64,
                                seed=42)
    test_ds = get_test_data('./data/mitdb', (36, 32, 1))
    samples = get_samples('./data/mitdb', (36, 32, 1), num_samples=1024)

    # The naive alternative:
    from arrhythmia_data import get_naive_data
    train_ds, val_ds, test_ds = get_naive_data('./data/mitdb', (36, 32, 1),
                                               batch_size=64, seed=42)

Rebuild the cache from scratch (after changing any preprocessing constant):
    python arrhythmia_data.py --rebuild
"""

import os
import time

import cv2
import numpy as np
import pywt
import tensorflow as tf
import wfdb
from joblib import Parallel, delayed
from scipy.signal import resample
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Dataset definition
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = './data/mitdb'
PHYSIONET_DB = 'mitdb'

# Inter-patient split (DS1 / DS2). Standard in the inter-patient arrhythmia
# literature; the four paced-rhythm records (102, 104, 107, 217) are excluded
# from both sides as the AAMI recommended practice directs.
TRAIN_RECORDS = [
    '101', '106', '108', '109', '112', '114', '115', '116',
    '118', '119', '122', '124', '201', '203', '205', '207',
    '208', '209', '215', '220', '223', '230',
]
TEST_RECORDS = [
    '100', '103', '105', '111', '113', '117', '121', '123',
    '200', '202', '210', '212', '213', '214', '219', '221',
    '222', '228', '231', '232', '233', '234',
]

# AAMI-style grouping of the MIT-BIH beat annotations. Symbols absent from this
# map (paced, fusion, unclassifiable beats and all non-beat annotations) are
# dropped rather than forced into a class.
LABEL_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',
    'V': 'V', 'E': 'V',
}
CLASS_TO_ID = {'N': 0, 'S': 1, 'V': 2}
TARGET_NAMES = ('N', 'S', 'V')
NUM_CLASSES = 3

# ---------------------------------------------------------------------------
# Preprocessing parameters
# ---------------------------------------------------------------------------
WINDOW_BEFORE = 250          # samples before the R-peak (~0.69 s at 360 Hz)
WINDOW_AFTER = 250           # samples after the R-peak
BEAT_TARGET_LENGTH = 360     # beats are resampled to this length
IMG_SIZE = 32                # scalogram is resized to IMG_SIZE x IMG_SIZE
CWT_SCALES = np.arange(1, 65)
CWT_WAVELET = 'morl'
NUM_RR_FEATURES = 4          # rr_prev, rr_next, rr_local, rr_avg

# Full model input: the scalogram with the RR feature rows stacked underneath.
INPUT_SHAPE = (IMG_SIZE + NUM_RR_FEATURES, IMG_SIZE, 1)

# RR features are standardised then squashed into [0, 1] so that they share the
# scalogram's range and survive uint8 encoding.
RR_CLIP_SIGMA = 1.0

# Fraction of the DS1 beats held out for monitoring during training.
VAL_SPLIT = 0.2

# Proportions of the naive (patient-blind) split, used only by get_naive_data.
# Training takes the 0.6 remainder.
NAIVE_VAL_SPLIT = 0.2
NAIVE_TEST_SPLIT = 0.2

# The 'ds1pool' marker denotes a cache holding the DS1 beats unsplit; the
# train/hold-out split is drawn per seed at load time.
CACHE_NAME = (f'{PHYSIONET_DB}_scalograms_{INPUT_SHAPE[0]}x{INPUT_SHAPE[1]}'
              f'_rr{RR_CLIP_SIGMA:g}sigma_ds1pool.npz')


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------
def _extract_beat(signal, peak):
    """Extract a fixed window around an R-peak, resampled to a fixed length."""
    start = peak - WINDOW_BEFORE
    end = peak + WINDOW_AFTER
    if start < 0 or end >= len(signal):
        return None

    beat = signal[start:end]
    if len(beat) != BEAT_TARGET_LENGTH:
        beat = resample(beat, BEAT_TARGET_LENGTH)
    return beat


def _compute_rr_features(rpeaks, i):
    """RR-interval features for beat i: previous, next, local ratio, average."""
    curr = rpeaks[i]
    rr_prev = 0 if i == 0 else curr - rpeaks[i - 1]
    rr_next = rr_prev if i == len(rpeaks) - 1 else rpeaks[i + 1] - curr

    start = max(0, i - 10)
    rr_history = np.diff(rpeaks[start:i + 1])
    rr_avg = np.mean(rr_history) if len(rr_history) > 0 else rr_prev
    rr_local = rr_prev / (rr_avg + 1e-8)

    return np.array([rr_prev, rr_next, rr_local, rr_avg])


def _beat_to_scalogram(beat):
    """Morlet CWT of a single beat, as a normalised IMG_SIZE x IMG_SIZE image."""
    beat = beat.astype(np.float32)
    coef, _ = pywt.cwt(beat, CWT_SCALES, CWT_WAVELET)

    img = np.log1p(np.abs(coef))
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


def _process_beat(ecg, peaks, peak, symbol, i):
    """Full per-beat pipeline. Returns None for beats we do not classify."""
    if symbol not in LABEL_MAP:
        return None

    beat = _extract_beat(ecg, peak)
    if beat is None:
        return None

    return (_beat_to_scalogram(beat),
            CLASS_TO_ID[LABEL_MAP[symbol]],
            _compute_rr_features(peaks, i))


def _load_records(data_path, records, label):
    """Read and process a list of records. Returns (scalograms, rr, labels)."""
    scalograms, rr_features, labels = [], [], []
    t_start = time.time()

    for n, record in enumerate(records, start=1):
        path = os.path.join(data_path, record)
        signal, _ = wfdb.rdsamp(path)
        ann = wfdb.rdann(path, 'atr')
        ecg = signal[:, 0]

        results = Parallel(n_jobs=-1, backend='loky')(
            delayed(_process_beat)(ecg, ann.sample, peak, symbol, i)
            for i, (peak, symbol) in enumerate(zip(ann.sample, ann.symbol))
        )
        valid = [r for r in results if r is not None]

        for scalogram, beat_label, rr in valid:
            scalograms.append(scalogram)
            labels.append(beat_label)
            rr_features.append(rr)

        print(f'  [{label} {n:2d}/{len(records)}] record {record}: '
              f'{len(valid)} beats kept of {len(results)} annotations')

    elapsed = time.time() - t_start
    total = len(scalograms)
    print(f'  {label}: {total} beats in {elapsed:.1f} s '
          f'({total / elapsed:.0f} beats/s, {elapsed / total * 1e3:.3f} ms/beat)')

    return np.array(scalograms), np.array(rr_features), np.array(labels)


def _normalise_rr(rr, rr_mean, rr_std):
    """Standardise RR features with train statistics, then squash into [0, 1]."""
    z = (rr - rr_mean) / (rr_std + 1e-8)
    z = np.clip(z, -RR_CLIP_SIGMA, RR_CLIP_SIGMA)
    return (z + RR_CLIP_SIGMA) / (2 * RR_CLIP_SIGMA)


def _add_rr_rows(scalograms, rr):
    """Stack the RR features underneath each scalogram as NUM_RR_FEATURES rows.

    Returns an array of shape (N,) + INPUT_SHAPE, i.e. with the trailing
    single-channel axis the model expects.
    """
    rr_rows = np.repeat(rr[:, :, np.newaxis], scalograms.shape[2], axis=2)
    merged = np.concatenate([scalograms, rr_rows], axis=1)
    return merged[..., np.newaxis]


def _to_uint8(x):
    """Encode a [0, 1] float array as uint8 spanning the full 0-255 range.

    The model's Rescaling layer inverts this exactly, so float training and
    Akida inference see the same values.
    """
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Download and cache
# ---------------------------------------------------------------------------
def _cache_path(data_path):
    """Cache location: alongside the record directory, not inside it."""
    return os.path.join(os.path.dirname(os.path.normpath(data_path)) or '.',
                        CACHE_NAME)


def _ensure_records(data_path):
    """Download the raw MIT-BIH records from PhysioNet if they are not present."""
    required = TRAIN_RECORDS + TEST_RECORDS
    missing = [r for r in required
               if not os.path.exists(os.path.join(data_path, f'{r}.dat'))]
    if not missing:
        return

    print(f'{len(missing)} of {len(required)} MIT-BIH records missing from '
          f'{data_path} - downloading from PhysioNet (~100 MB)...')
    os.makedirs(data_path, exist_ok=True)
    wfdb.dl_database(PHYSIONET_DB, data_path)
    print(f'Records downloaded to {data_path}')


def _build_cache(data_path, cache_path):
    """Run the full preprocessing pipeline and write the .npz cache."""
    print('Preprocessing MIT-BIH into scalograms. This runs once; later calls '
          'read the cache.')

    print(f'Train records (DS1, {len(TRAIN_RECORDS)} patients):')
    train_scalograms, train_rr, y_train = _load_records(
        data_path, TRAIN_RECORDS, 'DS1')

    # RR statistics come from the training split only, and are stored so that
    # val and test are normalised identically.
    rr_mean = train_rr.mean(axis=0)
    rr_std = train_rr.std(axis=0)

    print(f'Test records (DS2, {len(TEST_RECORDS)} patients):')
    test_scalograms, test_rr, y_test = _load_records(
        data_path, TEST_RECORDS, 'DS2')

    x_train = _to_uint8(_add_rr_rows(
        train_scalograms, _normalise_rr(train_rr, rr_mean, rr_std)))
    x_test = _to_uint8(_add_rr_rows(
        test_scalograms, _normalise_rr(test_rr, rr_mean, rr_std)))

    # The DS1 beats are stored as one pool. Splitting them into train and
    # hold-out is cheap and seed-dependent, so get_data() does it at load time
    # rather than baking one split into the cache.
    np.savez_compressed(
        cache_path,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
        rr_mean=rr_mean, rr_std=rr_std)

    print(f'Cache written to {cache_path}')


def _load_cache(data_path):
    """Return the cached arrays, building the cache (and downloading) if needed."""
    cache_path = _cache_path(data_path)
    if not os.path.exists(cache_path):
        _ensure_records(data_path)
        _build_cache(data_path, cache_path)
    return np.load(cache_path)


def _split_train_val(y, seed):
    """Stratified train/hold-out split of the DS1 beats.

    The hold-out is used only for monitoring during training; the reported
    metrics come from the DS2 test records. Stratifying keeps the S and V
    proportions of the pool in both halves, which matters because S is 2% of
    the beats.

    Args:
        y (np.ndarray): DS1 labels.
        seed (int): seed for the split. Varying it varies which beats are held
            out, one of the levels of run-to-run variability in this pipeline.

    Returns:
        np.ndarray, np.ndarray: training indices, hold-out indices.
    """
    return train_test_split(np.arange(len(y)), test_size=VAL_SPLIT, stratify=y,
                            random_state=seed)


def _split_naive(y, seed):
    """Stratified 60/20/20 split of the pooled beats, ignoring the patients.

    Drawn in two stages: 60% off the top, then the remaining 40% halved. Both
    stages are stratified and both take their randomness from `seed` alone, so
    the three partitions are a pure function of the seed and can be reproduced
    in a separate process - which is what lets a model trained here be evaluated
    on the matching test partition later.

    Args:
        y (np.ndarray): labels of the pooled DS1 + DS2 beats.
        seed (int): seed for both stages of the split.

    Returns:
        np.ndarray, np.ndarray, np.ndarray: training, validation and test
        indices, disjoint and covering the whole pool.
    """
    holdout = NAIVE_VAL_SPLIT + NAIVE_TEST_SPLIT
    train_idx, rest_idx = train_test_split(
        np.arange(len(y)), test_size=holdout, stratify=y, random_state=seed)
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=NAIVE_TEST_SPLIT / holdout, stratify=y[rest_idx],
        random_state=seed)
    return train_idx, val_idx, test_idx


def _describe(name, x, y):
    counts = np.bincount(y, minlength=NUM_CLASSES)
    breakdown = ', '.join(f'{n}: {c}' for n, c in zip(TARGET_NAMES, counts))
    print(f'{name}: {len(x)} beats ({breakdown})')


def _as_dataset(x, y, batch_size, shuffle=False, seed=42):
    """Wrap uint8 arrays as a batched dataset yielding float32 in [0, 255].

    The arrays are held as uint8 and cast per batch, which keeps memory down.
    Values stay integral, so the Akida evaluation path can cast straight back to
    uint8 without loss.
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda images, labels: (tf.cast(images, tf.float32), labels),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def _check_input_shape(input_shape):
    if tuple(input_shape) != INPUT_SHAPE:
        raise ValueError(
            f'This example is built around a fixed input shape of {INPUT_SHAPE} '
            f'(a {IMG_SIZE}x{IMG_SIZE} scalogram plus {NUM_RR_FEATURES} RR '
            f'feature rows), but got {tuple(input_shape)}.')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_data(data_path=DEFAULT_DATA_PATH, input_shape=INPUT_SHAPE, batch_size=64,
             seed=42):
    """Load the training data and the monitoring hold-out.

    Both splits come from the DS1 training patients: an 80/20 stratified split
    of their beats, drawn from `seed`. The hold-out is for tracking training
    only - because it shares patients with the training set it reports
    optimistically. Use get_test_data() for the numbers worth quoting.

    Args:
        data_path (str): path to the MIT-BIH record directory. Records are
            downloaded here if absent.
        input_shape (tuple): model input shape; must be INPUT_SHAPE.
        batch_size (int): the batch size.
        seed (int): seed for both the train/hold-out split and the shuffle
            order. The cached preprocessing is shared across seeds.

    Returns:
        tf.data.Dataset, tf.data.Dataset: training dataset, hold-out dataset.
        Both yield (uint8 images, int labels).
    """
    _check_input_shape(input_shape)
    cache = _load_cache(data_path)

    x_ds1, y_ds1 = cache['x_train'], cache['y_train']
    train_idx, val_idx = _split_train_val(y_ds1, seed)
    x_train, y_train = x_ds1[train_idx], y_ds1[train_idx]
    x_val, y_val = x_ds1[val_idx], y_ds1[val_idx]

    print(f'DS1 split with seed {seed}')
    _describe('Train (DS1 patients, 80%)', x_train, y_train)
    _describe('Hold-out (DS1 patients, 20%)', x_val, y_val)

    return (_as_dataset(x_train, y_train, batch_size, shuffle=True, seed=seed),
            _as_dataset(x_val, y_val, batch_size))


def get_test_data(data_path=DEFAULT_DATA_PATH, input_shape=INPUT_SHAPE,
                  batch_size=64):
    """Load the inter-patient test set (DS2 records).

    None of these patients appear in the training data, so this is the split
    the reported accuracy and per-class F1 scores come from.

    Args:
        data_path (str): path to the MIT-BIH record directory.
        input_shape (tuple): model input shape; must be INPUT_SHAPE.
        batch_size (int): the batch size.

    Returns:
        tf.data.Dataset: test dataset yielding (uint8 images, int labels).
    """
    _check_input_shape(input_shape)
    cache = _load_cache(data_path)

    x_test, y_test = cache['x_test'], cache['y_test']
    _describe('Test (DS2 patients)', x_test, y_test)

    return _as_dataset(x_test, y_test, batch_size)


def get_naive_data(data_path=DEFAULT_DATA_PATH, input_shape=INPUT_SHAPE,
                   batch_size=64, seed=42):
    """Load a naive, patient-blind 60/20/20 split of all the beats.

    Every beat from all 44 records is pooled and split at random, stratified by
    class. Beats from any one patient therefore land in all three partitions:
    the model is tested on new beats from people it has already learned, not on
    new people. This is the shortcut a great deal of published arrhythmia work
    takes, and it inflates the reported numbers substantially - most of all for
    the supraventricular class, whose morphology is highly patient-specific.

    It is provided for comparison against get_data() / get_test_data().

    The partitioning is a pure function of `seed`, so training and evaluation
    must be given the *same* seed. Evaluating with a different seed silently
    scores the model on beats it was trained on.

    One caveat on the preprocessing: the cached RR features were standardised
    with statistics fit on DS1 (see _build_cache), and the raw values are not
    retained, so under this split those statistics are a fixed preprocessing
    constant rather than one re-fit on the training partition. The effect is
    negligible next to the patient leakage this split is here to demonstrate.

    Args:
        data_path (str): path to the MIT-BIH record directory. Records are
            downloaded here if absent.
        input_shape (tuple): model input shape; must be INPUT_SHAPE.
        batch_size (int): the batch size.
        seed (int): seed for the three-way split and the shuffle order. The
            cached preprocessing is shared with the inter-patient path, so
            switching split modes never rebuilds the cache.

    Returns:
        tf.data.Dataset, tf.data.Dataset, tf.data.Dataset: training, validation
        and test datasets. All yield (uint8 images, int labels).
    """
    _check_input_shape(input_shape)
    cache = _load_cache(data_path)

    # The cache stores DS1 and DS2 separately; the whole point here is to ignore
    # that boundary.
    x_all = np.concatenate([cache['x_train'], cache['x_test']])
    y_all = np.concatenate([cache['y_train'], cache['y_test']])

    train_idx, val_idx, test_idx = _split_naive(y_all, seed)

    print(f'Naive patient-blind split with seed {seed} - for comparison only, '
          f'these beats share patients across all three partitions')
    _describe('Train (naive, 60%)', x_all[train_idx], y_all[train_idx])
    _describe('Validation (naive, 20%)', x_all[val_idx], y_all[val_idx])
    _describe('Test (naive, 20%)', x_all[test_idx], y_all[test_idx])

    return (_as_dataset(x_all[train_idx], y_all[train_idx], batch_size,
                        shuffle=True, seed=seed),
            _as_dataset(x_all[val_idx], y_all[val_idx], batch_size),
            _as_dataset(x_all[test_idx], y_all[test_idx], batch_size))


def get_samples(data_path=DEFAULT_DATA_PATH, input_shape=INPUT_SHAPE,
                num_samples=1024, seed=42):
    """Load a shuffled block of DS1 beats as a plain numpy array.

    Used for activation-sparsity measurement and hardware benchmarking, both of
    which need uint8 numpy input rather than a dataset. The train/hold-out
    split is irrelevant here - what matters is that the beats are real, since
    Akida timings depend on input activity - so the draw is over the whole DS1
    pool.

    Args:
        data_path (str): path to the MIT-BIH record directory.
        input_shape (tuple): model input shape; must be INPUT_SHAPE.
        num_samples (int): number of samples to return. Defaults to 1024.
        seed (int): seed for the sample draw. Akida timings and sparsity depend
            on the input activity, so varying this varies the measurement.

    Returns:
        np.ndarray: shape (num_samples,) + INPUT_SHAPE, dtype uint8.
    """
    _check_input_shape(input_shape)
    cache = _load_cache(data_path)

    x_ds1 = cache['x_train']
    # Take a class-representative spread rather than the first N beats, which
    # would come from only the first few patients.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x_ds1))[:num_samples]
    return x_ds1[idx].astype(np.uint8)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Build the MIT-BIH scalogram cache and report its contents')
    parser.add_argument('-d', '--data', default=DEFAULT_DATA_PATH,
                        help='MIT-BIH record directory')
    parser.add_argument('--seed', type=int, default=7,
                        help='Seed for the DS1 train/hold-out split, the '
                             'shuffle order and the benchmark sample draw')
    parser.add_argument('--rebuild', action='store_true',
                        help='Discard the .npz cache and preprocess the records '
                             'again. Only needed after changing a preprocessing '
                             'constant - the cache does not depend on the seed')
    parser.add_argument('--naive-split', action='store_true',
                        help='Report the naive patient-blind 60/20/20 split '
                             'instead of the inter-patient one. Both share the '
                             'same cache')
    args = parser.parse_args()

    if args.rebuild:
        cache_path = _cache_path(args.data)
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f'Removed {cache_path}')

    if args.naive_split:
        train_ds, val_ds, test_ds = get_naive_data(args.data, seed=args.seed)
    else:
        train_ds, val_ds = get_data(args.data, seed=args.seed)
        test_ds = get_test_data(args.data)
    samples = get_samples(args.data, seed=args.seed)

    print(f'\nBatches: train {len(train_ds)}, hold-out {len(val_ds)}, '
          f'test {len(test_ds)}')
    print(f'Samples for benchmarking: {samples.shape} {samples.dtype} '
          f'range [{samples.min()}, {samples.max()}]')
