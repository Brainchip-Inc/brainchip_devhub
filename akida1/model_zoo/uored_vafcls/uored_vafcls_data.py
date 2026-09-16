#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
UORED-VAFCLS bearing fault data pipeline.

Turns 10 s raw accelerometer recordings into 1 s uint8 "images" that an Akida
convolutional model can consume directly:

    raw vibration (42 kHz) -> 1 s window (42000 samples)
                           -> random gain (training only)
                           -> rescale to uint8-range using a
                              single per-dataset scale factor
                           -> reshape to (1200, 35, 1)

DATA SPLIT PROTOCOL

The dataset split implemented here is derived from the method proposed by 
Vieira et al (2026), "Towards a more realistic evaluation of machine learning
models for bearing fault diagnosis" (https://arxiv.org/abs/2509.22267)
The split is bearing-disjoint and drawn *before* any windowing to avoid the
data-leakage that compromises the majority of naive approaches to this dataset.

Fault mode is determined by bearing id in this dataset (inner=1-5, outer=6-10, 
ball=11-15, cage=16-20), so a model given any split that is not bearing-disjoint 
can score near-perfectly by re-identifying the bearing without learning anything 
about faults. Each bearing contributes three recordings - its own healthy one 
plus two severities of a single fault mode - and holding a bearing out removes 
all three.

Folds are enumerated, not tabulated: every way of holding out 2 bearings from
each of the 4 fault modes gives C(5,2)^4 = 10,000 folds, shuffled once with a
fixed seed and indexed by fold number. Folds 0-4 are the tuning budget; folds
5-104 are the 100 evaluation folds. FIXED_FOLD (5) is the one the pretrained
models and the hardware benchmark use.

Labels are multi-label, not one-of-N: [inner, outer, ball, cage], with healthy
encoded as the all-zero vector rather than a fifth class. The reported metric is
macro AUROC (see uored_vafcls_eval.py) - there is no argmax prediction here, so
an accuracy figure would depend on an arbitrary threshold.
                           
DATA PREPARATION

The raw input vectors are reshaped to a 2D input, (1200, 35, 1). Note that's a 
plain reshape, not a spectrogram: row j is samples [35j, 35j+35), so the row axis
is coarse time (1200 frames, 0.833 ms apart) and the column axis is fine phase 
within a frame. It happens here rather than in the model because an Akida graph 
cannot contain a reshape - the deployed model receives the already-framed tensor.


Usage:
    from uored_vafcls_data import get_data, get_test_data, get_samples

    train_ds, test_ds = get_data('./data/uored_vafcls', (1200, 35, 1),
                                 batch_size=120, fold=5, seed=0)
    test_ds = get_test_data('./data/uored_vafcls', (1200, 35, 1), fold=5)
    samples = get_samples('./data/uored_vafcls', (1200, 35, 1), num_samples=100)

Rebuild the cache from the raw Mendeley CSVs:
    python uored_vafcls_data.py --prepare-raw /path/to/1_CSV_Raw_Data_Files
"""

import functools
import os
import random
from itertools import combinations, product

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Dataset definition
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = './data/uored_vafcls'
CACHE_NAME = 'uored_vafcls_42khz_60rec.npz'

# The prepared cache is mirrored so the example runs without a manual download.
# The raw dataset is distributed as per-recording CSVs from Mendeley, which is
# not fetchable as a single file; --prepare-raw rebuilds this .npz from them.
MIRROR_URL = ('https://data.brainchip.com/dataset-mirror/uored_vafcls/'
              + CACHE_NAME)
MIRROR_HASH = None  # TODO: sha256 once the mirror is uploaded

SAMPLE_RATE = 42_000         # Hz
SIGNAL_LENGTH = 420_000      # 10 s per recording
N_RECORDINGS = 60            # 20 bearings x (1 healthy + 2 fault severities)

# Fault mode is a property of the bearing in this dataset. This mapping is what
# makes a non-bearing-disjoint split leak, and it is why the split enumerates
# bearings rather than recordings.
FAULT_BEARING_IDS = {
    'inner': (1, 2, 3, 4, 5),
    'outer': (6, 7, 8, 9, 10),
    'ball': (11, 12, 13, 14, 15),
    'cage': (16, 17, 18, 19, 20),
}
LABEL_COLUMNS = ('inner', 'outer', 'ball', 'cage')
NUM_LABELS = 4

# ---------------------------------------------------------------------------
# Split protocol
# ---------------------------------------------------------------------------
N_BEARINGS_PER_FAULT_MODE = 2   # 2 of 5 held out per mode -> 8 of 20 bearings
SPLIT_SEED = 42                 # fixes the fold numbering; see the note below
TUNING_FOLDS = range(0, 5)      # the only folds a hyperparameter may be chosen on
EVAL_FOLDS = range(5, 105)      # the 100 folds the reported mean comes from
FIXED_FOLD = 5                  # pretrained models + hardware benchmark

# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------
WINDOW_LENGTH = 42_000          # 1.0 s
FRAME_WIDTH = 35                # samples per frame; 42000 / 35 = 1200 frames
INPUT_SHAPE = (WINDOW_LENGTH // FRAME_WIDTH, FRAME_WIDTH, 1)   # (1200, 35, 1)

TRAIN_WINDOWS_PER_RECORDING = 20   # 36 recordings x 20 = 720 windows per epoch
EVAL_WINDOWS_PER_RECORDING = 10    # 24 recordings x 10 = 240 windows, tiled
GAIN_STD = 0.7                     # random scalar gain, N(1, GAIN_STD)
BATCH_SIZE = 120                   # exact divisor of 720 -> 6 steps per epoch

# ---------------------------------------------------------------------------
# uint8 input encoding
# ---------------------------------------------------------------------------
# Akida takes 8-bit inputs (uint8 range). Inspection of the UORED-VAFCLS data
# shows that amplitude varies widely across recordings (with, predictably, 
# vibration increasing markedly when bearings are faulty). Rather than set the
# range according to the global min-max of the data (which would completely 
# flatten the low amplitude vibrations of some recordings when reduced to 8-bit)
# we choose a range that preserves the majority of data points, while clipping
# the worst outliers to preserve resolution for low-amplitude signals. We
# chose a range of [-128, 127], which looks suitable for the data, and has the
# advantage of requiring minimal transformation of the input signal: no rescaling,
# a simple offset of +128 plus rounding and clipping to [0, 255].
#
ZERO_POINT = 128

# ---------------------------------------------------------------------------
# Split protocol
# ---------------------------------------------------------------------------
def held_out_bearings_for_fold(fold, n_per_mode=N_BEARINGS_PER_FAULT_MODE,
                               seed=SPLIT_SEED):
    """Return the sorted bearing ids held out for `fold`.

    Enumerates every way of choosing `n_per_mode` bearings from each of the four
    fault modes (2 -> C(5,2)^4 = 10,000 folds), shuffles that list once with a
    fixed seed, and indexes it by `fold`.

    The enumeration order and the shuffle are reproduced exactly from the
    protocol's reference implementation, so fold N here holds out the same
    bearings as fold N there. Do not "tidy" this function: any change to the
    ordering silently renumbers all 10,000 folds and every published per-fold
    number stops meaning what it says.

    Args:
        fold (int): fold index, 0 to 9999.
        n_per_mode (int): bearings held out per fault mode.
        seed (int): seed for the one-off shuffle of the enumeration.

    Returns:
        list: the held-out bearing ids, sorted ascending.
    """
    per_mode = [list(combinations(ids, n_per_mode))
                for ids in FAULT_BEARING_IDS.values()]
    all_combos = list(product(*per_mode))
    random.Random(seed).shuffle(all_combos)

    if not 0 <= fold < len(all_combos):
        raise IndexError(f'fold {fold} out of range for {len(all_combos)} folds')

    selected = all_combos[fold]
    return sorted({bearing for mode in selected for bearing in mode})


def fold_rows(bearing_id, fold, n_per_mode=N_BEARINGS_PER_FAULT_MODE,
              seed=SPLIT_SEED):
    """Split recording indices into train and test rows for `fold`.

    Args:
        bearing_id (np.ndarray): per-recording bearing id, shape (N_RECORDINGS,).
        fold (int): fold index.
        n_per_mode (int): bearings held out per fault mode.
        seed (int): seed for the fold enumeration shuffle.

    Returns:
        np.ndarray, np.ndarray: training row indices, test row indices.
    """
    test_bearings = held_out_bearings_for_fold(fold, n_per_mode, seed)
    is_test = np.isin(bearing_id, test_bearings)
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


# ---------------------------------------------------------------------------
# Input encoding
# ---------------------------------------------------------------------------
def encode_uint8(window):
    """Encode a float window as uint8 centred on ZERO_POINT.
    Args:
        window (np.ndarray): float32 samples, any shape.

    Returns:
        np.ndarray: uint8, same shape as `window`.
    """
    return np.clip(np.round(ZERO_POINT + window), 0, 255).astype(np.uint8)


def frame_window(x, width=FRAME_WIDTH):
    """Frame the trailing sample axis into (frames, width, 1).

    Leaves any leading batch axes alone, so it works on one window or a stack.

    Args:
        x (np.ndarray): shape (..., WINDOW_LENGTH).
        width (int): samples per frame.

    Returns:
        np.ndarray: shape (..., WINDOW_LENGTH // width, width, 1).
    """
    if x.shape[-1] % width:
        raise ValueError(f'window length {x.shape[-1]} is not divisible by the '
                         f'frame width {width}')
    return np.ascontiguousarray(x.reshape(*x.shape[:-1], -1, width))[..., None]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def _cache_path(data_path):
    return os.path.join(data_path, CACHE_NAME)


@functools.lru_cache(maxsize=1)
def _load_cache(data_path):
    """Load the prepared recordings and derive the encoding constants.

    lru_cached because uored_vafcls_cross_validate.py calls get_data() a hundred
    times in one process; without this it would re-read ~100 MB

    Args:
        data_path (str): directory holding (or to receive) the .npz cache.

    Returns:
        dict: signals, labels, bearing_id, fault_type, severity, rpm,
        waveform_id
    """
    path = _cache_path(data_path)
    if not os.path.exists(path):
        _download_cache(data_path)

    with np.load(path, allow_pickle=False) as data:
        cache = {key: data[key] for key in data.files}

    signals = cache['signals']
    if signals.shape != (N_RECORDINGS, SIGNAL_LENGTH):
        raise ValueError(f'{path}: expected signals of shape '
                         f'{(N_RECORDINGS, SIGNAL_LENGTH)}, got {signals.shape}')

    return cache


def _download_cache(data_path):
    """Fetch the prepared .npz from the BrainChip dataset mirror."""
    import pooch

    if MIRROR_HASH is None:
        print('Note: MIRROR_HASH is not set yet, so the download cannot be '
              'checksum-verified.')
    os.makedirs(data_path, exist_ok=True)
    print(f'Downloading the prepared UORED-VAFCLS cache to {data_path} ...')
    pooch.retrieve(url=MIRROR_URL, known_hash=MIRROR_HASH, path=data_path,
                   fname=CACHE_NAME)


def _check_input_shape(input_shape):
    if tuple(input_shape) != INPUT_SHAPE:
        raise ValueError(
            f'This example is built around a fixed input shape of {INPUT_SHAPE} '
            f'(a {WINDOW_LENGTH}-sample window framed {FRAME_WIDTH} samples '
            f'wide), but got {tuple(input_shape)}.')


def _labels(cache, rows):
    return cache['labels'][rows].astype(np.float32)


def _describe(name, cache, rows):
    """Print what a set of recordings contains, by bearing and fault mode."""
    bearings = sorted(set(cache['bearing_id'][rows].tolist()))
    counts = {}
    for fault_type in cache['fault_type'][rows]:
        counts[str(fault_type)] = counts.get(str(fault_type), 0) + 1
    breakdown = ', '.join(f'{k} {v}' for k, v in sorted(counts.items()))
    print(f'{name}: {len(rows)} recordings from {len(bearings)} bearings '
          f'{bearings} ({breakdown})')


# ---------------------------------------------------------------------------
# Window generation
# ---------------------------------------------------------------------------
def _train_dataset(cache, rows, batch_size, dtype, seed):
    """Random 1 s crops with a random gain, re-drawn every epoch."""
    signals = cache['signals']
    labels = _labels(cache, rows)
    n_items = len(rows) * TRAIN_WINDOWS_PER_RECORDING

    # An independent generator, not the global Keras RNG. The shell pipeline
    # re-seeds in a fresh process between building and training, so only an
    # independent data RNG makes the in-process cross-validation path and the
    # multi-process shell path draw the same crops for a given (fold, seed).
    rng = np.random.default_rng(seed)

    def generator():
        for idx in range(n_items):
            i = idx % len(rows)
            row = rows[i]
            start = rng.integers(0, SIGNAL_LENGTH - WINDOW_LENGTH)
            window = np.array(signals[row, start:start + WINDOW_LENGTH],
                              dtype=np.float32)
            # Gain is applied to the float waveform
            window *= np.float32(rng.normal(1.0, GAIN_STD))
            # window = encode_uint8(window, center[row], scale[row])
            window = encode_uint8(window)
            yield frame_window(window), labels[i]

    signature = (tf.TensorSpec(shape=INPUT_SHAPE, dtype=tf.uint8),
                 tf.TensorSpec(shape=(NUM_LABELS,), dtype=tf.float32))
    return (
        tf.data.Dataset.from_generator(generator, output_signature=signature)
        # Declared so that len(train_ds) works, which is how the training script
        # derives steps_per_epoch for the LR schedule.
        .apply(tf.data.experimental.assert_cardinality(n_items))
        .shuffle(n_items, seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=True)
        .map(lambda x, y: (tf.cast(x, dtype), y))
        .prefetch(tf.data.AUTOTUNE)
    )


def _eval_arrays(cache, rows):
    """Deterministic contiguous tiling: EVAL_WINDOWS_PER_RECORDING per recording.

    Small enough to materialise (240 x 1200 x 35 uint8 is about 10 MB).
    """
    signals = cache['signals']
    labels = _labels(cache, rows)
    windows, targets = [], []
    for i, row in enumerate(rows):
        for j in range(EVAL_WINDOWS_PER_RECORDING):
            start = j * WINDOW_LENGTH
            window = np.array(signals[row, start:start + WINDOW_LENGTH],
                              dtype=np.float32)
            windows.append(encode_uint8(window))
            targets.append(labels[i])
    return frame_window(np.stack(windows)), np.stack(targets)


def _eval_dataset(cache, rows, batch_size, dtype):
    x, y = _eval_arrays(cache, rows)
    return (
        tf.data.Dataset.from_tensor_slices((x, y))
        .batch(batch_size)
        .map(lambda a, b: (tf.cast(a, dtype), b))
        .prefetch(tf.data.AUTOTUNE)
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_data(data_path=DEFAULT_DATA_PATH, input_shape=INPUT_SHAPE,
             batch_size=BATCH_SIZE, dtype=tf.uint8, fold=FIXED_FOLD, seed=0):
    """Load the training and held-out datasets for one bearing-disjoint fold.

    There is deliberately no validation split. With 36 training recordings,
    carving one out would cost about a sixth of the data and would itself have to
    be bearing-disjoint to mean anything; nothing in the pipeline consumes it
    (the recipe is a fixed epoch count with no early stopping and no best-weight
    restore); and the protocol reserves folds 0-4 for exactly the tuning a
    validation set would otherwise serve. The second dataset returned is the
    held-out fold - the test set - and no decision may be taken from it.

    Args:
        data_path (str): directory holding the .npz cache. Downloaded if absent.
        input_shape (tuple): model input shape; must be INPUT_SHAPE.
        batch_size (int): the batch size. BATCH_SIZE divides the 720 training
            windows exactly, giving 6 steps per epoch.
        dtype (tf.dtypes.DType, optional): input data type. Defaults to tf.uint8.
        fold (int): bearing-disjoint fold index. Folds 0-4 are the tuning
            budget; 5-104 are the evaluation folds. Defaults to FIXED_FOLD.
        seed (int): seed for the crop positions, the gains and the shuffle
            order. The split itself does not depend on it.

    Returns:
        tf.data.Dataset, tf.data.Dataset: training dataset, held-out dataset.
    """
    _check_input_shape(input_shape)
    cache = _load_cache(data_path)

    train_rows, test_rows = fold_rows(cache['bearing_id'], fold)
    print(f'Fold {fold}: holding out bearings '
          f'{held_out_bearings_for_fold(fold)}')
    _describe('Train', cache, train_rows)
    _describe('Held out', cache, test_rows)

    return (_train_dataset(cache, train_rows, batch_size, dtype, seed),
            _eval_dataset(cache, test_rows, batch_size, dtype))


# def get_test_data(data_path=DEFAULT_DATA_PATH, input_shape=INPUT_SHAPE,
#                   batch_size=BATCH_SIZE, dtype=tf.uint8, fold=FIXED_FOLD):
#     """Load only the held-out dataset for one fold.

#     Deterministic: EVAL_WINDOWS_PER_RECORDING contiguous windows per recording.
#     Evaluation of the dataset at this temporal resolution  is defined by the
#     reference protocol.

#     Args:
#         data_path (str): directory holding the .npz cache.
#         input_shape (tuple): model input shape; must be INPUT_SHAPE.
#         batch_size (int): the batch size.
#         dtype (tf.dtypes.DType, optional): input data type. Defaults to tf.uint8.
#         fold (int): bearing-disjoint fold index. Defaults to FIXED_FOLD.

#     Returns:
#         tf.data.Dataset: held-out dataset yielding (uint8 windows, float labels).
#     """
#     _check_input_shape(input_shape)
#     cache = _load_cache(data_path)

#     _, test_rows = fold_rows(cache['bearing_id'], fold)
#     _describe(f'Held out (fold {fold})', cache, test_rows)

#     return _eval_dataset(cache, test_rows, batch_size, dtype)


def get_samples(data_path=DEFAULT_DATA_PATH, input_shape=INPUT_SHAPE,
                num_samples=1024, fold=FIXED_FOLD, seed=0):
    """Load a block of training windows as a plain uint8 numpy array.

    Used for activation-sparsity measurement and hardware benchmarking, both of
    which need uint8 numpy input rather than a dataset. The draw is over the
    fold's *training* recordings: Akida timings depend on input activity, so the
    windows have to be real, but to avoid data leakage at quantization, they must
    not come from held-out data.

    Up to EVAL_WINDOWS_PER_RECORDING per training recording (360 for a standard
    fold) are the deterministic tiled windows; beyond that the remainder are
    seeded random crops.

    Args:
        data_path (str): directory holding the .npz cache.
        input_shape (tuple): model input shape; must be INPUT_SHAPE.
        num_samples (int): number of windows to return. Defaults to 1024.
        fold (int): bearing-disjoint fold index. Defaults to FIXED_FOLD.
        seed (int): seed for the draw order and for any random crops. Akida
            timings and sparsity depend on input activity, so varying this
            varies the measurement.

    Returns:
        np.ndarray: shape (num_samples,) + INPUT_SHAPE, dtype uint8.
    """
    _check_input_shape(input_shape)
    cache = _load_cache(data_path)
    signals = cache['signals']

    train_rows, _ = fold_rows(cache['bearing_id'], fold)
    rng = np.random.default_rng(seed)

    tiled, _ = _eval_arrays(cache, train_rows)
    idx = rng.permutation(len(tiled))[:num_samples]
    samples = tiled[idx]

    shortfall = num_samples - len(samples)
    if shortfall > 0:
        extra = np.empty((shortfall, WINDOW_LENGTH), dtype=np.uint8)
        for k in range(shortfall):
            row = train_rows[rng.integers(0, len(train_rows))]
            start = rng.integers(0, SIGNAL_LENGTH - WINDOW_LENGTH)
            window = np.array(signals[row, start:start + WINDOW_LENGTH],
                              dtype=np.float32)
            extra[k] = encode_uint8(window)
        samples = np.concatenate([samples, frame_window(extra)])

    return samples.astype(np.uint8)


# ---------------------------------------------------------------------------
# Raw data preparation
# ---------------------------------------------------------------------------
FOLDER_FAULT_TYPE = {
    '1_Healthy': 'Healthy',
    '2_Inner_Race_Faults': 'Inner',
    '3_Outer_Race_Faults': 'Outer',
    '4_Ball_Faults': 'Ball',
    '5_Cage_Faults': 'Cage',
}
FAULT_TYPE_TO_LABEL = {'Inner': 'inner', 'Outer': 'outer',
                       'Ball': 'ball', 'Cage': 'cage'}


def prepare_raw(raw_dir, data_path=DEFAULT_DATA_PATH):
    """Rebuild the .npz cache from the raw Mendeley CSV tree.

    Only needed to reproduce the mirrored cache from the original dataset, or
    after changing what the cache stores. Expects the layout of
    `1_CSV_Raw_Data_Files (.csv)` from https://data.mendeley.com/datasets/y2px5tg92h/5:

        <raw_dir>/1_Healthy/H_<bearing>_0.csv
        <raw_dir>/2_Inner_Race_Faults/I_<bearing>_<severity>.csv
        <raw_dir>/3_Outer_Race_Faults/O_<bearing>_<severity>.csv
        <raw_dir>/4_Ball_Faults/B_<bearing>_<severity>.csv
        <raw_dir>/5_Cage_Faults/C_<bearing>_<severity>.csv

    Folder order and the sorted glob within each folder fix the row ordering,
    which is load-bearing: the fold logic indexes recordings by row.

    Args:
        raw_dir (str): directory holding the five fault-type subfolders.
        data_path (str): directory to write the .npz cache into.

    Returns:
        str: path to the written cache.
    """
    import glob

    import pandas as pd

    signals = np.zeros((N_RECORDINGS, SIGNAL_LENGTH), dtype=np.float32)
    records = []

    for folder, fault_type in FOLDER_FAULT_TYPE.items():
        directory = os.path.join(raw_dir, folder)
        if not os.path.isdir(directory):
            raise FileNotFoundError(f'missing raw folder {directory}')

        for path in sorted(glob.glob(os.path.join(directory, '*.csv'))):
            stem = os.path.splitext(os.path.basename(path))[0]
            _, bearing_id, severity = stem.split('_')
            frame = pd.read_csv(path, usecols=['Accelerometer', 'Speed'])
            vibration = frame['Accelerometer'].to_numpy(dtype=np.float32)
            if vibration.size != SIGNAL_LENGTH:
                raise ValueError(f'{os.path.basename(path)}: expected '
                                 f'{SIGNAL_LENGTH} samples, got {vibration.size}')

            row_idx = len(records)
            signals[row_idx] = vibration
            label = np.zeros(NUM_LABELS, dtype=np.float32)
            if fault_type != 'Healthy':
                label[LABEL_COLUMNS.index(FAULT_TYPE_TO_LABEL[fault_type])] = 1.0

            records.append({
                'waveform_id': f'{bearing_id}_{fault_type}_{severity}',
                'bearing_id': int(bearing_id),
                'fault_type': fault_type,
                'severity': int(severity),
                # The tachometer channel is sparsely populated; the first row's
                # value is what the protocol uses. Analysis only - never an input.
                'rpm': int(frame['Speed'].iloc[0]),
                'label': label,
            })

    if len(records) != N_RECORDINGS:
        raise ValueError(f'expected {N_RECORDINGS} recordings, '
                         f'found {len(records)}')

    os.makedirs(data_path, exist_ok=True)
    path = _cache_path(data_path)
    np.savez(
        path,
        signals=signals,
        labels=np.stack([r['label'] for r in records]),
        row_idx=np.arange(N_RECORDINGS, dtype=np.int32),
        bearing_id=np.array([r['bearing_id'] for r in records], dtype=np.int32),
        severity=np.array([r['severity'] for r in records], dtype=np.int32),
        rpm=np.array([r['rpm'] for r in records], dtype=np.int32),
        waveform_id=np.array([r['waveform_id'] for r in records]),
        fault_type=np.array([r['fault_type'] for r in records]),
        label_columns=np.array(LABEL_COLUMNS),
        fs=np.array(SAMPLE_RATE, dtype=np.int32),
    )
    return path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Build, check and report on the UORED-VAFCLS data pipeline')
    parser.add_argument('-d', '--data', default=DEFAULT_DATA_PATH,
                        help='Directory holding the prepared .npz cache')
    parser.add_argument('--fold', type=int, default=FIXED_FOLD,
                        help='Bearing-disjoint fold to summarise')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for the crop positions, gains and shuffle')
    parser.add_argument('--prepare-raw', metavar='RAW_DIR', default=None,
                        help='Rebuild the .npz cache from the raw Mendeley CSV '
                             'tree instead of downloading it')
    args = parser.parse_args()

    if args.prepare_raw is not None:
        path = prepare_raw(args.prepare_raw, args.data)
        print(f'Wrote {path} ({os.path.getsize(path) / 1e6:.0f} MB)')
        _load_cache.cache_clear()

    if not (args.check_splits or args.report_encoding):
        train_ds, test_ds = get_data(args.data, fold=args.fold, seed=args.seed)
        samples = get_samples(args.data, num_samples=100, fold=args.fold,
                              seed=args.seed)
        print(f'\nBatches: train {len(train_ds)}, held out {len(test_ds)}')
        print(f'Samples for benchmarking: {samples.shape} {samples.dtype} '
              f'range [{samples.min()}, {samples.max()}]')
