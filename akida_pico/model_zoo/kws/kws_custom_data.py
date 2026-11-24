# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Train the Pico KWS model on your own keywords: data loading from a folder of recordings.

Layout (the folder name is the label)::

    <data>/
        hello/*.wav          one folder per keyword listed in the config
        stop/*.wav
        silence/*.wav        optional: your own "no speech" clips
        unknown/*.wav        optional: other speech; every other folder is "unknown" too
        _background_noise_/*.wav    long noise recordings, for silence and augmentation

Classes are the keywords in the config order, then ``silence`` and ``unknown`` if enabled.
That is the same head structure as the 12-class Speech Commands model; measure the mapping
on Pico after changing the number of classes rather than assuming it matches.

Recordings can be any sample rate and mono or stereo, in 8/16/24/32-bit PCM or float WAV;
they are resampled to 16 kHz, mixed to mono and fitted to the model's 16,384-sample window
(zero-padded at the end, or cropped keeping the last samples, as the Speech Commands recipe
does). Convert other formats first, e.g. ``ffmpeg -i in.mp3 out.wav``.

**Splits.** Files are assigned to train, validation and test by a hash of the file name,
so the split is the same on every run. The part of the name before ``_nohash_`` is the
grouping key, as in Speech Commands (``<speaker>_nohash_<n>.wav``): put the speaker id there
and no speaker is in two splits. Files with the same name in different folders also stay
together. Without that pattern each file is its own group.

**Silence.** Without a ``silence/`` folder, silence clips are cut from the background-noise
recordings at a random volume. Each recording is split in time between the three splits
(first part to train, then validation, then test), so the test split never hears training
noise. Validation and test sets are balanced: ``unknown`` and generated ``silence`` are
capped at the median keyword count of the split.
"""
import hashlib
import pathlib
import warnings
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy import signal
from scipy.io import wavfile

from kws_data import SAMPLE_LENGTH, SAMPLE_RATE

SPLITS = ('train', 'validation', 'test')
NOISE_FOLDER = '_background_noise_'

DEFAULT_CONFIG = {
    # data
    'data': 'data/keywords',
    'keywords': None,           # required: list of keyword folder names
    'silence': True,
    'unknown': True,
    'validation_fraction': 0.1,
    'test_fraction': 0.1,
    'silence_clips': None,      # silence clips per validation/test split; default: median keyword count
    # training
    'epochs': 100,
    'batch_size': 128,
    'clips_per_class': None,    # clips per class per epoch; default: the largest keyword class
    'model': None,              # None: tenn_recurrent_sc12; or 'module:function(num_classes)'
    'seed': 42,
    'augmentations': [],
}


def load_config(path):
    """Read a YAML training config, check it, and fill in the defaults."""
    import yaml
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    unknown = set(user) - set(DEFAULT_CONFIG)
    if unknown:
        raise ValueError(f"{path}: unknown setting(s) {sorted(unknown)}. "
                         f"Valid: {sorted(DEFAULT_CONFIG)}")
    config = {**DEFAULT_CONFIG, **user}
    if not config['keywords']:
        raise ValueError(f"{path}: 'keywords' is required, e.g. keywords: [hello, stop]")
    bad = [k for k in config['keywords'] if not isinstance(k, str)]
    if bad:
        raise ValueError(f"{path}: keywords must be text, but {bad} were read as other types. YAML reads "
                         "an unquoted yes, no, on, off, true or false as a boolean (and 1, 2 as numbers): "
                         "quote them, e.g. keywords: ['yes', 'no', 'on', 'off']")
    if len(set(config['keywords'])) != len(config['keywords']):
        raise ValueError(f"{path}: 'keywords' has duplicates")
    reserved = {'silence', 'unknown'} & set(config['keywords'])
    if reserved:
        raise ValueError(f"{path}: {sorted(reserved)} can't be keywords; use the silence/unknown settings")
    if config['validation_fraction'] + config['test_fraction'] >= 1:
        raise ValueError(f"{path}: validation_fraction + test_fraction must be below 1")
    return config


def read_wav(path, sample_rate=SAMPLE_RATE):
    """Read a WAV file as mono ``float32`` on the int16 scale, resampled to ``sample_rate``."""
    rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) * 256
    elif data.dtype == np.int16:
        data = data.astype(np.float32)
    elif data.dtype == np.int32:
        data = data.astype(np.float64) / 65536
    elif np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float64) * 32768     # float WAVs are in [-1, 1]
    else:
        raise ValueError(f'{path}: unsupported WAV sample format {data.dtype}')
    if rate != sample_rate:
        ratio = Fraction(sample_rate, rate)
        data = signal.resample_poly(data, ratio.numerator, ratio.denominator)
    return np.clip(data, -32768, 32767).astype(np.float32)


def fit_window(audio, length=SAMPLE_LENGTH):
    """Zero-pad at the end, or crop keeping the last samples (the Speech Commands rule)."""
    if len(audio) < length:
        return np.pad(audio, (0, length - len(audio)))
    return audio[-length:]


def split_of(path, validation_fraction=0.1, test_fraction=0.1):
    """Split a file belongs to, from a hash of its group key (see the module docstring)."""
    key = pathlib.Path(path).stem.split('_nohash_')[0]
    position = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 10000 / 10000
    if position < test_fraction:
        return 'test'
    if position < test_fraction + validation_fraction:
        return 'validation'
    return 'train'


@dataclass
class KwsDataset:
    """Clips as ``(X, y)`` per split: ``X`` is ``(N, 16384, 1)`` float32 on the int16 scale."""
    labels: list
    train: tuple
    validation: tuple
    test: tuple
    noise: list = field(default_factory=list)   # the training part of the noise recordings
    num_keywords: int = 0

    @property
    def num_classes(self):
        return len(self.labels)

    def default_clips_per_class(self):
        """The size of the largest keyword class in the training split."""
        counts = _counts(self.train[1], self.num_classes)[:self.num_keywords]
        return int(counts.max())


def _counts(labels, num_classes):
    """Clips per class (an empty list gives all zeros instead of an error)."""
    return np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)


def _wav_files(folder):
    return sorted(p for p in pathlib.Path(folder).rglob('*') if p.suffix.lower() == '.wav')


def _stack(clips, labels):
    if not clips:
        return np.zeros((0, SAMPLE_LENGTH, 1), np.float32), np.zeros((0,), np.int64)
    return np.stack(clips)[..., np.newaxis], np.array(labels, dtype=np.int64)


def _noise_portions(root, validation_fraction, test_fraction, sample_rate):
    """Split each noise recording in time: train first, then validation, then test."""
    portions = {s: [] for s in SPLITS}
    for path in _wav_files(pathlib.Path(root) / NOISE_FOLDER):
        audio = read_wav(path, sample_rate)
        a = int(len(audio) * (1 - validation_fraction - test_fraction))
        b = int(len(audio) * (1 - test_fraction))
        for name, part in zip(SPLITS, (audio[:a], audio[a:b], audio[b:])):
            if len(part) >= 1:
                portions[name].append(part)
    return portions


def load_dataset(root, keywords, silence=True, unknown=True, validation_fraction=0.1,
                 test_fraction=0.1, silence_clips=None, seed=42, sample_rate=SAMPLE_RATE,
                 length=SAMPLE_LENGTH):
    """Load a folder of recordings (see the module docstring) into a :class:`KwsDataset`."""
    root = pathlib.Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"data folder '{root}' not found")
    folders = {p.name: p for p in sorted(root.iterdir())
               if p.is_dir() and not p.name.startswith(('_', '.'))}
    missing = [k for k in keywords if k not in folders]
    if missing:
        raise FileNotFoundError(f"no folder for keyword(s) {missing} in '{root}'; found {sorted(folders)}")

    labels = list(keywords) + (['silence'] if silence else []) + (['unknown'] if unknown else [])
    index = {name: i for i, name in enumerate(labels)}
    rng = np.random.default_rng(seed)

    clips = {s: ([], []) for s in SPLITS}

    def add(path, label):
        split = split_of(path, validation_fraction, test_fraction)
        clips[split][0].append(fit_window(read_wav(path, sample_rate), length))
        clips[split][1].append(label)

    for name in keywords:
        for path in _wav_files(folders[name]):
            add(path, index[name])
    if silence and 'silence' in folders:
        for path in _wav_files(folders['silence']):
            add(path, index['silence'])
    others = [n for n in folders if n not in keywords and n != 'silence']
    if unknown:
        for name in others:
            for path in _wav_files(folders[name]):
                add(path, index['unknown'])
    elif others:
        warnings.warn(f"ignoring folders {others}: 'unknown' is off in the config")

    noise = _noise_portions(root, validation_fraction, test_fraction, sample_rate)

    # Balance validation and test: cap unknown, and generate silence, at the median keyword count.
    for split in ('validation', 'test'):
        xs, ys = clips[split]
        counts = _counts(ys, len(labels))[:len(keywords)]
        target = int(np.median(counts)) if len(ys) else 0
        if unknown:
            positions = [i for i, y in enumerate(ys) if y == index['unknown']]
            if len(positions) > target:
                drop = set(rng.choice(positions, len(positions) - target, replace=False).tolist())
                clips[split] = ([x for i, x in enumerate(xs) if i not in drop],
                                [y for i, y in enumerate(ys) if i not in drop])
    if silence and 'silence' not in folders:
        if not noise['train']:
            raise FileNotFoundError(
                f"silence is on but there is no '{NOISE_FOLDER}' folder with .wav noise in '{root}' "
                "and no 'silence' folder; add one of them or set silence: false")
        for split in SPLITS:
            if not noise[split]:
                continue
            xs, ys = clips[split]
            counts = _counts(ys, len(labels))[:len(keywords)]
            n = (silence_clips if split != 'train' and silence_clips else
                 int(np.median(counts)) if len(ys) else 0)
            for _ in range(max(n, 0)):
                recording = noise[split][rng.integers(len(noise[split]))]
                if len(recording) < length:
                    recording = np.tile(recording, -(-length // len(recording)))
                start = int(rng.integers(0, len(recording) - length + 1))
                volume = rng.uniform(0.0, 1.0)
                xs.append((recording[start:start + length] * volume).astype(np.float32))
                ys.append(index['silence'])

    data = {s: _stack(*clips[s]) for s in SPLITS}
    for split in ('validation', 'test'):
        present = set(data[split][1].tolist())
        absent = [labels[i] for i in range(len(labels)) if i not in present]
        if absent:
            warnings.warn(f"the {split} split has no clips of {absent}: add more recordings, "
                          "or change validation_fraction / test_fraction")
    if not len(data['train'][1]):
        raise ValueError(f"no training clips found in '{root}'")
    return KwsDataset(labels=labels, train=data['train'], validation=data['validation'],
                      test=data['test'], noise=noise['train'], num_keywords=len(keywords))


def load_dataset_from_config(config, data_override=None):
    """:func:`load_dataset` with the settings of a config from :func:`load_config`."""
    return load_dataset(
        data_override or config['data'], config['keywords'], silence=config['silence'],
        unknown=config['unknown'], validation_fraction=config['validation_fraction'],
        test_fraction=config['test_fraction'], silence_clips=config['silence_clips'],
        seed=config['seed'])


def balanced_indices(y, clips_per_class, rng):
    """One epoch of indices: ``clips_per_class`` per class, upsampling small classes."""
    chosen = []
    for c in np.unique(y):
        members = np.where(y == c)[0]
        chosen.append(rng.choice(members, size=clips_per_class, replace=len(members) < clips_per_class))
    indices = np.concatenate(chosen)
    return rng.permutation(indices).astype(np.int64)


def make_train_dataset(X, y, batch_size=128, clips_per_class=None, augment=None, seed=42,
                       num_parallel_calls=2):
    """Infinite, class-balanced, augmented ``tf.data`` stream of ``(clips, labels)`` batches.

    Every epoch (``steps_per_epoch`` batches) draws ``clips_per_class`` clips of each class
    at random, upsampling the small classes with replacement. Augmentation runs on the CPU
    in numpy and scipy, which share Python's GIL: more parallel calls than 2 only adds
    contention, so the default is 2. Each clip's augmentation is seeded by its position in
    the stream, so a run repeats exactly for the same seed.

    Returns:
        (tf.data.Dataset, int): the stream and the number of batches per epoch.
    """
    import tensorflow as tf

    classes = np.unique(y)
    if clips_per_class is None:
        clips_per_class = int(np.bincount(y).max())
    per_epoch = len(classes) * clips_per_class
    steps_per_epoch = per_epoch // batch_size
    if steps_per_epoch < 1:
        raise ValueError(f'an epoch has {per_epoch} clips, fewer than one batch of {batch_size}: '
                         'lower batch_size or raise clips_per_class')
    labels = tf.constant(y, dtype=tf.int64)

    def epochs():
        epoch = 0
        while True:
            yield balanced_indices(y, clips_per_class, np.random.default_rng([seed, epoch]))
            epoch += 1

    def load(position, index):
        def py_load(position, index):
            audio = X[int(index), :, 0]
            if augment is not None:
                audio = augment(audio, np.random.default_rng([seed, 1, int(position)]))
            return audio.astype(np.float32)[:, np.newaxis]
        clip = tf.numpy_function(py_load, [position, index], tf.float32)
        clip.set_shape(X.shape[1:])
        return clip, labels[index]

    indices = tf.data.Dataset.from_generator(
        epochs, output_signature=tf.TensorSpec((per_epoch,), tf.int64)
    ).flat_map(tf.data.Dataset.from_tensor_slices).enumerate()
    dataset = indices.map(load, num_parallel_calls=num_parallel_calls)
    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return dataset, steps_per_epoch


def make_eval_dataset(X, y, batch_size=128):
    """Plain batches of a validation or test split (no augmentation, nothing dropped)."""
    import tensorflow as tf
    return tf.data.Dataset.from_tensor_slices((X, y.astype(np.int64))).batch(batch_size)


def calibration_samples(X, count=100, length=1024, seed=42):
    """Quantization calibration set: ``count`` segments of ``length`` samples as int16.

    Each is cut from the middle of a random training clip, as the Speech Commands set
    was. Calibrate on audio that covers every class and the noise you expect.
    """
    rng = np.random.default_rng(seed)
    clips = X[rng.choice(len(X), size=count, replace=len(X) < count), :, 0]
    start = X.shape[1] // 2
    return clips[:, start:start + length, np.newaxis].astype(np.int16)
