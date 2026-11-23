# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Waveform augmentations for training the Pico KWS model on your own data.

The model reads raw 16 kHz audio on one global int16 scale, so every augmentation here takes
and returns a 1-D ``float32`` array **in the int16 range** (about -32768 to 32767) and keeps
its length, so a clip that was fitted to the model window stays fitted. Nothing normalises
a clip to its own peak: a live microphone has no per-clip peak.

An augmentation is a function ``apply(audio, rng) -> audio`` where ``rng`` is a
``numpy.random.Generator``. It is built by a *factory*, ``factory(context, **params)``, which
is registered under a name. The training config lists augmentations by name; each step runs
with probability ``p``, in the order listed:

    augmentations:
      - {name: speed, p: 0.5, min_factor: 0.9, max_factor: 1.1}
      - {name: background_noise, p: 0.8, min_snr_db: 5, max_snr_db: 20}

Built in: ``speed``, ``pitch_shift``, ``time_shift``, ``echo``, ``background_noise``, ``gain``.

Adding your own takes one function. Register it here with :func:`register`, or put it in a
module of yours and name it in the config as ``module:factory``:

    def my_clipping(context, level=0.9):
        def apply(audio, rng):
            limit = level * 32767
            return np.clip(audio, -limit, limit).astype(np.float32)
        return apply

    augmentations:
      - {name: "my_module:my_clipping", p: 0.2, level: 0.8}

``context`` is a dict with ``sample_rate`` and ``noise_clips`` (a list of background-noise
recordings, empty if you have none).
"""
import importlib
import math
from fractions import Fraction

import numpy as np
from scipy import signal

SAMPLE_RATE = 16000
INT16_MIN, INT16_MAX = -32768.0, 32767.0

AUGMENTATIONS = {}


def register(name):
    """Decorator that registers an augmentation factory under ``name``."""
    def decorator(factory):
        AUGMENTATIONS[name] = factory
        return factory
    return decorator


def fit_length(audio, length):
    """Zero-pad at the end, or crop the end, to ``length`` samples (the start is kept)."""
    if len(audio) < length:
        return np.pad(audio, (0, length - len(audio)))
    return audio[:length]


def rms(audio):
    """Root mean square of a signal."""
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def _clip(audio):
    return np.clip(audio, INT16_MIN, INT16_MAX).astype(np.float32)


def _check_range(name, low, high):
    if low > high:
        raise ValueError(f'{name}: the minimum ({low}) is larger than the maximum ({high})')


@register('speed')
def speed(context, min_factor=0.9, max_factor=1.1):
    """Play the clip faster or slower by resampling it.

    A factor above 1 is faster and shorter. **This also shifts the pitch**, the way a tape
    played at the wrong speed does. For pitch alone, see ``pitch_shift``. The clip is padded
    or cropped at the end back to its original length.
    """
    _check_range('speed', min_factor, max_factor)

    def apply(audio, rng):
        factor = Fraction(rng.uniform(min_factor, max_factor)).limit_denominator(32)
        if factor == 1:
            return audio
        resampled = signal.resample_poly(audio, factor.denominator, factor.numerator)
        return _clip(fit_length(resampled, len(audio)))
    return apply


@register('pitch_shift')
def pitch_shift(context, min_semitones=-2.0, max_semitones=2.0):
    """Shift the pitch without changing the duration. Needs ``librosa`` (not installed by default).

    Much slower than ``speed``, so give it a modest ``p``.
    """
    _check_range('pitch_shift', min_semitones, max_semitones)
    try:
        import librosa
    except ImportError as e:
        raise ImportError("the 'pitch_shift' augmentation needs librosa: pip install librosa. "
                          "The 'speed' augmentation shifts pitch without it.") from e
    sample_rate = context['sample_rate']

    def apply(audio, rng):
        shifted = librosa.effects.pitch_shift(audio, sr=sample_rate,
                                              n_steps=rng.uniform(min_semitones, max_semitones))
        return _clip(fit_length(shifted, len(audio)))
    return apply


@register('time_shift')
def time_shift(context, max_shift_ms=100):
    """Move the clip earlier or later by up to ``max_shift_ms``, filling with silence."""
    max_shift = int(max_shift_ms * context['sample_rate'] / 1000)

    def apply(audio, rng):
        shift = int(rng.integers(-max_shift, max_shift + 1))
        if shift > 0:
            return np.concatenate([np.zeros(shift, np.float32), audio[:-shift]])
        if shift < 0:
            return np.concatenate([audio[-shift:], np.zeros(-shift, np.float32)])
        return audio
    return apply


@register('echo')
def echo(context, min_rt60=0.1, max_rt60=0.5, min_wet=0.2, max_wet=0.8, ir_dir=None):
    """Add room reverberation by convolving the clip with an impulse response.

    By default the impulse response is synthetic: exponentially decaying noise whose decay
    time (RT60, in seconds) is drawn from ``[min_rt60, max_rt60]``, added to the direct
    sound at a level ``wet`` (the tail's amplitude relative to the direct sound, drawn from
    ``[min_wet, max_wet]``). To use recorded rooms instead, point ``ir_dir`` at a folder of
    impulse-response WAV files. The output keeps the input's RMS level.
    """
    _check_range('echo (rt60)', min_rt60, max_rt60)
    _check_range('echo (wet)', min_wet, max_wet)
    sample_rate = context['sample_rate']
    recorded = None
    if ir_dir is not None:
        import pathlib
        from kws_custom_data import read_wav
        files = sorted(pathlib.Path(ir_dir).glob('*.wav'))
        if not files:
            raise ValueError(f"echo: no .wav impulse responses in '{ir_dir}'")
        recorded = [read_wav(f, sample_rate) for f in files]

    def apply(audio, rng):
        if recorded is not None:
            ir = recorded[rng.integers(len(recorded))]
            ir = ir / max(np.max(np.abs(ir)), 1e-9)
        else:
            rt60 = rng.uniform(min_rt60, max_rt60)
            t = np.arange(int(rt60 * sample_rate)) / sample_rate
            tail = rng.standard_normal(len(t)) * 10 ** (-3 * t / rt60)    # -60 dB at RT60
            ir = tail / max(np.sqrt(np.sum(tail ** 2)), 1e-9) * rng.uniform(min_wet, max_wet)
            ir[0] += 1.0                                                  # the direct sound
        wet = signal.fftconvolve(audio, ir)[:len(audio)]
        level = rms(wet)
        if level > 0:
            wet = wet * (rms(audio) / level)
        return _clip(wet)
    return apply


@register('background_noise')
def background_noise(context, min_snr_db=5, max_snr_db=20, silent_gain_max=1.0):
    """Mix in a random stretch of a background-noise recording at a random loudness.

    The signal-to-noise ratio is drawn from ``[min_snr_db, max_snr_db]`` and measured over
    the whole window, padding included. A clip that is effectively silent has no meaningful
    SNR, so the noise is instead scaled to a random fraction (up to ``silent_gain_max``) of
    its recorded level. Does nothing if there are no noise recordings.
    """
    _check_range('background_noise', min_snr_db, max_snr_db)
    clips = context['noise_clips']

    def apply(audio, rng):
        if not clips:
            return audio
        noise = clips[rng.integers(len(clips))]
        n = len(audio)
        if len(noise) < n:
            noise = np.tile(noise, math.ceil(n / len(noise)))
        start = int(rng.integers(0, len(noise) - n + 1))
        segment = noise[start:start + n]
        segment_level = rms(segment)
        if segment_level < 1e-6:
            return audio
        signal_level = rms(audio)
        if signal_level < 1.0:
            gain = rng.uniform(0.0, silent_gain_max)
        else:
            snr_db = rng.uniform(min_snr_db, max_snr_db)
            gain = signal_level / 10 ** (snr_db / 20) / segment_level
        return _clip(audio + gain * segment)
    return apply


@register('gain')
def gain(context, min_db=-12.0, max_db=6.0):
    """Make the clip quieter or louder by a random number of decibels (clipped at int16)."""
    _check_range('gain', min_db, max_db)

    def apply(audio, rng):
        return _clip(audio * 10 ** (rng.uniform(min_db, max_db) / 20))
    return apply


class Compose:
    """Runs a list of ``(name, probability, function)`` steps in order."""

    def __init__(self, steps):
        self.steps = steps

    def __call__(self, audio, rng):
        for _, p, apply in self.steps:
            if p >= 1 or rng.random() < p:
                audio = apply(audio, rng)
        return audio

    def __len__(self):
        return len(self.steps)

    def __repr__(self):
        return 'Compose([' + ', '.join(f'{n} (p={p})' for n, p, _ in self.steps) + '])'


def resolve(name):
    """Look up a factory by its registered name, or import ``module:factory``."""
    if ':' in name:
        module, attribute = name.split(':', 1)
        return getattr(importlib.import_module(module), attribute)
    if name not in AUGMENTATIONS:
        raise ValueError(f"unknown augmentation '{name}'. Built in: {', '.join(sorted(AUGMENTATIONS))}; "
                         "or name your own as 'module:factory'")
    return AUGMENTATIONS[name]


def build_pipeline(specs, noise_clips=None, sample_rate=SAMPLE_RATE):
    """Build a :class:`Compose` from config entries.

    Args:
        specs (list[dict]): each has ``name``, an optional ``p`` (probability, default 1) and
            the parameters of that augmentation.
        noise_clips (list[np.ndarray], optional): background-noise recordings, int16 range.
        sample_rate (int): sample rate of the audio.
    """
    context = {'sample_rate': sample_rate, 'noise_clips': list(noise_clips or [])}
    steps = []
    for spec in specs or []:
        params = dict(spec)
        name = params.pop('name')
        p = float(params.pop('p', 1.0))
        if not 0 <= p <= 1:
            raise ValueError(f"{name}: p must be between 0 and 1, got {p}")
        steps.append((name, p, resolve(name)(context, **params)))
    return Compose(steps)
