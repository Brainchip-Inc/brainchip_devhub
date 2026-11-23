"""Unit tests for the Pico KWS custom-data training code (no dataset download, no hardware)."""
import pathlib
import sys

import numpy as np
import pytest
from scipy.io import wavfile

KWS_DIR = pathlib.Path(__file__).resolve().parents[1] / 'akida_pico' / 'model_zoo' / 'kws'
sys.path.insert(0, str(KWS_DIR))

import kws_augment  # noqa: E402
import kws_custom_data as kcd  # noqa: E402

SR = 16000
N = 16384


def tone(n=N, freq=440.0, amplitude=8000.0):
    return (amplitude * np.sin(2 * np.pi * freq * np.arange(n) / SR)).astype(np.float32)


def pipeline(name, noise=None, **params):
    return kws_augment.build_pipeline([{'name': name, **params}], noise_clips=noise)


# ---------------------------------------------------------------- augmentations

@pytest.mark.parametrize('name, params', [
    ('speed', {}), ('time_shift', {}), ('echo', {}), ('background_noise', {}), ('gain', {}),
])
def test_builtin_augmentations_keep_length_dtype_and_range(name, params):
    noise = [np.random.default_rng(0).normal(0, 3000, 2 * SR).astype(np.float32)]
    augment = pipeline(name, noise=noise, **params)
    for seed in range(5):
        out = augment(tone(), np.random.default_rng(seed))
        assert out.shape == (N,) and out.dtype == np.float32
        assert np.all(np.isfinite(out)) and np.abs(out).max() <= 32768


def test_same_seed_gives_same_output():
    steps = [{'name': 'speed'}, {'name': 'echo'}, {'name': 'gain'}]
    augment = kws_augment.build_pipeline(steps)
    a = augment(tone(), np.random.default_rng(7))
    b = augment(tone(), np.random.default_rng(7))
    c = augment(tone(), np.random.default_rng(8))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_probability_zero_leaves_audio_untouched():
    augment = kws_augment.build_pipeline([{'name': 'gain', 'p': 0, 'min_db': -20, 'max_db': -20}])
    audio = tone()
    assert np.array_equal(augment(audio, np.random.default_rng(0)), audio)


def test_gain_applies_the_requested_decibels():
    out = pipeline('gain', min_db=-6, max_db=-6)(tone(), np.random.default_rng(0))
    assert np.allclose(out, tone() * 10 ** (-6 / 20), atol=1e-2)


def test_gain_clips_at_int16():
    out = pipeline('gain', min_db=20, max_db=20)(tone(), np.random.default_rng(0))
    assert out.max() == 32767 and out.min() == -32768


def test_time_shift_moves_audio_and_fills_with_silence():
    audio = np.arange(1, 101, dtype=np.float32)
    augment = kws_augment.build_pipeline([{'name': 'time_shift', 'max_shift_ms': 1}])
    seen = set()
    for seed in range(40):
        out = augment(audio, np.random.default_rng(seed))
        shift = int(np.argmax(out != 0)) if out[0] == 0 else -int(100 - np.count_nonzero(out))
        seen.add(shift)
        if shift > 0:
            assert np.array_equal(out[shift:], audio[:-shift])
        elif shift < 0:
            assert np.array_equal(out[:shift], audio[-shift:]) and not out[shift:].any()
    assert len(seen) > 3 and max(abs(s) for s in seen) <= 16     # 1 ms is 16 samples


def test_speed_changes_the_pitch():
    out = pipeline('speed', min_factor=1.25, max_factor=1.25)(tone(freq=400), np.random.default_rng(0))
    spectrum = np.abs(np.fft.rfft(out[:8192]))
    assert np.argmax(spectrum) * SR / 8192 == pytest.approx(500, abs=5)    # 400 Hz * 1.25


def test_background_noise_hits_the_requested_snr():
    noise = [np.random.default_rng(1).normal(0, 2000, 3 * SR).astype(np.float32)]
    out = pipeline('background_noise', noise=noise, min_snr_db=10, max_snr_db=10)(
        tone(), np.random.default_rng(0))
    snr = 20 * np.log10(kws_augment.rms(tone()) / kws_augment.rms(out - tone()))
    assert snr == pytest.approx(10, abs=0.1)


def test_background_noise_on_silence_is_bounded_by_its_recorded_level():
    noise = [np.random.default_rng(1).normal(0, 2000, 3 * SR).astype(np.float32)]
    out = pipeline('background_noise', noise=noise, silent_gain_max=0.5)(
        np.zeros(N, np.float32), np.random.default_rng(0))
    assert 0 < kws_augment.rms(out) <= 0.5 * 2100


def test_background_noise_without_recordings_does_nothing():
    audio = tone()
    assert np.array_equal(pipeline('background_noise')(audio, np.random.default_rng(0)), audio)


def test_echo_adds_a_tail_and_keeps_the_level():
    click = np.zeros(N, np.float32)
    click[100:140] = 10000
    out = pipeline('echo', min_rt60=0.3, max_rt60=0.3)(click, np.random.default_rng(0))
    assert np.abs(out[1000:]).max() > 0                   # reverberation after the click
    assert 100 <= np.argmax(np.abs(out)) < 140            # the direct sound is still the loudest part
    assert kws_augment.rms(out) == pytest.approx(kws_augment.rms(click), rel=1e-3)


def test_echo_with_recorded_impulse_responses(tmp_path):
    ir = np.zeros(800, np.int16)
    ir[0], ir[400] = 20000, 10000                         # direct sound and one echo at 25 ms
    wavfile.write(tmp_path / 'room.wav', SR, ir)
    click = np.zeros(N, np.float32)
    click[100] = 10000
    out = pipeline('echo', ir_dir=str(tmp_path))(click, np.random.default_rng(0))
    assert np.argmax(np.abs(out)) == 100 and np.abs(out[500]) == pytest.approx(np.abs(out[100]) / 2, rel=0.01)


def test_echo_without_impulse_responses_in_the_folder_is_an_error(tmp_path):
    with pytest.raises(ValueError, match='no .wav'):
        pipeline('echo', ir_dir=str(tmp_path))


def my_clipping(context, level=0.5):
    def apply(audio, rng):
        return np.clip(audio, -level * 32767, level * 32767).astype(np.float32)
    return apply


def test_custom_augmentation_by_module_name():
    augment = kws_augment.build_pipeline([{'name': 'test_kws_custom:my_clipping', 'level': 0.1}])
    assert np.abs(augment(tone(), np.random.default_rng(0))).max() <= 0.1 * 32767 + 1


def test_unknown_augmentation_and_bad_ranges_are_clear_errors():
    with pytest.raises(ValueError, match='unknown augmentation'):
        kws_augment.build_pipeline([{'name': 'warp'}])
    with pytest.raises(ValueError, match='minimum'):
        pipeline('gain', min_db=3, max_db=-3)
    with pytest.raises(ValueError, match='between 0 and 1'):
        kws_augment.build_pipeline([{'name': 'gain', 'p': 2}])


# ------------------------------------------------------------------------- data

def write_wav(path, n=8000, rate=SR, dtype=np.int16, seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.default_rng(seed).integers(-3000, 3000, n).astype(dtype)
    wavfile.write(path, rate, data)


@pytest.fixture
def folder(tmp_path):
    """Two keywords, one 'other word' folder, and a noise recording; 80 speakers each."""
    for label, seed in (('alpha', 1), ('beta', 2), ('other', 3)):
        for i in range(80):
            write_wav(tmp_path / label / f'spk{i}_nohash_0.wav', seed=seed * 1000 + i)
    write_wav(tmp_path / kcd.NOISE_FOLDER / 'hum.wav', n=10 * SR, seed=9)
    return tmp_path


def test_read_wav_converts_formats_to_the_int16_scale(tmp_path):
    ints = np.array([0, 16384, -16384, 32767], np.int16)
    wavfile.write(tmp_path / 'i16.wav', SR, ints)
    wavfile.write(tmp_path / 'f32.wav', SR, (ints / 32768).astype(np.float32))
    wavfile.write(tmp_path / 'i32.wav', SR, ints.astype(np.int32) * 65536)
    wavfile.write(tmp_path / 'u8.wav', SR, (ints // 256 + 128).astype(np.uint8))
    for name, tolerance in (('i16', 0), ('f32', 1), ('i32', 1), ('u8', 256)):
        assert np.allclose(kcd.read_wav(tmp_path / f'{name}.wav'), ints, atol=tolerance), name


def test_read_wav_resamples_and_mixes_stereo_to_mono(tmp_path):
    stereo = np.stack([tone(8000, 300), tone(8000, 300)], axis=1).astype(np.int16)
    wavfile.write(tmp_path / 's.wav', 8000, stereo)                    # 1 s at 8 kHz
    audio = kcd.read_wav(tmp_path / 's.wav')
    assert audio.ndim == 1 and len(audio) == pytest.approx(16000, abs=2)


def test_fit_window_pads_the_end_and_crops_keeping_the_last_samples():
    short = kcd.fit_window(np.ones(10, np.float32), 16)
    assert np.array_equal(short, np.r_[np.ones(10), np.zeros(6)])
    assert np.array_equal(kcd.fit_window(np.arange(20, dtype=np.float32), 16), np.arange(4, 20))


def test_split_is_stable_and_groups_by_speaker():
    a = kcd.split_of('x/hello/abc123_nohash_0.wav')
    assert a == kcd.split_of('y/stop/abc123_nohash_7.wav') == kcd.split_of('abc123_nohash_1.wav')
    shares = [kcd.split_of(f'{i}.wav', 0.2, 0.2) for i in range(2000)]
    assert 0.15 < shares.count('test') / 2000 < 0.25 and 0.15 < shares.count('validation') / 2000 < 0.25


def test_load_dataset_classes_splits_and_balance(folder):
    data = kcd.load_dataset(folder, ['alpha', 'beta'], validation_fraction=0.2, test_fraction=0.2)
    assert data.labels == ['alpha', 'beta', 'silence', 'unknown']
    for split in kcd.SPLITS:
        X, y = getattr(data, split)
        assert X.shape[1:] == (kcd.SAMPLE_LENGTH, 1) and X.dtype == np.float32 and len(X) == len(y)
        assert set(y.tolist()) == {0, 1, 2, 3}
    for split in ('validation', 'test'):                     # balanced: unknown/silence <= median keyword count
        y = getattr(data, split)[1]
        keyword_median = int(np.median(np.bincount(y, minlength=4)[:2]))
        assert (y == 3).sum() <= keyword_median and (y == 2).sum() == keyword_median
    assert data.noise and data.default_clips_per_class() == np.bincount(data.train[1])[:2].max()


def test_files_of_one_speaker_stay_in_one_split(folder):
    data = kcd.load_dataset(folder, ['alpha', 'beta'], validation_fraction=0.2, test_fraction=0.2)
    sizes = {s: len(getattr(data, s)[1]) for s in kcd.SPLITS}
    # 80 speakers x 3 folders, the noise-generated silence on top; 'other' is capped in val/test.
    assert sizes['train'] > sizes['validation'] and sizes['train'] > sizes['test']


def test_loading_is_deterministic(folder):
    a = kcd.load_dataset(folder, ['alpha'], seed=5)
    b = kcd.load_dataset(folder, ['alpha'], seed=5)
    assert all(np.array_equal(getattr(a, s)[0], getattr(b, s)[0]) for s in kcd.SPLITS)


def test_noise_recordings_are_split_in_time_between_splits(folder):
    data = kcd.load_dataset(folder, ['alpha'], validation_fraction=0.2, test_fraction=0.2)
    full = kcd.read_wav(folder / kcd.NOISE_FOLDER / 'hum.wav')
    assert len(data.noise) == 1 and len(data.noise[0]) == int(len(full) * 0.6)


def test_missing_keyword_folder_and_missing_noise_are_clear_errors(folder, tmp_path_factory):
    with pytest.raises(FileNotFoundError, match="keyword.*'gamma'|gamma"):
        kcd.load_dataset(folder, ['alpha', 'gamma'])
    no_noise = tmp_path_factory.mktemp('no_noise')
    for i in range(10):
        write_wav(no_noise / 'alpha' / f's{i}.wav')
    with pytest.raises(FileNotFoundError, match='_background_noise_'):
        kcd.load_dataset(no_noise, ['alpha'])
    assert kcd.load_dataset(no_noise, ['alpha'], silence=False, unknown=False).labels == ['alpha']


def test_own_silence_folder_is_used_instead_of_generating(folder):
    for i in range(20):
        write_wav(folder / 'silence' / f'room{i}.wav', seed=500 + i)
    data = kcd.load_dataset(folder, ['alpha', 'beta'])
    assert data.labels[2] == 'silence' and (data.train[1] == 2).sum() <= 20


def test_train_stream_is_balanced_augmented_and_reproducible():
    X = np.concatenate([np.full((30, 64, 1), 100.0, np.float32), np.full((6, 64, 1), 200.0, np.float32)])
    y = np.array([0] * 30 + [1] * 6)
    augment = kws_augment.build_pipeline([{'name': 'gain', 'min_db': -6, 'max_db': -6}])

    def first_epoch(seed):
        dataset, steps = kcd.make_train_dataset(X, y, batch_size=10, clips_per_class=20,
                                                augment=augment, seed=seed)
        batches = list(dataset.take(steps))
        return steps, np.concatenate([b[0].numpy() for b in batches]), np.concatenate(
            [b[1].numpy() for b in batches])

    steps, clips, labels = first_epoch(3)
    assert steps == 4 and clips.shape == (40, 64, 1)
    assert np.bincount(labels).tolist() == [20, 20]                   # balanced, class 1 upsampled
    assert np.allclose(clips[labels == 0], 100 * 10 ** (-6 / 20), rtol=1e-3)     # augmentation ran
    again = first_epoch(3)
    assert np.array_equal(clips, again[1]) and np.array_equal(labels, again[2])
    assert not np.array_equal(labels, first_epoch(4)[2])


def test_train_stream_needs_at_least_one_batch():
    with pytest.raises(ValueError, match='fewer than one batch'):
        kcd.make_train_dataset(np.zeros((4, 8, 1), np.float32), np.array([0, 0, 1, 1]),
                               batch_size=64, clips_per_class=2)


def test_calibration_samples_are_int16_segments_from_the_middle():
    X = np.zeros((10, 2048, 1), np.float32)
    X[:, 1024:1124, 0] = 123
    samples = kcd.calibration_samples(X, count=5, length=512)
    assert samples.dtype == np.int16 and samples.shape == (5, 512, 1)
    assert samples[:, :100].min() == 123 and samples[:, 100:].max() == 0


# ----------------------------------------------------------------------- config

def test_config_defaults_and_errors(tmp_path):
    pytest.importorskip('yaml')
    path = tmp_path / 'c.yaml'
    path.write_text('keywords: [a, b]\nepochs: 3\n')
    config = kcd.load_config(path)
    assert config['epochs'] == 3 and config['batch_size'] == 128 and config['silence'] is True
    for text, message in (('epochs: 3\n', 'keywords'), ('keywords: [a]\nepoch: 3\n', 'unknown setting'),
                          ('keywords: [a, a]\n', 'duplicates'),
                          ('keywords: [yes, no]\n', 'quote them'),      # YAML reads these as booleans
                          ('keywords: [up, 1]\n', 'quote them'), ('keywords: [silence]\n', "can't be keywords"),
                          ('keywords: [a]\nvalidation_fraction: 0.6\ntest_fraction: 0.4\n', 'below 1')):
        path.write_text(text)
        with pytest.raises(ValueError, match=message):
            kcd.load_config(path)


def test_shipped_config_is_valid_and_its_augmentations_build():
    pytest.importorskip('yaml')
    config = kcd.load_config(KWS_DIR / 'kws_config.yaml')
    augment = kws_augment.build_pipeline(config['augmentations'])
    assert len(augment) == len(config['augmentations']) >= 4
    assert len(augment(tone(), np.random.default_rng(0))) == N
