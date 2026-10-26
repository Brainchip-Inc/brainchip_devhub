#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Evaluation of the Jester TENN model on Akida 2.

Evaluates one of the three variants (see ``jester_model.py``):

  float      the Conv3D Keras model, run on whole 16-frame clips
  quantized  the 8-bit buffered Keras model, streamed one frame at a time
  akida      the converted Akida model, streamed one frame at a time on the
             software backend

Each variant produces one prediction per frame, and two accuracies are reported
from them:

* **Clip accuracy** (the headline figure): one decision per clip, from the
  per-frame softmax outputs accumulated over the clip with a decay of 0.8, so
  that recent frames count most. This is how BrainChip's
  ``evaluate_bufferized_model`` scores the quantized and Akida models.
* **Frame accuracy**: every frame's own prediction, scored against the clip's
  label, as the ``TemporalAccuracy`` metric BrainChip trains with does. It is
  lower because the first frames of a clip have seen little of the gesture yet.

Streaming means the buffers carry state from frame to frame, so they are reset
at the start of every clip. For the Akida variant it also measures the mean
activation sparsity, which needs no hardware.

Examples
--------
    # BrainChip Jester subset (downloaded into data/ on first use)
    python jester_eval.py --variant akida

    # Quick check on the first 32 clips
    python jester_eval.py --variant quantized -n 32
"""
import argparse
import json
import pathlib

import numpy as np
import tensorflow as tf
from scipy.special import softmax
from tqdm import tqdm

import akida
from akida_models.model_io import load_model
from akida_models.sparsity import compute_sparsity
from quantizeml.models import reset_buffers

from jester_data import get_data, get_data_dir, get_clips
from jester_model import VARIANTS, model_path, verify_model

# Weight of the accumulated past predictions at each new frame (evaluate_bufferized_model)
DECAY = 0.8
# Clips used to measure activation sparsity: the first clip of each class
SPARSITY_CLIPS = 27


def score(frame_logits, labels):
    """Scores the per-frame predictions of a batch of clips.

    Args:
        frame_logits (np.ndarray): (B, T, C) logits, one row per frame, in order.
        labels (np.ndarray): (B,) integer labels.

    Returns:
        tuple: (correct clips, correct frames), for B clips and B * T frames.
    """
    accumulated = np.zeros((frame_logits.shape[0], frame_logits.shape[2]))
    for t in range(frame_logits.shape[1]):
        accumulated = accumulated * DECAY + softmax(frame_logits[:, t], axis=-1)
    clips = int(np.sum(accumulated.argmax(-1) == labels))
    frames = int(np.sum(frame_logits.argmax(-1) == labels[:, None]))
    return clips, frames


def float_frame_logits(model):
    """Per-frame logits of the float model: it sees the whole clip at once."""
    model_fn = tf.function(model)
    return lambda clips: model_fn(clips).numpy().reshape(clips.shape[0], clips.shape[1], -1)


def quantized_frame_logits(model):
    """Per-frame logits of the buffered Keras model, streaming each clip frame by frame."""
    model_fn = tf.function(model)

    def run(clips):
        reset_buffers(model)  # a new batch of clips: forget the previous frames
        return np.stack([model_fn(clips[:, t]).numpy() for t in range(clips.shape[1])], 1)
    return run


def akida_frame_logits(model):
    """Per-frame logits of the Akida model, streaming each clip frame by frame.

    An akida.Model has no buffer reset: a new model built from the same layers
    starts with empty buffers, so one is built per clip. On a buffered model,
    Akida reads the batch axis as time: forwarding a clip's 16 frames as one
    batch is bit-identical to forwarding them one at a time.
    """
    def run(clips):
        return np.stack([akida.Model(layers=model.layers).forward(clip).reshape(len(clip), -1)
                         for clip in clips])
    return run


def evaluate(frame_logits_fn, dataset, num_clips=None, desc='Evaluating'):
    """Runs a variant over the dataset.

    Returns:
        dict: ``clip_acc``, ``frame_acc`` and ``n`` (clips evaluated).
    """
    correct_clips = correct_frames = clips_seen = frames_seen = 0
    for clips, labels in tqdm(dataset, desc=desc):
        clips, labels = clips.numpy(), labels.numpy()
        if num_clips is not None:
            clips, labels = clips[:num_clips - clips_seen], labels[:num_clips - clips_seen]
        c, f = score(frame_logits_fn(clips), labels)
        correct_clips += c
        correct_frames += f
        clips_seen += len(labels)
        frames_seen += clips.shape[0] * clips.shape[1]
        if num_clips is not None and clips_seen >= num_clips:
            break
    return {'clip_acc': correct_clips / clips_seen, 'frame_acc': correct_frames / frames_seen,
            'n': clips_seen}


def akida_sparsity(model, data_dir, num_clips=SPARSITY_CLIPS):
    """Mean activation sparsity of the Akida model, streaming real clips.

    ``compute_sparsity`` builds fresh sub-models, so each clip starts with empty
    buffers. Returns the per-layer sparsity averaged over clips, and its mean.
    """
    clips, _ = get_clips(data_dir, num_clips, one_per_class=True)
    per_clip = [compute_sparsity(model, samples=clip, batch_size=len(clip)) for clip in clips]
    per_layer = {name: float(np.mean([s[name] for s in per_clip])) for name in per_clip[0]}
    return per_layer, float(np.mean(list(per_layer.values())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the Jester TENN model for Akida 2')
    parser.add_argument('--variant', default='akida', choices=VARIANTS,
                        help='Model variant to evaluate. Defaults to %(default)s.')
    parser.add_argument('-m', '--models-dir', default=None,
                        help='Directory holding the models. Defaults to pretrained_models/.')
    parser.add_argument('-d', '--data', default=None,
                        help='Jester directory. Defaults to the BrainChip Jester subset, '
                             'downloaded into data/.')
    parser.add_argument('-b', '--batch-size', type=int, default=32,
                        help='Clips per batch (Keras variants). Defaults to %(default)s.')
    parser.add_argument('-n', '--num-clips', type=int, default=None,
                        help='Evaluate only the first N clips')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Write accuracy (and sparsity/params) to docs/metrics.json')
    args = parser.parse_args()

    if args.save_metrics and args.num_clips is not None:
        raise SystemExit('--save-metrics records whole-dataset runs only: drop -n.')

    path = model_path(args.variant, args.models_dir)
    if args.models_dir is None:
        verify_model(path)
    print(f'Evaluating {path.name}')

    if args.variant == 'akida':
        model = akida.Model(str(path))
        frame_logits_fn = akida_frame_logits(model)
        batch_size = 1
    else:
        model = load_model(str(path))
        frame_logits_fn = (float_frame_logits if args.variant == 'float'
                           else quantized_frame_logits)(model)
        batch_size = args.batch_size

    data_dir = get_data_dir(args.data)
    dataset, num_clips = get_data(data_dir, batch_size)
    print(f'{data_dir}: {num_clips} clips')
    result = evaluate(frame_logits_fn, dataset, args.num_clips, desc=f'Evaluating {args.variant}')
    print(f'\n{args.variant} over {result["n"]} clips: clip accuracy '
          f'{result["clip_acc"] * 100:.2f}%, frame accuracy {result["frame_acc"] * 100:.2f}%')

    sparsity = None
    if args.variant == 'akida':
        _, sparsity = akida_sparsity(model, data_dir)
        print(f'Mean activation sparsity ({SPARSITY_CLIPS} clips): {sparsity * 100:.2f}%')

    if args.save_metrics:
        # Used to update the stored metrics behind the README tables. This is a
        # maintenance step, run when the models or the pipeline change.
        metrics_path = pathlib.Path(__file__).parent / 'docs' / 'metrics.json'
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        # Keys follow the other Akida 2 examples: float_*, then w8a8_quant_* and
        # w8a8_akida_* for the 8-bit model
        prefix = {'float': 'float', 'quantized': 'w8a8_quant', 'akida': 'w8a8_akida'}[args.variant]
        metrics[f'{prefix}_clip_acc'] = f'{result["clip_acc"] * 100:.2f}%'
        metrics[f'{prefix}_frame_acc'] = f'{result["frame_acc"] * 100:.2f}%'
        metrics[f'{prefix}_n'] = str(result['n'])
        if sparsity is not None:
            metrics['w8a8_sparsity'] = f'{sparsity * 100:.2f}%'
        if args.variant == 'float':
            metrics['params'] = f'{model.count_params():,}'
        metrics_path.write_text(json.dumps(metrics, indent=4, sort_keys=True) + '\n')
        print(f'Metrics saved to {metrics_path}')
