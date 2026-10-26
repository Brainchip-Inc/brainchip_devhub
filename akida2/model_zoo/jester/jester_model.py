#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Model files, quantization and the reproduction check for the Jester TENN example.

**The models.** BrainChip's published TENN spatiotemporal model for Jester, in
three variants, stored in ``pretrained_models/`` (Git LFS):

  float      ``tenn_spatiotemporal_jester.h5``: the trained Keras model, built
             from ``Conv3D`` layers. It takes a whole clip, (16, 100, 100, 3),
             and returns one prediction per frame, (16, 1, 1, 27).
  quantized  ``tenn_spatiotemporal_jester_buffer_i8_w8_a8.h5``: the 8-bit
             model. Quantization replaces each temporal ``Conv3D`` with a
             ``BufferTempConv`` (or ``DepthwiseBufferTempConv``), which keeps the
             last 5 frames' inputs in a FIFO buffer, and each spatial ``Conv3D``
             with a 2D convolution. The model takes one frame, (100, 100, 3), and
             returns one prediction, (27,).
  akida      ``tenn_spatiotemporal_jester_buffer_i8_w8_a8.fbz``: the 8-bit model
             converted by ``cnn2snn convert``, which names its output after its
             input.

The files are the ones ``akida_models.tenn_spatiotemporal_jester_pretrained``
downloads, checked against our own sha256 pins (:data:`SHA256`). akida_models
1.14.0 pins a stale hash for the float file and ``fetch_file`` ignores hash
mismatches, so its own check does not hold.

**Quantization.** :func:`quantize_float_model` repeats BrainChip's quantization
step, the ``quantizeml quantize`` call in ``train_spatiotemporal_jester.sh``, with
the same parameters and calibration pack. :func:`compare_models` checks the result
against the published 8-bit model. Run as a script::

    python jester_model.py --compare models/tenn_spatiotemporal_jester_buffer_i8_w8_a8.h5
"""
import argparse
import hashlib
import pathlib

import numpy as np
from cnn2snn import convert
from quantizeml.models import QuantizationParams, quantize

__all__ = ["VARIANTS", "PRETRAINED_DIR", "SHA256", "model_path", "verify_model",
           "quantization_params", "quantize_float_model", "compare_models"]

VARIANTS = ("float", "quantized", "akida")
PRETRAINED_DIR = pathlib.Path(__file__).parent / 'pretrained_models'

_FILENAMES = {
    'float': 'tenn_spatiotemporal_jester.h5',
    'quantized': 'tenn_spatiotemporal_jester_buffer_i8_w8_a8.h5',
    'akida': 'tenn_spatiotemporal_jester_buffer_i8_w8_a8.fbz',
}

# sha256 of the published files (float and quantized as served by data.brainchip.com,
# akida as converted from the quantized model with cnn2snn 2.19.1)
SHA256 = {
    'tenn_spatiotemporal_jester.h5':
        '5950e4d188cbda893c19c47d0a6e9e5c6200f902bb72f64d155d3e7a6f85816f',
    'tenn_spatiotemporal_jester_buffer_i8_w8_a8.h5':
        '388f0e312499190ed2cc839fb4c4e6c791e54e5ee4d35736f33fc36acae24015',
    'tenn_spatiotemporal_jester_buffer_i8_w8_a8.fbz':
        '058e6c8de86330f564730e62b4cf88813926becabfc5fef8abacdeff85972469',
}


def model_path(variant='quantized', models_dir=None):
    """Returns the path of a model file.

    Args:
        variant (str, optional): 'float', 'quantized' or 'akida'. Defaults to
            'quantized'.
        models_dir (str, optional): directory holding the models. Defaults to
            this example's ``pretrained_models/``.

    Returns:
        pathlib.Path: path to the model file.
    """
    if variant not in VARIANTS:
        raise ValueError(f'variant must be one of {VARIANTS}, received {variant}')
    directory = pathlib.Path(models_dir) if models_dir else PRETRAINED_DIR
    return directory / _FILENAMES[variant]


def verify_model(path):
    """Checks a published model file against its sha256 pin.

    Raises:
        FileNotFoundError: if the file is missing.
        ValueError: if its hash differs, e.g. a Git LFS pointer that was never pulled.
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f'{path} not found. These models are tracked with Git LFS - '
                                'run `git lfs pull` to fetch them.')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256[path.name]:
        raise ValueError(f'{path.name}: sha256 {digest} does not match the published file. '
                         'If it is a Git LFS pointer, run `git lfs pull`.')


def quantization_params():
    """The QuantizationParams of ``quantizeml quantize -w 8 -i 8 -a 8
    --per_tensor_activations -id uint8``: 8-bit weights (input layer included)
    and activations, one scale per activation tensor rather than per channel,
    and uint8 input. The other defaults (8-bit outputs, 32-bit buffers) match the
    CLI's.
    """
    return QuantizationParams(input_weight_bits=8, weight_bits=8, activation_bits=8,
                              per_tensor_activations=True, input_dtype='uint8')


def quantize_float_model(model, samples, num_clips=100, epochs=1):
    """Quantizes the float model as BrainChip did, with ``-e 1 -bs 100``.

    The calibration samples are frames in clip order, (num_clips * 16, 100, 100, 3).
    For a spatiotemporal TENN, ``batch_size`` must be the number of clips: the
    calibration uses it to cut the frames back into temporally coherent clips.

    Returns:
        keras.Model: the quantized, buffered model, taking one frame at a time.
    """
    return quantize(model, qparams=quantization_params(), samples=samples,
                    num_samples=len(samples), batch_size=num_clips, epochs=epochs)


def compare_models(reference, candidate):
    """Compares two quantized models, structurally and as converted Akida models.

    Returns:
        dict: ``same_structure`` (layer names, types and output shapes),
        ``keras_diffs`` (Keras variables that differ, with their max absolute
        difference) and ``akida_diffs`` (Akida layer variables that differ after
        conversion: the integer weights and thresholds that run on the NPs, and
        the float output scales).
    """
    def signature(m):
        return [(layer.name, type(layer).__name__, str(layer.output_shape))
                for layer in m.layers]

    ref_vars = {v.name: v.numpy() for v in reference.variables}
    cand_vars = {v.name: v.numpy() for v in candidate.variables}
    keras_diffs = {k: float(np.abs(ref_vars[k] - cand_vars[k]).max())
                   for k in ref_vars.keys() & cand_vars.keys()
                   if not np.array_equal(ref_vars[k], cand_vars[k])}

    akida_diffs = {}
    for ref_layer, cand_layer in zip(convert(reference).layers, convert(candidate).layers):
        for name in ref_layer.variables.names:
            a, b = ref_layer.variables[name], cand_layer.variables[name]
            if a.shape != b.shape or not np.array_equal(a, b):
                akida_diffs[f'{ref_layer.name}/{name}'] = (
                    str(a.dtype), float(np.abs(a.astype(np.float64) - b).max()))

    return {'same_structure': (signature(reference) == signature(candidate)
                               and ref_vars.keys() == cand_vars.keys()),
            'keras_diffs': keras_diffs,
            'akida_diffs': akida_diffs}


def print_comparison(result):
    """Prints a :func:`compare_models` result."""
    print(f'Same structure (layers, shapes, variables): {result["same_structure"]}')
    print(f'Keras variables that differ: {len(result["keras_diffs"])}')
    for name, diff in sorted(result['keras_diffs'].items()):
        print(f'  {name:60} max |diff| {diff:.3g}')
    print(f'Akida variables that differ after conversion: {len(result["akida_diffs"])}')
    for name, (dtype, diff) in sorted(result['akida_diffs'].items()):
        print(f'  {name:60} {dtype:8} max |diff| {diff:.3g}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Check a quantized Jester model against the published one')
    parser.add_argument('--compare', required=True,
                        help='Quantized model to compare, e.g. one written by jester_eval.sh')
    args = parser.parse_args()

    from akida_models.model_io import load_model

    published = model_path('quantized')
    verify_model(published)
    print(f'Comparing {args.compare} with the published {published.name}')
    print_comparison(compare_models(load_model(str(published)), load_model(args.compare)))
