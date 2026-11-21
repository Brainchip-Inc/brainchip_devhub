#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Model definition and Pico conversion steps for the 12-class Speech Commands KWS model.

The architecture is ``akida_models.tenn_recurrent_sc12``: a stack of six *kernelized*
state-space (SSM) blocks on raw 16 kHz audio. Training uses the kernelized form, which
processes a whole 16,384-sample clip at once. Pico runs the equivalent *stateful* form, one
256-sample chunk at a time, carrying the SSM state between chunks. The pipeline is:

    kernelized float  ->  stateful float  ->  quantized (int8 w/a, int16 input)
                      ->  Akida model  ->  mapped on Pico (one TNP_R)

The stateful conversion is exact, so it costs no accuracy. Quantization is post-training
(no QAT) and needs calibration samples.

Usage:
    python kws_model.py [-s OUTPUT_PATH]        # build an untrained float model
"""
import argparse

import akida
import cnn2snn
import quantizeml
import akida_models
from akida_models.tenn_recurrent.convert_recurrent import convert_to_stateful
from quantizeml.models import QuantizationParams

from kws_data import CHUNK_SIZE, NUM_CLASSES, SAMPLE_LENGTH


def build_kws_model(num_classes=NUM_CLASSES):
    """Untrained kernelized model: input ``(batch, 16384, 1)`` int16, ``num_classes`` logits."""
    return akida_models.tenn_recurrent_sc12(input_shape=(SAMPLE_LENGTH, 1), num_classes=num_classes)


def load_kws_model(weights_path):
    """Kernelized float model with the given weights loaded."""
    model = build_kws_model()
    model.load_weights(weights_path)
    return model


def to_stateful(model, chunk_size=CHUNK_SIZE):
    """Rewrite the kernelized model to consume ``chunk_size`` samples per call.

    Input becomes ``(1, 256, 1)`` and output ``(1, 1, num_classes)``. 256 is the smallest
    chunk that completes an output step (the model downsamples time by 256x).
    """
    return convert_to_stateful(model, chunk_size)


def quantize_kws_model(stateful_model, samples):
    """Post-training quantization: 8-bit weights and activations, 16-bit input.

    The input stays 16-bit because the model reads raw audio samples; 8 bits would discard
    most of the signal's dynamic range.

    Args:
        stateful_model: output of :func:`to_stateful`.
        samples (np.ndarray): calibration segments, ``(100, 1024, 1)`` int16.
    """
    return quantizeml.models.quantize(
        stateful_model,
        qparams=QuantizationParams(activation_bits=8, weight_bits=8, input_dtype='int16'),
        batch_size=100, epochs=1, samples=samples)


def to_akida(quantized_model, device=None):
    """Convert a quantized model to Akida and, optionally, map it on ``device``.

    With no device the model is returned unmapped. ``device`` can be a Pico FPGA from
    ``akida.devices()`` or a software ``akida.PicoIP()``.
    """
    model_akida = cnn2snn.convert(quantized_model)
    if device is not None:
        model_akida.map(device, hw_only=True)
    return model_akida


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the 12-class Pico KWS model')
    parser.add_argument('-s', '--savemodel', default=None, help='Where to save the untrained model (.h5)')
    args = parser.parse_args()

    model = build_kws_model()
    model.summary()
    if args.savemodel:
        model.save(args.savemodel, include_optimizer=False)
        print(f'Saved {args.savemodel}')
