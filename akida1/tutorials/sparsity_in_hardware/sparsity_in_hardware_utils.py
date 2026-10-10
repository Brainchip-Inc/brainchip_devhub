# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Helpers for the "Sparsity in Akida hardware" tutorial (Akida 1).

Two kinds of measurement are built from these:

- Hand-built models whose last layer is the layer under test, fed inputs with an event
  density you choose. They show how one layer's processing time and energy depend on the
  number of events it receives, with everything else held fixed.
- Sub-models cut from a trained model zoo model and fed real samples. They show the same
  dependence on the layers of a real network.

Processing time is read from the chip's own clock counter
(`model.metrics['inference_clk']`). A layer's share is isolated by layer subtraction: the
clock count of a model ending with that layer, minus that of the same model without it.
"""
import contextlib
import io

import akida
import numpy as np

from brainchip_utils.hardware_utils import (full_model_benchmark, remove_final_maxpool,
                                            silence_output_layer)


# -----------------------------------------------------------------------------
# Hand-built single-layer models
# -----------------------------------------------------------------------------

def _set_identity_weights(layer):
    """Make a layer pass its input events through unchanged.

    The kernel has a single non-zero tap, at its centre. For the input layer, that copies
    the single input channel to every output channel; for a separable layer, the
    depthwise and pointwise parts are both identities.
    """
    weights = layer.get_variable('weights')
    new_weights = np.zeros(weights.shape, dtype=np.int8)
    centre = weights.shape[0] // 2
    if layer.parameters.layer_type == akida.LayerType.InputConvolutional:
        new_weights[centre, centre, :, :] = 1
    else:
        new_weights[centre, centre, :, 0] = 1
        pw = np.zeros(layer.get_variable('weights_pw').shape, dtype=np.int8)
        channels = np.arange(pw.shape[2])
        pw[0, 0, channels, channels] = 1
        layer.set_variable('weights_pw', pw)
    layer.set_variable('weights', new_weights)


def _set_random_weights(layer, rng):
    """Fill a layer with random, non-zero-heavy 4-bit weights, like a trained layer's."""
    for name in ('weights', 'weights_pw'):
        if name in layer.get_variable_names():
            shape = layer.get_variable(name).shape
            values = np.clip(np.round(rng.normal(scale=2.0, size=shape)), -7, 7)
            layer.set_variable(name, values.astype(np.int8))


def build_single_layer_model(height=32, width=32, channels=32, filters=32, kernel_size=3,
                             layer_type='conv', seed=0):
    """Build an Akida 1 model whose last layer is the layer under test.

    The model takes a single-channel (height, width, 1) input and has three layers:

    1. the input layer, with identity kernels, which copies the input to `channels`
       channels. Every channel then carries the same pattern of events as the input;
    2. a separable layer, also with identity kernels. It is there so that the layer under
       test receives its events from a standard NP, as it would in a real model, rather
       than straight from the input layer;
    3. the layer under test: a `Convolutional` (`layer_type='conv'`) or
       `SeparableConvolutional` (`'sepconv'`) layer with random weights.

    So the event density of the input is exactly the event density the layer under test
    receives.
    """
    model = akida.Model()
    model.add(akida.InputConvolutional((height, width, 1), [3, 3], channels,
                                       weights_bits=8, act_bits=4))
    model.add(akida.SeparableConvolutional([3, 3], channels, weights_bits=4, act_bits=4))
    for layer in model.layers:
        _set_identity_weights(layer)
    layer_class = {'conv': akida.Convolutional,
                   'sepconv': akida.SeparableConvolutional}[layer_type]
    model.add(layer_class([kernel_size, kernel_size], filters, weights_bits=4, act_bits=4))
    _set_random_weights(model.layers[-1], np.random.default_rng(seed))
    return model


def random_events(input_shape, density, num_samples=1, seed=None):
    """Return uint8 inputs in which a fraction `density` of the values, chosen at random,
    are 1 and the rest are 0."""
    rng = np.random.default_rng(seed)
    size = int(np.prod(input_shape))
    inputs = np.zeros((num_samples, size), dtype=np.uint8)
    for sample in inputs:
        sample[rng.choice(size, size=round(density * size), replace=False)] = 1
    return inputs.reshape((num_samples, *input_shape))


# -----------------------------------------------------------------------------
# Measurement
# -----------------------------------------------------------------------------

def sub_model(model, num_layers, silence=True):
    """Return a new model made of the first `num_layers` layers of `model`.

    With `silence=True` the last layer is set so that it never fires. It still does all
    its processing, but sends no events off the chip, so the copy back to the host
    doesn't count in the clocks.
    """
    cut = akida.Model(layers=model.layers[:num_layers])
    if silence:
        cut = remove_final_maxpool(silence_output_layer(cut))
    return cut


def clock_counts(model, inputs):
    """Run each input through a mapped model, one at a time, and return the on-chip clock
    count of each inference."""
    model.forward(inputs[:1])  # a priming inference, not counted
    clocks = np.empty(len(inputs))
    for ii in range(len(inputs)):
        model.forward(inputs[ii:ii + 1])
        clocks[ii] = model.metrics['inference_clk']
    return clocks


def layer_clocks(model, device, inputs, layer_index):
    """Measure one layer by layer subtraction.

    Returns the clock counts per input of the model up to and including layer
    `layer_index`, and of the model up to the layer before it. Their difference is the
    layer's own processing time.
    """
    results = []
    for num_layers in (layer_index + 1, layer_index):
        cut = sub_model(model, num_layers)
        cut.map(device, mode=akida.MapMode.Minimal, hw_only=True)
        results.append(clock_counts(cut, inputs))
    return results[0], results[1]


def benchmark(model, device, inputs, repeats=5):
    """Run `full_model_benchmark` from brainchip_utils without its progress output.

    Returns its results dict: mean clocks per inference, latency and, on an AKD1500 with
    power sensing, a 'power' dict (None otherwise).
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return full_model_benchmark(model, device, inputs, repeats=repeats)


def layer_energy(model, device, inputs, layer_index, repeats=5):
    """Measure one layer's dynamic energy per inference by layer subtraction (AKD1500
    with power sensing only).

    Returns the dynamic energy (mJ per inference) of the model up to and including layer
    `layer_index`, and of the model up to the layer before it, or None if power can't be
    measured.
    """
    results = []
    for num_layers in (layer_index + 1, layer_index):
        res = benchmark(sub_model(model, num_layers), device, inputs, repeats=repeats)
        if res is None or res['power'] is None:
            return None
        results.append(res['power']['avg_dynamic_energy_mj'])
    return results[0], results[1]


def input_events(model, samples, layer_index):
    """Count the events (non-zero values) arriving at layer `layer_index` for each sample.

    The counts come from the software backend, which gives the same outputs as the
    hardware.
    """
    outputs = akida.Model(layers=model.layers[:layer_index]).forward(samples)
    return np.count_nonzero(outputs.reshape(len(samples), -1), axis=1)


def layer_nps(model):
    """Return the number of NPs each layer of a mapped model uses. The input layer counts
    as one: its dedicated unit."""
    return np.array([len(layer.mapping.nps) if layer.mapping is not None else 0
                     for layer in model.layers])


def fit_line(x, y):
    """Least-squares fit of y = slope * x + intercept. Returns (slope, intercept)."""
    slope, intercept = np.polyfit(np.ravel(x), np.ravel(y), 1)
    return slope, intercept


def shift_thresholds(model, steps):
    """Return a copy of an Akida 1 model with every layer's firing threshold moved.

    Each threshold moves by `steps` activation steps of its own neuron: positive values
    make neurons fire less often (sparser activations), negative values more often. With
    `steps=None` every threshold is set so low that every neuron always fires, giving
    fully dense activations. The model's accuracy suffers either way: this is a tool for
    seeing how the hardware responds to activity, not a way to make a model sparser.
    """
    shifted = akida.Model.from_dict(model.to_dict())
    for layer in shifted.layers:
        if 'threshold' not in layer.get_variable_names():
            continue
        threshold = layer.get_variable('threshold')
        if steps is None:
            new_threshold = np.full_like(threshold, -2**19)
        else:
            new_threshold = threshold + np.round(steps * layer.get_variable('act_step'))
        layer.set_variable('threshold', new_threshold.astype(threshold.dtype))
    return shifted


def activation_density(model, samples):
    """Mean fraction of non-zero values arriving at each layer after the input layer, for
    the given samples (computed with the software backend)."""
    densities = [input_events(model, samples, kk).mean() / np.prod(model.layers[kk].input_dims)
                 for kk in range(1, len(model.layers))]
    return np.array(densities)
