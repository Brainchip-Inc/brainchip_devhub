#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Structured (channel/filter) sparsity for Speech Commands (DS-CNN) via Network
Slimming (Liu et al. 2017, "Learning Efficient Convolutional Networks through
Network Slimming"): an L1 penalty on each targeted BatchNormalization layer's
gamma (per-channel scale) during training identifies which output channels of
the preceding SeparableConv2D contribute least, then those channels are
physically pruned and the surrounding layers resized to match.

Unlike activation-sparsity regularization (SPARSITY_EXPERIMENT.md), this axis
reduces the model's actual parameter count -- Akida 1 hardware only
accelerates on activation/event sparsity, not weight sparsity, so this is
about model footprint, not runtime latency (see JOINT_SPARSITY_EXPERIMENT.md).
Ported from ../vww/vww_structured_sparsity.py -- the implementation is
architecture-agnostic (it walks the model graph rather than hardcoding layer
names), so it applies unchanged to DS-CNN.

`speech_commands_model.py`'s `build_ds_cnn` is a strict feed-forward chain (no
residual/skip connections) confirmed via `model.summary()`, so pruning a
SeparableConv2D's output channels only ever affects the very next weighted
layer's input dimension. DS-CNN's final separable-conv block (the one with
`pooling='global_avg'`, feeding the classifier head via a Reshape+Conv2D or
Dense) is excluded from the prunable set for the same reason VWW/PlantVillage
exclude their own final feature layer -- see
`speech_commands_joint_sparsity_sweep.py`.
"""
import numpy as np
import tensorflow as tf
from tf_keras import Input
from tf_keras.layers import BatchNormalization, SeparableConv2D


def add_gamma_l1_loss(model, strength, layer_names):
    """Attach an L1 penalty on gamma to each named BatchNormalization layer.

    BatchNormalization layers built by akida_models.layer_blocks aren't given
    a `gamma_regularizer` at construction time, so this can't be set after the
    fact the way activity_regularizer is (that's checked dynamically on every
    call; gamma_regularizer is only wired up at weight-creation time). Instead,
    attach the penalty directly as a model-level loss, using the zero-argument
    callable form of add_loss -- the standard way to regularize a weight that
    isn't produced by the layer's own `call()` inputs.

    Args:
        model (tf_keras.Model): model to attach the penalty to (modified in place).
        strength (float): L1 strength.
        layer_names (list[str]): BatchNormalization layer names to regularize.
    """
    for name in layer_names:
        bn = model.get_layer(name)
        model.add_loss(lambda bn=bn: strength * tf.reduce_sum(tf.abs(bn.gamma)))


def _channel_keep_indices(gamma, target_fraction):
    """Indices (ascending) of the channels to KEEP, dropping the lowest-|gamma| ones."""
    n = gamma.shape[0]
    n_prune = min(int(round(target_fraction * n)), n - 1)  # always keep >= 1 channel
    order = np.argsort(np.abs(gamma))
    prune_idx = set(order[:n_prune].tolist())
    return np.array([i for i in range(n) if i not in prune_idx])


def prune_model(model, target_fraction, prune_layer_names):
    """Prune output channels of the named SeparableConv2D layers and rebuild.

    For each layer in `prune_layer_names`, ranks output channels by the paired
    BN layer's |gamma| and drops the bottom `target_fraction` (uniformly per
    layer). Walks the full model graph in order, physically slicing weights
    and rebuilding every layer so shapes stay consistent end to end -- the
    layer immediately downstream of a pruned SeparableConv2D has its INPUT
    channel dimension shrunk to match, even if that downstream layer itself
    isn't in `prune_layer_names`.

    Args:
        model (tf_keras.Model): trained float model to prune (not modified).
        target_fraction (float): fraction of output channels to drop, per
            targeted layer (0 disables pruning for that layer).
        prune_layer_names (list[str]): SeparableConv2D layer names whose
            output channels are eligible for pruning.

    Returns:
        tf_keras.Model: a new, smaller model with pruned weights copied in.
    """
    prune_set = set(prune_layer_names)
    new_input = Input(shape=model.input.shape[1:], dtype=model.input.dtype,
                       name=model.input.name.split(':')[0] + '_pruned')
    x = new_input
    current_keep_idx = None  # None = the current tensor's channels are unchanged from original

    for layer in model.layers[1:]:  # skip the original InputLayer
        if isinstance(layer, SeparableConv2D):
            depthwise_kernel, pointwise_kernel = layer.get_weights()[:2]
            bias = layer.get_weights()[2] if layer.use_bias else None

            if current_keep_idx is not None:
                depthwise_kernel = depthwise_kernel[:, :, current_keep_idx, :]
                pointwise_kernel = pointwise_kernel[:, :, current_keep_idx, :]

            out_keep_idx = None
            if layer.name in prune_set:
                bn = model.get_layer(layer.name + '/BN')
                gamma = bn.get_weights()[0]
                out_keep_idx = _channel_keep_indices(gamma, target_fraction)
                pointwise_kernel = pointwise_kernel[:, :, :, out_keep_idx]
                if bias is not None:
                    bias = bias[out_keep_idx]

            new_cfg = dict(layer.get_config())
            new_cfg['filters'] = pointwise_kernel.shape[-1]
            new_layer = SeparableConv2D.from_config(new_cfg)
            x = new_layer(x)
            new_weights = [depthwise_kernel, pointwise_kernel]
            if bias is not None:
                new_weights.append(bias)
            new_layer.set_weights(new_weights)

            current_keep_idx = out_keep_idx  # None if this layer wasn't pruned
            continue

        if isinstance(layer, BatchNormalization):
            new_layer = BatchNormalization.from_config(layer.get_config())
            x = new_layer(x)
            weights = layer.get_weights()
            if current_keep_idx is not None:
                weights = [w[current_keep_idx] for w in weights]
            new_layer.set_weights(weights)
            continue

        # Everything else (stem Conv2D, ReLU, Dropout, pooling, input scaling,
        # the Reshape/Conv2D or Dense classifier head, ...) is channel-count-
        # preserving as long as no upstream pruning is pending (current_keep_idx
        # is None) by the time we reach it -- true here since every prune-
        # eligible layer's downstream neighbor consumes the pending index
        # immediately (the next SeparableConv2D or BN handles it above), and
        # the final (GAP) separable-conv block, never itself pruned, resets
        # current_keep_idx to None before the classifier head is reached.
        new_layer = layer.__class__.from_config(layer.get_config())
        x = new_layer(x)
        if layer.get_weights():
            new_layer.set_weights(layer.get_weights())

    from tf_keras import Model
    return Model(new_input, x, name=model.name + '_pruned')
