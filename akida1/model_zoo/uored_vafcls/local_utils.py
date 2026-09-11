from cnn2snn import quantize
from tf_keras import Model


def quantize_until(model, target_layer):
    '''Quantize a model up to the named target layer only.

    Layers up to and including `target_layer` are quantized; everything
    after is left as-is and re-attached to the quantized output. Useful for
    bisecting which layer introduces a quantization accuracy drop.

    Note: assumes a linear (single input/output per layer) topology from
    `target_layer` onwards, since remaining layers are re-called in sequence.
    '''
    layer_names = [layer.name for layer in model.layers]
    target_idx = layer_names.index(target_layer)

    # Get sub-model up to and including named layer
    sub_model = Model(inputs=model.input, outputs=model.layers[target_idx].output)
    # Quantize that sub-model
    qsub_model = quantize(sub_model, weight_quantization=4, activ_quantization=4, input_weight_quantization=8)

    # Stitch remaining unquantized layers to the end of that
    x = qsub_model.output
    for layer in model.layers[target_idx + 1:]:
        x = layer(x)

    # Return complete partially-quantized model
    pq_model = Model(inputs=qsub_model.input, outputs=x, name='quantized_until_' + target_layer)
    return pq_model