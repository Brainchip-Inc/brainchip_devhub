#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Create a YOLOv2 object detection model for the PASCAL VOC dataset.
This model targets the Akida 1 platform, and is based on the
AkidaNet architecture (width alpha=0.5), with weights pre-trained on
ImageNet loaded into the backbone and a YOLOv2 detection head added on top.

The model is built at 224x224 RGB input resolution and includes input
scaling as part of the model (a Rescaling layer, dividing by 128) - thus
the preprocessing pipeline should NOT include any normalization of the
data, but rather deliver inputs in the uint8 range.

The AkidaNet backbone is instantiated directly via
`akidanet_imagenet_pretrained`, which returns the full ImageNet classifier
(pooling + dense classifier included). The YOLOv2 head is attached to the
backbone's last spatial (7x7) feature map - i.e. the output of
`separable_13`, from before that pooling - reusing the pretrained model's
own input and layers (and therefore their trained weights) directly by
reference, rather than rebuilding an untrained backbone and loading
weights with `load_weights(..., by_name=True)`, which would silently skip
any layer whose name failed to line up between the two models.

Usage:
    python detection_model.py [-s OUTPUT_PATH]
"""

import argparse

import numpy as np
from tf_keras import Model
from tf_keras.layers import BatchNormalization, ReLU
from tf_keras.utils import set_random_seed

from akida_models.imagenet.model_akidanet import akidanet_imagenet_pretrained
from akida_models.layer_blocks import separable_conv_block
from cnn2snn import set_akida_version, AkidaVersion


def yolo_head_block(x, num_boxes, classes, filters=1024):
    """Adds the `YOLOv2 detection head <https://arxiv.org/pdf/1612.08242.pdf>`_, at the output
    of a model.

    Args:
        x (:obj:`tf.Tensor`): input tensor of shape `(rows, cols, channels)`.
        num_boxes (int): number of boxes.
        classes (int): number of classes.
        filters (int, optional): number of filters in hidden layers. Defaults to 1024.

    Returns:
        :obj:`tf.Tensor`: output tensor of yolo detection head block.

    Notes:
        This block replaces conv layers by separable_conv, to decrease the amount of parameters.
    """
    # Model version management
    # Targeting Akida 1
    fused = True
    relu_activation = 'ReLU6.0'
    x = separable_conv_block(x, filters=filters, name='1conv',
                             kernel_size=(3, 3), padding='same', use_bias=False,
                             relu_activation=relu_activation, add_batchnorm=True, fused=fused)
    x = separable_conv_block(x, filters=filters, name='2conv',
                             kernel_size=(3, 3), padding='same', use_bias=False,
                             relu_activation=relu_activation, add_batchnorm=True, fused=fused)
    x = separable_conv_block(x, filters=filters, name='3conv',
                             kernel_size=(3, 3), padding='same', use_bias=False,
                             relu_activation=relu_activation, add_batchnorm=True, fused=fused)
    x = separable_conv_block(x, filters=(num_boxes * (4 + 1 + classes)), name='detection_layer',
                             kernel_size=(3, 3), padding='same', use_bias=True,
                             relu_activation=False, add_batchnorm=False, fused=fused)
    return x


def _reset_detection_layer_weights(model):
    """Re-initializes the YOLO detection_layer with small, zero-centered weights.

    The detection head's preceding conv layers keep tf_keras' default random
    initialization (to be trained later); only the final detection_layer is
    deliberately scaled down so early training isn't dominated by large
    box/class predictions.
    """
    detection_layers = [layer for layer in model.layers if "detection_layer" in layer.name]
    assert len(detection_layers) == 1, "Expected a single (fused) detection_layer for Akida 1."
    layer = detection_layers[0]

    dw_shape, pw_shape, bias_shape = (w.shape for w in layer.get_weights())
    grid_h, grid_w = model.output_shape[1:3]
    scale = grid_h * grid_w

    mu, sigma = 0, 0.1
    layer.set_weights([
        np.random.normal(mu, sigma, size=dw_shape) / scale,
        np.random.normal(mu, sigma, size=pw_shape) / scale,
        np.random.normal(mu, sigma, size=bias_shape) / scale,
    ])


def build_detection_model(anchors=5, classes=20, seed=42, alpha=0.5, head_filters=1024):
    set_random_seed(seed)

    with set_akida_version(AkidaVersion.v1):
        base_model = akidanet_imagenet_pretrained(alpha=alpha, quantized=False)

        # Take the backbone's last spatial (7x7) feature map - the raw
        # separable_13 output, from before the global-average-pooling baked
        # into `base_model` for ImageNet classification.
        x = base_model.get_layer('separable_13').output
        # Include batch norm and ReLU for the final layer of the backbone
        x = BatchNormalization(name='separable_13/BN')(x)
        x = ReLU(max_value=6.0, name='separable_13/relu')(x)

        # Add the YOLOv2 detection head on top of the backbone features
        x = yolo_head_block(x, num_boxes=anchors, classes=classes, filters=head_filters)
        model = Model(base_model.input, x, name='yolo_akidanet_detection')

        _reset_detection_layer_weights(model)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Build the YOLOv2-AkidaNet detection model for Akida 1')
    parser.add_argument("-s",
                        "--savepath",
                        type=str,
                        default='./models/yolo_akidanet_detection.h5',
                        help="Save model with the specified path + name")
    parser.add_argument("-c",
                        "--classes", type=int, default=20,
                            help="Number of classes (default 20)"),
    parser.add_argument("--anchors", type=int, default=5,
                            help="Number of anchors (default 5)"),
    parser.add_argument("--alpha", type=float, default=0.5,
                                help="Width multiplier for Akidanet backbone (default 0.5)"),
    parser.add_argument("--head_filters", type=int, default=1024,
                                help="Number of filters in head layers (default 1024)"),
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    model = build_detection_model(anchors=args.anchors, classes=args.classes, seed=args.seed, alpha=args.alpha, head_filters=args.head_filters)
    model.summary()
    model.save(args.savepath, include_optimizer=False)
    print(f'Model saved to {args.savepath}')
