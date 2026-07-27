#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Create a YOLOv2 object detection model for the PASCAL VOC ('car' / 'person')
subset. This model targets the Akida 1 platform, and is based on the
AkidaNet architecture (width alpha=0.5), with weights pre-trained on
ImageNet loaded into the backbone and a YOLOv2 detection head added on top.

The model is built at 224x224 RGB input resolution and includes input
scaling as part of the model (a Rescaling layer) - thus the preprocessing
pipeline should NOT include any normalization of the data, but rather
deliver inputs in the uint8 range.

Only the backbone layer names match between the ImageNet-pretrained
weights file and this model, so `load_weights(..., by_name=True)` loads
the backbone only; the newly-added detection head keeps the small,
zero-centered random initialization set up by `yolo_base`.

Usage:
    python detection_model.py [-s OUTPUT_PATH]
"""

import argparse

from tf_keras.utils import set_random_seed
from akida_models.detection.model_yolo import yolo_base
from akida_models.utils import fetch_file
from cnn2snn import set_akida_version, AkidaVersion


def build_detection_model(anchors=5, classes=20, seed=42):
    set_random_seed(seed)

    # Use alpha=0.5 width multiplier for the akidanet backbone
    alpha=0.5
    BACKBONE_URL = 'https://data.brainchip.com/models/AkidaV1/akidanet/akidanet_imagenet_224_alpha_50.h5'
    BACKBONE_HASH = '61f2883a6b798f922a5c0411296219a85f25581d7571f65546557b46066f058f'
    fname = 'akidanet_imagenet_224_alpha_50.h5'

    with set_akida_version(AkidaVersion.v1):
        model = yolo_base(input_shape=(224, 224, 3),
                          classes=classes,
                          nb_box=anchors,
                          alpha=alpha)

        backbone_weights = fetch_file(BACKBONE_URL,
                                      fname=fname,
                                      file_hash=BACKBONE_HASH,
                                      cache_subdir='models')
        model.load_weights(backbone_weights, by_name=True)

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
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    model = build_detection_model(seed=args.seed)
    model.summary()
    model.save(args.savepath, include_optimizer=False)
    print(f'Model saved to {args.savepath}')
