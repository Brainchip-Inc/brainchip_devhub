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

# ImageNet-pretrained AkidaNet backbone (alpha=0.5), used as the starting
# point for the detection backbone. Same URL/hash used by
# `akida_models.imagenet.akidanet_imagenet_pretrained(alpha=0.5, quantized=False)`
# for Akida 1.
BACKBONE_URL = 'https://data.brainchip.com/models/AkidaV1/akidanet/akidanet_imagenet_224_alpha_50.h5'
BACKBONE_HASH = '61f2883a6b798f922a5c0411296219a85f25581d7571f65546557b46066f058f'

NUM_CLASSES = 2  # 'car', 'person' - see LABELS in detection_data.py
NUM_ANCHORS = 5
ALPHA = 0.5


def build_detection_model(seed=42):
    set_random_seed(seed)

    with set_akida_version(AkidaVersion.v1):
        model = yolo_base(input_shape=(224, 224, 3),
                          classes=NUM_CLASSES,
                          nb_box=NUM_ANCHORS,
                          alpha=ALPHA)

        backbone_weights = fetch_file(BACKBONE_URL,
                                      fname='akidanet_imagenet_224_alpha_50.h5',
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
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    model = build_detection_model(seed=args.seed)
    model.summary()
    model.save(args.savepath, include_optimizer=False)
    print(f'Model saved to {args.savepath}')
