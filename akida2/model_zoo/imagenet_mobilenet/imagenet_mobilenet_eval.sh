#!/usr/bin/env bash
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
#
# Build, quantize, convert and evaluate one ImageNet MobileNet for Akida 2.
#
# Usage:
#   bash imagenet_mobilenet_eval.sh [VERSION] [ALPHA] [DATADIR]
#
#   VERSION  v1 or v2                                        (default v1)
#   ALPHA    width multiplier: v1 0.25, 0.5, 0.75 or 1.0;
#            v2 0.35, 0.5, 0.75 or 1.0                       (default 1.0)
#   DATADIR  ImageNet directory                              (default ./data/imagenet_tfds)
#
# There is no training stage: the float models are the ImageNet-pretrained
# MobileNets from tf_keras.applications, and the quantized models come from
# 8-bit post-training quantization (PTQ) alone. This script reproduces that into
# models/. The models in pretrained_models/ were produced the same way, and the
# benchmark uses those.
#
# Every step needs the ImageNet dataset set up (see the README), except the
# benchmark, which uses a 10-image sample pack.

set -e

VERSION="${1:-v1}"
ALPHA="${2:-1.0}"
DATADIR="${3:-}"
DATA_ARG=${DATADIR:+-d "$DATADIR"}

MODEL_ARGS="-v $VERSION -a $ALPHA"
NAME="mobilenet_${VERSION}_imagenet_224_alpha_${ALPHA}"

# 1. Build the float model from the Keras ImageNet weights, and evaluate it:
#    top-1 / top-5 over the ImageNet validation set
python imagenet_mobilenet_model.py $MODEL_ARGS -s models/${NAME}.h5
python imagenet_mobilenet_eval.py $MODEL_ARGS --variant float $DATA_ARG

# 2. Calibration samples: 1024 images from the ImageNet train split (written once,
#    shared by every model)
if [ ! -f data/imagenet_calib1024.npz ]; then
    python imagenet_mobilenet_data.py $DATA_ARG -o data/imagenet_calib1024.npz
fi

# 3. 8-bit post-training quantization (8-bit input layer weights, weights and
#    activations), then evaluate the quantized model
quantizeml quantize -m models/${NAME}.h5 -i 8 -w 8 -a 8 \
    -s models/${NAME}_i8_w8_a8.h5 \
    -sa data/imagenet_calib1024.npz -e 2 -bs 100
python imagenet_mobilenet_eval.py $MODEL_ARGS --variant quantized -m models $DATA_ARG

# 4. Convert to Akida, then evaluate it and measure its activation sparsity.
#    Runs on the software backend unless an Akida 2 device is connected and the
#    model fits on it.
cnn2snn convert -m models/${NAME}_i8_w8_a8.h5
python imagenet_mobilenet_eval.py $MODEL_ARGS --variant akida -m models $DATA_ARG
