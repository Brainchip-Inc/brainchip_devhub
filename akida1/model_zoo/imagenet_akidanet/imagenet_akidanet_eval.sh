#!/usr/bin/env bash
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
#
# Evaluate and benchmark one published AkidaNet ImageNet model.
#
# Usage:
#   bash imagenet_akidanet_eval.sh [ALPHA] [RESOLUTION] [DATADIR]
#
#   ALPHA       width multiplier: 0.25, 0.5 or 1.0   (default 1.0)
#   RESOLUTION  input resolution: 160 or 224          (default 224)
#   DATADIR     ImageNet directory                    (default ./data/imagenet_tfds)
#
# There is no training stage here: ImageNet training is far too slow to be a
# reproducible exercise, so the published models are evaluated as-is. The
# quantization and conversion steps that produced them are shown, commented out,
# at the bottom for reference.

ALPHA="${1:-1.0}"
RESOLUTION="${2:-224}"
DATADIR="${3:-}"
DATA_ARG=${DATADIR:+-d "$DATADIR"}

MODEL_ARGS="-a $ALPHA -i $RESOLUTION"

# Full-precision model: top-1 / top-5 over the ImageNet validation set
python imagenet_akidanet_eval.py $MODEL_ARGS --variant float $DATA_ARG

# Quantization-aware-trained model (8-bit input, 4-bit weights and activations)
python imagenet_akidanet_eval.py $MODEL_ARGS --variant qat $DATA_ARG

# Converted Akida model, plus mean activation sparsity
python imagenet_akidanet_eval.py $MODEL_ARGS --variant akida $DATA_ARG

# Hardware latency and power. Exits early if no AKD1500 is connected.
# Uses the 10-image sample pack by default, so no dataset setup is needed.
python imagenet_akidanet_benchmark.py $MODEL_ARGS

# -----------------------------------------------------------------------------
# How the published models were produced (for reference; not run here)
# -----------------------------------------------------------------------------
# The float models were trained on ImageNet by BrainChip, then quantized and
# fine-tuned, then converted to Akida format:
#
#   cnn2snn quantize -m akidanet_imagenet_224.h5 -i 8 -w 4 -a 4
#   # ... quantization-aware fine-tuning on ImageNet ...
#   cnn2snn convert -m akidanet_imagenet_224_qat.h5
#
# See ../plant_village/plant_village_train.sh for a complete, runnable version
# of that pipeline on a dataset small enough to train in minutes.
