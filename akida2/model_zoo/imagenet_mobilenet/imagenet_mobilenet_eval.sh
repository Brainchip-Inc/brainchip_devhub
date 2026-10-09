#!/usr/bin/env bash
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
#
# Evaluate and benchmark one MobileNetV1 ImageNet model, at every pipeline step.
#
# Usage:
#   bash imagenet_mobilenet_eval.sh [ALPHA] [DATADIR] [RESOLUTION]
#
#   ALPHA       width multiplier: 1.0 or 1.25                (default 1.0)
#   DATADIR     ImageNet root, with train/ and validation/   (default ./data/imagenet)
#   RESOLUTION  input resolution: 224 or 256                 (default 224)
#
# Set REBUILD=1 to first re-create the ONNX, quantized and Akida models from the
# timm weights. This overwrites the published models in pretrained_models/, and
# needs the ImageNet train split for quantization calibration:
#   REBUILD=1 bash imagenet_mobilenet_eval.sh 1.0 /path/to/imagenet 256

set -e

ALPHA="${1:-1.0}"
DATADIR="${2:-./data/imagenet}"
RES="${3:-224}"

# Optional: timm float model -> ONNX -> quantizeml (8-bit) -> cnn2snn (Akida)
if [ "${REBUILD:-0}" = "1" ]; then
    python imagenet_mobilenet_model.py -a "$ALPHA" -i "$RES" -d "$DATADIR"
fi

# Float timm model: top-1 / top-5 over the ImageNet validation set
python imagenet_mobilenet_eval.py -a "$ALPHA" -i "$RES" --variant float -d "$DATADIR" --save-metrics

# The same model exported to ONNX: should match the float model exactly
python imagenet_mobilenet_eval.py -a "$ALPHA" -i "$RES" --variant onnx -d "$DATADIR" --save-metrics

# 8-bit post-training quantized model (quantizeml)
python imagenet_mobilenet_eval.py -a "$ALPHA" -i "$RES" --variant quantized -d "$DATADIR" --save-metrics

# Converted Akida model, plus mean activation sparsity
python imagenet_mobilenet_eval.py -a "$ALPHA" -i "$RES" --variant akida -d "$DATADIR" --save-metrics

# Hardware latency. Exits early if no Akida 2 device is connected.
# Uses the 10-image sample pack, so no dataset setup is needed.
python imagenet_mobilenet_benchmark.py -a "$ALPHA" -i "$RES" --save-metrics
python update_readme.py