#!/usr/bin/env bash
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
#
# End-to-end segmentation pipeline for Akida 2.
#
# Unlike every other example in this zoo, this model is trained in PyTorch
# and never touches Keras: quantizeml quantizes the ONNX export directly
# (`quantizeml.models.quantize(onnx_model, samples=...)`), so there is no
# `quantizeml quantize` CLI step here (that CLI targets Keras .h5 models) --
# the ONNX quantization step below is a small inline Python block instead.
# 8-bit weights and activations (i8/w8/a8) only -- no lower-precision/QAT
# variant was needed.
#
# Usage:
#   bash segmentation_train.sh [DATADIR]
# where the optional DATADIR overrides the default dataset location
# (./data/cityscapes). See README.md "Dataset setup" -- Cityscapes requires
# a free account and manual download, so there is no wget step here.

set -e

DATADIR="${1:-./data/cityscapes}"

# 1. Build untrained model (ImageNet-pretrained MobileNetV4 encoder, random decoder)
python segmentation_model.py -s models/unetv4akida_segmentation_untrained.pth

# 2. Train (from the untrained model; epochs/LR from the source project).
python segmentation_train.py -l models/unetv4akida_segmentation_untrained.pth \
    -s models/unetv4akida_segmentation.pth -d "$DATADIR" -e 100 -lr 5e-4
python segmentation_eval.py -l models/unetv4akida_segmentation.pth -d "$DATADIR" --save-metrics

# 3. Export to ONNX.
python -c "
import torch
from segmentation_model import build_segmentation_model

model = build_segmentation_model()
model.load_state_dict(torch.load('models/unetv4akida_segmentation.pth', map_location='cpu'))
model.eval()

dummy_input = torch.randn(1, 3, 384, 384, dtype=torch.float32)
torch.onnx.export(
    model, dummy_input, 'models/unetv4akida_segmentation.onnx',
    input_names=['input'], output_names=['output'], opset_version=12,
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    dynamo=False,  # required for this model -- the dynamo exporter path fails on it
)
print('Exported models/unetv4akida_segmentation.onnx')
"
python segmentation_eval.py -l models/unetv4akida_segmentation.onnx -d "$DATADIR" --save-metrics

# 4. Quantize (8-bit weights/activations) and convert to Akida, directly from ONNX.
python -c "
import onnx
from cnn2snn import convert
from quantizeml.models import quantize, QuantizationParams

from segmentation_data import get_samples

onnx_model = onnx.load('models/unetv4akida_segmentation.onnx')
samples = get_samples('$DATADIR', num_samples=1024)

qparams = QuantizationParams(input_weight_bits=8, weight_bits=8, activation_bits=8)
q_model = quantize(onnx_model, qparams=qparams, samples=samples, num_samples=1024, batch_size=32)

akida_model = convert(q_model)
akida_model.summary()
akida_model.save('models/unetv4akida_segmentation_i8_w8_a8.fbz')
print('Saved models/unetv4akida_segmentation_i8_w8_a8.fbz')
"
python segmentation_eval.py -l models/unetv4akida_segmentation_i8_w8_a8.fbz -d "$DATADIR" \
    --max-images 500 --save-metrics

python segmentation_benchmark.py -l models/unetv4akida_segmentation_i8_w8_a8.fbz -d "$DATADIR" \
    --save-metrics || true
