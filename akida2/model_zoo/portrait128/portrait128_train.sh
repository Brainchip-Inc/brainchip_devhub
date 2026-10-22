#!/usr/bin/env bash
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
#
# End-to-end Portrait128 segmentation pipeline for Akida 2.
#
# The model's encoder starts from ImageNet-trained AkidaNet 0.5 weights; the
# decoder is trained from random initialization.
#
# Quantization uses `quantizeml`. Two quantized variants are produced:
#   * 8-bit (i8/w8/a8) via PTQ -- no QAT needed (8-bit PTQ is accurate enough).
#   * 4-bit (i8/w4/a4) via QAT only -- 4-bit PTQ accuracy is poor, so the PTQ
#     model is a throwaway on the way to QAT (not stored/evaluated).
# The input layer weights are always 8-bit (-i 8).
# Conversion to .fbz is done with `cnn2snn convert` (accepts quantizeml models).
#
# Usage:
#   bash portrait128_train.sh [DATADIR]
# where the optional DATADIR overrides the default dataset location
# (./data/portrait128).

set -e

DATADIR="${1:-}"
DATA_ARG=${DATADIR:+-d "$DATADIR"}

# Download batch of samples for calibration
wget -N https://data.brainchip.com/dataset-mirror/samples/portrait128/portrait128_batch1024.npz \
     -P data/

# 1. Build the model (ImageNet-pretrained AkidaNet encoder + untrained decoder)
python portrait128_model.py -bw imagenet -s models/akida_unet_portrait128.h5

# 2. Train the model
python portrait128_train.py -l models/akida_unet_portrait128.h5 -s models/akida_unet_portrait128.h5 -e 50 -lr 3e-5 $DATA_ARG
python portrait128_eval.py -l models/akida_unet_portrait128.h5 $DATA_ARG

# 8-BIT VARIANT (i8/w8/a8) -- PTQ only, no QAT
quantizeml quantize -m models/akida_unet_portrait128.h5 -i 8 -w 8 -a 8 \
    -s models/akida_unet_portrait128_i8_w8_a8.h5 \
    --samples data/portrait128_batch1024.npz

python portrait128_eval.py -l models/akida_unet_portrait128_i8_w8_a8.h5 $DATA_ARG

cnn2snn convert -m models/akida_unet_portrait128_i8_w8_a8.h5
python portrait128_eval.py -l models/akida_unet_portrait128_i8_w8_a8.fbz $DATA_ARG

python portrait128_benchmark.py -l models/akida_unet_portrait128_i8_w8_a8.fbz $DATA_ARG || true

# 4-BIT VARIANT (i8/w4/a4) -- QAT only. 4-bit PTQ is a throwaway (_pretmp).
quantizeml quantize -m models/akida_unet_portrait128.h5 -i 8 -w 4 -a 4 \
    -s models/akida_unet_portrait128_i8_w4_a4_pretmp.h5 \
    --samples data/portrait128_batch1024.npz

# QAT at a tenth of the float learning rate
python portrait128_train.py -l models/akida_unet_portrait128_i8_w4_a4_pretmp.h5 -s models/akida_unet_portrait128_i8_w4_a4_qat.h5 -e 10 -lr 3e-6 $DATA_ARG
python portrait128_eval.py -l models/akida_unet_portrait128_i8_w4_a4_qat.h5 $DATA_ARG

cnn2snn convert -m models/akida_unet_portrait128_i8_w4_a4_qat.h5
python portrait128_eval.py -l models/akida_unet_portrait128_i8_w4_a4_qat.fbz $DATA_ARG

python portrait128_benchmark.py -l models/akida_unet_portrait128_i8_w4_a4_qat.fbz $DATA_ARG || true

rm -f models/akida_unet_portrait128_i8_w4_a4_pretmp.h5
