#!/usr/bin/env bash
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
#
# Evaluate the published float Jester TENN model, quantize it as BrainChip did,
# check the result against the published 8-bit model, convert it to Akida, and
# evaluate both.
#
# Usage:
#   bash jester_eval.sh [DATADIR]
#
#   DATADIR  Jester directory (default: the BrainChip Jester subset, downloaded
#            into data/)
#
# There is no training stage: the float model is BrainChip's published model.
# Its training recipe is train_spatiotemporal_jester.sh, quoted in the README.
# This script reproduces the steps after training into models/. The models in
# pretrained_models/ are the published ones, and the benchmark uses those.

set -e

DATADIR="${1:-}"
DATA_ARG=${DATADIR:+-d "$DATADIR"}
NAME=tenn_spatiotemporal_jester

# 1. Evaluate the float model: whole 16-frame clips at once
python jester_eval.py --variant float $DATA_ARG

# 2. 8-bit quantization, as in train_spatiotemporal_jester.sh: 8-bit weights
#    (input layer included) and activations, per-tensor activation scales, uint8
#    input, calibrated for one epoch on BrainChip's pack of 100 train clips
#    (-bs 100: one batch row per clip)
CALIBRATION=$(python jester_data.py --fetch-calibration | tail -n 1)
mkdir -p models
quantizeml quantize -m pretrained_models/${NAME}.h5 -w 8 -i 8 -a 8 \
    --per_tensor_activations -id uint8 -e 1 -bs 100 -sa "$CALIBRATION" \
    -s models/${NAME}_buffer_i8_w8_a8.h5

# 3. Compare with the published 8-bit model, then evaluate it: streamed frame
#    by frame, buffers reset per clip
python jester_model.py --compare models/${NAME}_buffer_i8_w8_a8.h5
python jester_eval.py --variant quantized -m models $DATA_ARG

# 4. Convert to Akida, then evaluate it on the software backend and measure its
#    activation sparsity
cnn2snn convert -m models/${NAME}_buffer_i8_w8_a8.h5
python jester_eval.py --variant akida -m models $DATA_ARG
