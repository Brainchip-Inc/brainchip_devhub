#!/usr/bin/env bash
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
#
# End-to-end VWW pipeline for Akida 2.
#
# Quantization uses `quantizeml`. Two quantized variants are produced:
#   * 8-bit (i8/w8/a8) via PTQ -- no QAT needed (8-bit PTQ is accurate enough).
#   * 4-bit (i8/w4/a4) via QAT only -- 4-bit PTQ accuracy is poor, so the PTQ
#     model is a throwaway on the way to QAT (not stored/evaluated).
# The input layer weights are always 8-bit (-i 8).
# Conversion to .fbz is done with `cnn2snn convert` (accepts quantizeml models).
#
# Usage:
#   bash vww_train.sh [DATADIR]
# where the optional DATADIR overrides the default dataset location
# (./data/vw_coco2014_96).

set -e

DATADIR="${1:-}"
DATA_ARG=${DATADIR:+-d "$DATADIR"}

# 1. Build untrained float model
python vww_model.py -s models/akidanet_vww_untrained.h5

# 2. Float training (from scratch). Epoch/LR provisional -- confirm on first run.
python vww_train.py -l models/akidanet_vww_untrained.h5 -s models/akidanet_vww.h5 -e 50 -lr 1e-3 $DATA_ARG
python vww_eval.py -l models/akidanet_vww.h5 $DATA_ARG

# 8-BIT VARIANT (i8/w8/a8) -- PTQ only, no QAT
quantizeml quantize -m models/akidanet_vww.h5 -i 8 -w 8 -a 8 \
    -s models/akidanet_vww_i8_w8_a8.h5
python vww_eval.py -l models/akidanet_vww_i8_w8_a8.h5 $DATA_ARG

cnn2snn convert -m models/akidanet_vww_i8_w8_a8.h5
python vww_eval.py -l models/akidanet_vww_i8_w8_a8.fbz $DATA_ARG

python vww_benchmark.py -l models/akidanet_vww_i8_w8_a8.fbz $DATA_ARG || true

# 4-BIT VARIANT (i8/w4/a4) -- QAT only. 4-bit PTQ is a throwaway (_pretmp).
quantizeml quantize -m models/akidanet_vww.h5 -i 8 -w 4 -a 4 \
    -s models/akidanet_vww_i8_w4_a4_pretmp.h5

python vww_train.py -l models/akidanet_vww_i8_w4_a4_pretmp.h5 -s models/akidanet_vww_i8_w4_a4_qat.h5 -e 5 -lr 1e-4 $DATA_ARG
python vww_eval.py -l models/akidanet_vww_i8_w4_a4_qat.h5 $DATA_ARG

cnn2snn convert -m models/akidanet_vww_i8_w4_a4_qat.h5
python vww_eval.py -l models/akidanet_vww_i8_w4_a4_qat.fbz $DATA_ARG

python vww_benchmark.py -l models/akidanet_vww_i8_w4_a4_qat.fbz $DATA_ARG || true

rm -f models/akidanet_vww_i8_w4_a4_pretmp.h5
