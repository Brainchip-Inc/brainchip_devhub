#!/usr/bin/env bash
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
#
# End-to-end VWW pipeline for Akida 2.
#
# Differences from the Akida 1 pipeline:
#   * Quantization uses `quantizeml` (not `cnn2snn quantize`).
#   * The primary model is quantized to 8-bit weights + 8-bit activations
#     (i8/w8/a8) and does NOT require QAT -- 8-bit PTQ is accurate enough.
#   * We additionally produce a 4-bit-weight variant (w4/a8) in two forms:
#       - PTQ only (no QAT)
#       - with QAT fine-tuning (to recover any accuracy lost at 4-bit)
#   * Conversion to .fbz is still done with `cnn2snn convert`, which accepts
#     quantizeml-quantized models.
#
# Usage:
#   bash vww_train.sh [DATADIR]
# where the optional DATADIR overrides the default dataset location
# (./data/vw_coco2014_96).

set -e

DATADIR="${1:-}"
DATA_ARG=${DATADIR:+-d "$DATADIR"}

# ---------------------------------------------------------------------------
# 1. Build untrained float model
# ---------------------------------------------------------------------------
python vww_model.py -s models/akidanet_vww_untrained.h5

# ---------------------------------------------------------------------------
# 2. Float training (from scratch -- no ImageNet transfer learning for v2)
#    NOTE: epoch count is provisional. The v1 example fine-tuned a pretrained
#    backbone in 20 epochs; this v2 model trains from scratch and will likely
#    need more. Confirm/adjust on first real training run.
# ---------------------------------------------------------------------------
python vww_train.py -l models/akidanet_vww_untrained.h5 -s models/akidanet_vww.h5 -e 3 -lr 1e-3 $DATA_ARG
python vww_eval.py -l models/akidanet_vww.h5 $DATA_ARG

# ===========================================================================
# 8-BIT VARIANT (i8 / w8 / a8) -- no QAT
# ===========================================================================
# quantizeml calibrates during quantization; with no -sa provided it uses
# randomly generated calibration samples (default -ns 1024).
# TODO: confirm whether the v2 flow should pass explicit calibration samples
#       (-sa samples.npz) instead of the default random calibration.
quantizeml quantize -m models/akidanet_vww.h5 -i 8 -w 8 -a 8 \
    -s models/akidanet_vww_i8_w8_a8.h5
python vww_eval.py -l models/akidanet_vww_i8_w8_a8.h5 $DATA_ARG

cnn2snn convert -m models/akidanet_vww_i8_w8_a8.h5
python vww_eval.py -l models/akidanet_vww_i8_w8_a8.fbz $DATA_ARG

python vww_benchmark.py -l models/akidanet_vww_i8_w8_a8.fbz $DATA_ARG

# ===========================================================================
# 4-BIT-WEIGHT VARIANT (i8 / w4 / a8) -- PTQ only, no QAT
# ===========================================================================
quantizeml quantize -m models/akidanet_vww.h5 -i 8 -w 4 -a 8 \
    -s models/akidanet_vww_i8_w4_a8.h5
python vww_eval.py -l models/akidanet_vww_i8_w4_a8.h5 $DATA_ARG

cnn2snn convert -m models/akidanet_vww_i8_w4_a8.h5
python vww_eval.py -l models/akidanet_vww_i8_w4_a8.fbz $DATA_ARG

python vww_benchmark.py -l models/akidanet_vww_i8_w4_a8.fbz $DATA_ARG

# ===========================================================================
# 4-BIT-WEIGHT VARIANT (i8 / w4 / a8) -- with QAT fine-tuning
# ===========================================================================
# Start from the PTQ 4-bit model and fine-tune to recover accuracy.
# NOTE: QAT epochs/LR are provisional -- confirm on first real run.
python vww_train.py -l models/akidanet_vww_i8_w4_a8.h5 -s models/akidanet_vww_i8_w4_a8_qat.h5 -e 5 -lr 1e-4 $DATA_ARG
python vww_eval.py -l models/akidanet_vww_i8_w4_a8_qat.h5 $DATA_ARG

cnn2snn convert -m models/akidanet_vww_i8_w4_a8_qat.h5
python vww_eval.py -l models/akidanet_vww_i8_w4_a8_qat.fbz $DATA_ARG

python vww_benchmark.py -l models/akidanet_vww_i8_w4_a8_qat.fbz $DATA_ARG
