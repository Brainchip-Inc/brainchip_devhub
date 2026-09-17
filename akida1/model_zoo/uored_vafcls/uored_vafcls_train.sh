#!/usr/bin/env bash
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
#
# Full UORED-VAFCLS pipeline on ONE bearing-disjoint fold: build, train,
# quantize, tune, convert, evaluate at each stage, and benchmark on hardware.
#
# Usage:  bash uored_vafcls_train.sh [DATADIR] [FOLD] [SEED]
#
# FOLD defaults to 5, the first of the 100 evaluation folds and the fold the
# pretrained models and the published hardware numbers come from.
#
# A single fold tells you very little about this model. The across-fold spread
# is 0.04-0.05 macro AUROC and folds range from about 0.70 to about 0.99 with
# everything else held fixed. For the number that means something, run:
#
#     python uored_vafcls_cross_validate.py
#
# Folds 0-4 are the tuning folds. Any change to the recipe must be evaluated
# there, never on folds 5-104.

DATADIR="${1:-}"
DATA_ARG=${DATADIR:+-d "$DATADIR"}
FOLD="${2:-${FOLD:-5}}"
SEED="${3:-${SEED:-0}}"
COMMON="--fold $FOLD --seed $SEED $DATA_ARG"

echo "UORED-VAFCLS pipeline: fold ${FOLD}, seed ${SEED}"

# 1 - Build the untrained model
python uored_vafcls_model.py -s models/akdcnn_uored_vafcls_untrained.h5 --seed "$SEED"

# 2 - Full-precision training
python uored_vafcls_train.py -l models/akdcnn_uored_vafcls_untrained.h5 \
    -s models/akdcnn_uored_vafcls.h5 -e 30 -lr 2e-4 -b 120 $COMMON

# 3 - Evaluate the float model
python uored_vafcls_eval.py -l models/akdcnn_uored_vafcls.h5 $COMMON

# 4 - Post-training quantization: 4-bit weights and activations, 8-bit input
cnn2snn quantize -m models/akdcnn_uored_vafcls.h5 -i 8 -w 4 -a 4

# 5 - Quantization-aware tuning. Larger than this repository's usual 2 epochs:
#     at 6 steps per epoch, 2 epochs is 12 optimizer steps and recovers nothing.
python uored_vafcls_train.py -l models/akdcnn_uored_vafcls_iq8_wq4_aq4.h5 \
    -s models/akdcnn_uored_vafcls_qat.h5 -e 10 -lr 5e-5 -b 120 $COMMON

# 6 - Evaluate the tuned quantized model
python uored_vafcls_eval.py -l models/akdcnn_uored_vafcls_qat.h5 $COMMON

# 7 - Convert to Akida
cnn2snn convert -m models/akdcnn_uored_vafcls_qat.h5

# 8 - Evaluate on Akida
python uored_vafcls_eval.py -l models/akdcnn_uored_vafcls_qat.fbz $COMMON

# 9 - Hardware benchmark (needs a physical AKD1500)
python uored_vafcls_benchmark.py -l models/akdcnn_uored_vafcls_qat.fbz $COMMON
