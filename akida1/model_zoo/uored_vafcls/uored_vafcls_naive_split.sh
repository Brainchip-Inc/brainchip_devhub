#!/usr/bin/env bash
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
#
# The naive comparator: the same pipeline as uored_vafcls_train.sh, run under a
# deliberately leaky split. Build, train, quantize, tune, convert, and evaluate
# This builds the same pipeline as uored_vafcls_train.sh (Build, train, quantize, tune, convert, and evaluate
), run under a
# deliberately leaky data split (see uored_vafcls_data.py docstring for more info). 
#
# Usage:  bash uored_vafcls_naive_split.sh [DATADIR] [SEED]
#
# THE NUMBER THIS PRINTS IS NOT A RESULT. It is inflated by construction and
# exists only to be compared against uored_vafcls_train.sh, which runs the same
# model and the same recipe under the bearing-disjoint protocol.
#
# This naive split approach divides each recording by time: the first 60% is used for training and the
# last 40% is used for testing. This means that all 20 bearings appear in both train and test, leading to data leakage.
# Both splits draw the same total number of training and test windows,
# so the difference in score is attributable to the leakage and nothing else.
# See the DATA SPLIT PROTOCOL notes in uored_vafcls_data.py.
#
# There is no FOLD argument: unlike the protocol, this split is singular, so
# there is nothing to cross-validate. Its score is stable to about 0.001 where
# the bearing-disjoint folds span 0.04-0.05. Re-run with a different SEED to
# confirm that.
#
# Models are written with a _segment suffix so this never overwrites the
# protocol-trained checkpoints that the README tables describe.

DATADIR="${1:-}"
DATA_ARG=${DATADIR:+-d "$DATADIR"}
SEED="${2:-${SEED:-0}}"
COMMON="--split-mode segment --seed $SEED $DATA_ARG"

echo "UORED-VAFCLS NAIVE pipeline: seed ${SEED}, segment split (leaky by design)"

# 1 - Build the untrained model
python uored_vafcls_model.py -s models/akdcnn_uored_vafcls_segment_untrained.h5 \
    --seed "$SEED"

# 2 - Full-precision training
python uored_vafcls_train.py -l models/akdcnn_uored_vafcls_segment_untrained.h5 \
    -s models/akdcnn_uored_vafcls_segment.h5 -e 30 -lr 2e-4 -b 120 $COMMON

# 3 - Evaluate the float model
python uored_vafcls_eval.py -l models/akdcnn_uored_vafcls_segment.h5 $COMMON

# 4 - Post-training quantization: 4-bit weights and activations, 8-bit input
cnn2snn quantize -m models/akdcnn_uored_vafcls_segment.h5 -i 8 -w 4 -a 4

# 5 - Quantization-aware tuning
python uored_vafcls_train.py \
    -l models/akdcnn_uored_vafcls_segment_iq8_wq4_aq4.h5 \
    -s models/akdcnn_uored_vafcls_segment_qat.h5 -e 10 -lr 1e-5 -b 120 $COMMON

# 6 - Evaluate the tuned quantized model
python uored_vafcls_eval.py -l models/akdcnn_uored_vafcls_segment_qat.h5 $COMMON

# 7 - Convert to Akida
cnn2snn convert -m models/akdcnn_uored_vafcls_segment_qat.h5

# 8 - Evaluate on Akida
python uored_vafcls_eval.py -l models/akdcnn_uored_vafcls_segment_qat.fbz $COMMON

# No hardware benchmark: latency and sparsity are properties of the graph and
# the input statistics, not of how the data was split. For those numbers run
# uored_vafcls_benchmark.py against the protocol-trained model.
