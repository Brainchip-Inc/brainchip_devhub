#!/usr/bin/env bash
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
#
# End-to-end Speech Commands Keyword Spotting (KWS) pipeline for Akida 2.
#
# Quantization uses `quantizeml`. Two quantized variants are produced:
#   * 8-bit (i8/w8/a8) via PTQ -- no QAT needed (8-bit PTQ is accurate enough).
#   * 4-bit (i8/w4/a4) via QAT only -- 4-bit PTQ accuracy is poor, so the PTQ
#     model is a throwaway on the way to QAT (not stored/evaluated).
# Sparse variants of both are then trained with Hoyer-Square activity
# regularization (see the end of this script).
# The input layer weights are always 8-bit (-i 8).
# Conversion to .fbz is done with `cnn2snn convert` (accepts quantizeml models).
#
# Usage:
#   bash speech_commands_train.sh [DATADIR]
# where the optional DATADIR overrides the default dataset location
# (./data/speech_commands).

set -e

DATADIR="${1:-}"
DATA_ARG=${DATADIR:+-d "$DATADIR"}

# Download batch of samples for calibration
wget -N https://data.brainchip.com/dataset-mirror/samples/kws/kws_batch1024.npz \
     -P data/

# 1. Build untrained model
python speech_commands_model.py -s models/ds_cnn_speech_commands_untrained.h5

# 2. Float training (from scratch). Epoch/LR provisional -- confirm on first run.
python speech_commands_train.py -l models/ds_cnn_speech_commands_untrained.h5 -s models/ds_cnn_speech_commands.h5 -e 16 -lr 1e-3 $DATA_ARG
python speech_commands_eval.py -l models/ds_cnn_speech_commands.h5 $DATA_ARG

# 8-BIT VARIANT (i8/w8/a8) -- PTQ only, no QAT
quantizeml quantize -m models/ds_cnn_speech_commands.h5 -i 8 -w 8 -a 8 \
    -s models/ds_cnn_speech_commands_i8_w8_a8.h5 \
    --samples data/kws_batch1024.npz -e 2 -bs 100

python speech_commands_eval.py -l models/ds_cnn_speech_commands_i8_w8_a8.h5 $DATA_ARG

cnn2snn convert -m models/ds_cnn_speech_commands_i8_w8_a8.h5
python speech_commands_eval.py -l models/ds_cnn_speech_commands_i8_w8_a8.fbz $DATA_ARG

python speech_commands_benchmark.py -l models/ds_cnn_speech_commands_i8_w8_a8.fbz $DATA_ARG || true

# 4-BIT VARIANT (i8/w4/a4) -- QAT only. 4-bit PTQ is a throwaway (_pretmp).
quantizeml quantize -m models/ds_cnn_speech_commands.h5 -i 8 -w 4 -a 4 \
    -s models/ds_cnn_speech_commands_i8_w4_a4_pretmp.h5 \
    --samples data/kws_batch1024.npz -e 2 -bs 100

python speech_commands_train.py -l models/ds_cnn_speech_commands_i8_w4_a4_pretmp.h5 -s models/ds_cnn_speech_commands_i8_w4_a4_qat.h5 -e 16 -lr 1e-3 $DATA_ARG
python speech_commands_eval.py -l models/ds_cnn_speech_commands_i8_w4_a4_qat.h5 $DATA_ARG

cnn2snn convert -m models/ds_cnn_speech_commands_i8_w4_a4_qat.h5
python speech_commands_eval.py -l models/ds_cnn_speech_commands_i8_w4_a4_qat.fbz $DATA_ARG

python speech_commands_benchmark.py -l models/ds_cnn_speech_commands_i8_w4_a4_qat.fbz $DATA_ARG || true

rm -f models/ds_cnn_speech_commands_i8_w4_a4_pretmp.h5

# SPARSE VARIANTS -- Hoyer-Square activity regularization on every ReLU, to raise
# activation sparsity (fewer events, so lower latency on Akida). The regularizer is
# applied in float training and again in QAT, so the model stays sparse after
# quantization; the 8-bit model gets QAT too, because PTQ alone is not reliable on a
# regularized model. Each bit width has its own strength: the strongest that kept the
# Akida accuracy above target (93% at 8-bit, 92% at 4-bit) over three seeds.

# 8-bit sparse (strength 1e-4)
python speech_commands_train.py -l models/ds_cnn_speech_commands_untrained.h5 -s models/ds_cnn_speech_commands_sparse_1e-4.h5 -e 16 -lr 1e-3 -reg 1e-4 $DATA_ARG
python speech_commands_eval.py -l models/ds_cnn_speech_commands_sparse_1e-4.h5 $DATA_ARG

quantizeml quantize -m models/ds_cnn_speech_commands_sparse_1e-4.h5 -i 8 -w 8 -a 8 \
    -s models/ds_cnn_speech_commands_sparse_i8_w8_a8_pretmp.h5 \
    --samples data/kws_batch1024.npz -e 2 -bs 100

python speech_commands_train.py -l models/ds_cnn_speech_commands_sparse_i8_w8_a8_pretmp.h5 -s models/ds_cnn_speech_commands_sparse_i8_w8_a8_qat.h5 -e 16 -lr 1e-3 -reg 1e-4 $DATA_ARG
python speech_commands_eval.py -l models/ds_cnn_speech_commands_sparse_i8_w8_a8_qat.h5 $DATA_ARG

cnn2snn convert -m models/ds_cnn_speech_commands_sparse_i8_w8_a8_qat.h5
python speech_commands_eval.py -l models/ds_cnn_speech_commands_sparse_i8_w8_a8_qat.fbz $DATA_ARG

python speech_commands_benchmark.py -l models/ds_cnn_speech_commands_sparse_i8_w8_a8_qat.fbz $DATA_ARG || true

# 4-bit sparse (strength 2e-5)
python speech_commands_train.py -l models/ds_cnn_speech_commands_untrained.h5 -s models/ds_cnn_speech_commands_sparse_2e-5.h5 -e 16 -lr 1e-3 -reg 2e-5 $DATA_ARG
python speech_commands_eval.py -l models/ds_cnn_speech_commands_sparse_2e-5.h5 $DATA_ARG

quantizeml quantize -m models/ds_cnn_speech_commands_sparse_2e-5.h5 -i 8 -w 4 -a 4 \
    -s models/ds_cnn_speech_commands_sparse_i8_w4_a4_pretmp.h5 \
    --samples data/kws_batch1024.npz -e 2 -bs 100

python speech_commands_train.py -l models/ds_cnn_speech_commands_sparse_i8_w4_a4_pretmp.h5 -s models/ds_cnn_speech_commands_sparse_i8_w4_a4_qat.h5 -e 16 -lr 1e-3 -reg 2e-5 $DATA_ARG
python speech_commands_eval.py -l models/ds_cnn_speech_commands_sparse_i8_w4_a4_qat.h5 $DATA_ARG

cnn2snn convert -m models/ds_cnn_speech_commands_sparse_i8_w4_a4_qat.h5
python speech_commands_eval.py -l models/ds_cnn_speech_commands_sparse_i8_w4_a4_qat.fbz $DATA_ARG

python speech_commands_benchmark.py -l models/ds_cnn_speech_commands_sparse_i8_w4_a4_qat.fbz $DATA_ARG || true

rm -f models/ds_cnn_speech_commands_sparse_i8_w8_a8_pretmp.h5 models/ds_cnn_speech_commands_sparse_i8_w4_a4_pretmp.h5
