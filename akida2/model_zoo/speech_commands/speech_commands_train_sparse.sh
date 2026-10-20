#!/usr/bin/env bash
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
#
# Sparse Speech Commands Keyword Spotting (KWS) variants for Akida 2. Runs
# independently of speech_commands_train.sh, which produces the dense variants.
#
# Hoyer-Square activity regularization on every ReLU raises activation sparsity
# (fewer events, so lower latency on Akida). The regularizer is applied in float
# training and again in QAT, so the model stays sparse after quantization; the 8-bit
# model gets QAT too, because PTQ alone is not reliable on a regularized model. Each
# bit width has its own strength: the strongest that kept the Akida accuracy above
# target (93% at 8-bit, 92% at 4-bit) over three seeds.
# The input layer weights are always 8-bit (-i 8).
# Conversion to .fbz is done with `cnn2snn convert` (accepts quantizeml models).
#
# Usage:
#   bash speech_commands_train_sparse.sh [DATADIR]
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

# 8-BIT SPARSE VARIANT (i8/w8/a8, strength 1e-4) -- QAT, PTQ is a throwaway (_pretmp)
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

# 4-BIT SPARSE VARIANT (i8/w4/a4, strength 2e-5) -- QAT, PTQ is a throwaway (_pretmp)
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
