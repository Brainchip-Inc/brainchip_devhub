#!/usr/bin/env bash
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
#
# End-to-end KWS pipeline for Akida Pico: build, train, convert to stateful, quantize,
# convert to Akida, evaluate at every stage.
#
# Quantization is post-training (no QAT): 8-bit weights and activations, 16-bit input.
# Calibration uses 100 segments of 1,024 samples; the committed file is used by default.
#
# Usage:
#   bash kws_train.sh [DATADIR]
# where the optional DATADIR is the TFDS data directory (default ~/tensorflow_datasets).
# Training takes 200 epochs; to skip it, set SKIP_TRAINING=1 to start from the pretrained
# weights in pretrained_models/. Extra arguments for every kws_eval.py call can be passed in
# EVAL_ARGS, e.g. EVAL_ARGS=--save-metrics to record the accuracies in docs/metrics.json.

set -e

DATADIR="${1:-}"
DATA_ARG=${DATADIR:+-d "$DATADIR"}
CALIB=pretrained_models/sc12_calibration_batch100_1024samples.npz

mkdir -p models

if [ -n "$SKIP_TRAINING" ]; then
    cp pretrained_models/tenn_recurrent_sc12.h5 models/tenn_recurrent_sc12.h5
else
    # 1. Build untrained float model, 2. float training (from scratch)
    python kws_model.py -s models/tenn_recurrent_sc12_untrained.h5
    python kws_train.py -l models/tenn_recurrent_sc12_untrained.h5 -s models/tenn_recurrent_sc12.h5 \
        -e 200 -b 128 $DATA_ARG
fi

# 3. Float, kernelized
python kws_eval.py -l models/tenn_recurrent_sc12.h5 $DATA_ARG $EVAL_ARGS

# 4. Stateful conversion: exact, 256-sample chunks (writes *_stateful.h5 next to the input)
tenn_sc12_train convert -m models/tenn_recurrent_sc12.h5 -ts 256 $DATA_ARG
python kws_eval.py -l models/tenn_recurrent_sc12_stateful.h5 $DATA_ARG $EVAL_ARGS

# 5. Quantize: int16 input, 8-bit weights and activations (PTQ)
quantizeml quantize -m models/tenn_recurrent_sc12_stateful.h5 -id int16 -w 8 -a 8 \
    -s models/tenn_recurrent_sc12_stateful_i8_w8_a8.h5 \
    --samples "$CALIB" -e 1 -bs 100
python kws_eval.py -l models/tenn_recurrent_sc12_stateful_i8_w8_a8.h5 $DATA_ARG $EVAL_ARGS

# 6. Convert to Akida and evaluate (simulator; add --hardware on a machine with a Pico FPGA)
cnn2snn convert -m models/tenn_recurrent_sc12_stateful_i8_w8_a8.h5
python kws_eval.py -l models/tenn_recurrent_sc12_stateful_i8_w8_a8.fbz $DATA_ARG $EVAL_ARGS

# 7. Latency on the Pico FPGA (skipped when no device is attached)
python kws_benchmark.py -l models/tenn_recurrent_sc12_stateful_i8_w8_a8.fbz $DATA_ARG || true
