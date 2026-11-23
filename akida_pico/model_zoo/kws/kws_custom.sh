#!/usr/bin/env bash
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
#
# End-to-end KWS pipeline on your own keywords: train with augmentation, convert to stateful,
# quantize, convert to Akida, and score the held-out test split at every stage.
#
# Same conversion and quantization steps as kws_train.sh (post-training quantization, 8-bit
# weights and activations, 16-bit input); only the data and the training stage differ.
# Calibration uses the 100 segments of 1,024 samples that kws_train_custom.py wrote from your
# training clips.
#
# Usage:
#   bash kws_custom.sh [CONFIG] [DATADIR]
# CONFIG defaults to kws_config.yaml; DATADIR, if given, overrides the config's `data`.
# Set SKIP_TRAINING=1 to reuse an existing models/kws_custom.h5.

set -e

CONFIG="${1:-kws_config.yaml}"
DATADIR="${2:-}"
DATA_ARG=${DATADIR:+-d "$DATADIR"}
MODEL=models/kws_custom

mkdir -p models

if [ -z "$SKIP_TRAINING" ]; then
    python kws_train_custom.py -c "$CONFIG" -s $MODEL.h5 $DATA_ARG
fi

# Float, kernelized
python kws_eval.py -c "$CONFIG" -l $MODEL.h5 $DATA_ARG

# Stateful conversion: exact, 256-sample chunks (writes *_stateful.h5 next to the input)
tenn_sc12_train convert -m $MODEL.h5 -ts 256
python kws_eval.py -c "$CONFIG" -l ${MODEL}_stateful.h5 $DATA_ARG

# Quantize: int16 input, 8-bit weights and activations (PTQ)
quantizeml quantize -m ${MODEL}_stateful.h5 -id int16 -w 8 -a 8 \
    -s ${MODEL}_stateful_i8_w8_a8.h5 \
    --samples ${MODEL}_calibration.npz -e 1 -bs 100
python kws_eval.py -c "$CONFIG" -l ${MODEL}_stateful_i8_w8_a8.h5 $DATA_ARG

# Convert to Akida and evaluate (simulator; the Akida model is evaluated one clip at a time)
cnn2snn convert -m ${MODEL}_stateful_i8_w8_a8.h5
python kws_eval.py -c "$CONFIG" -l ${MODEL}_stateful_i8_w8_a8.fbz $DATA_ARG
