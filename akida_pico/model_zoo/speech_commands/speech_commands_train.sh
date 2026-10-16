#!/bin/bash
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
#
# Full pipeline for the Akida Pico keyword spotting example: train the recurrent TENN,
# convert it to stateful form, quantize it to 8 bits and convert it to Akida.
#
# Usage: bash speech_commands_train.sh [DATA_DIR]
#   DATA_DIR: TFDS data_dir for speech_commands (default ./data/speech_commands; TFDS
#   downloads and prepares the dataset there on first use).
set -e

DATADIR="${1:-}"
DATA_ARG=${DATADIR:+-d "$DATADIR"}

# Calibration samples for quantization: int16 audio chunks from the training set
wget -N https://data.brainchip.com/dataset-mirror/samples/sc12/sc12_batch100_1024samples.npz \
     -P data/

# 1. Build and train the model in its training (Kernelized) form
python speech_commands_model.py -s models/tenn_recurrent_sc12_untrained.h5
python speech_commands_train.py -l models/tenn_recurrent_sc12_untrained.h5 \
    -s models/tenn_recurrent_sc12.h5 -e 200 $DATA_ARG
python speech_commands_eval.py -l models/tenn_recurrent_sc12.h5 $DATA_ARG

# 2. Convert to the stateful (streaming) form: same weights, 256-sample chunks
python speech_commands_model.py --stateful -l models/tenn_recurrent_sc12.h5 \
    -s models/tenn_recurrent_sc12_stateful.h5
python speech_commands_eval.py -l models/tenn_recurrent_sc12_stateful.h5 $DATA_ARG

# 3. Quantize to 8-bit weights and activations, with int16 inputs
quantizeml quantize -m models/tenn_recurrent_sc12_stateful.h5 -w 8 -a 8 -id int16 \
    -s models/tenn_recurrent_sc12_stateful_i8_w8_a8.h5 \
    -sa data/sc12_batch100_1024samples.npz -e 1 -bs 100
python speech_commands_eval.py -l models/tenn_recurrent_sc12_stateful_i8_w8_a8.h5 $DATA_ARG

# 4. Convert to Akida and evaluate (Akida runtime; software simulation without a device)
cnn2snn convert -m models/tenn_recurrent_sc12_stateful_i8_w8_a8.h5
python speech_commands_eval.py -l models/tenn_recurrent_sc12_stateful_i8_w8_a8.fbz $DATA_ARG
