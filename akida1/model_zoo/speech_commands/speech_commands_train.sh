#!/usr/bin/env bash

set -e

DATADIR="${1:-data/sc10}"
CONFIG="${2:-configs/training_cfg.yml}"
OUT_DIR="${3:-models}"
BENCHMARK="${4:-ON}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR/../../..:$PYTHONPATH"

python speech_commands_model.py \
    --config "$CONFIG" \
    -s "$OUT_DIR"/speech_commands_untrained.h5

python speech_commands_train.py \
    -l "$OUT_DIR"/speech_commands_untrained.h5 \
    -s "$OUT_DIR"/speech_commands.h5  \
    -d "$DATADIR" \
    --config "$CONFIG" \
    --float

python speech_commands_eval.py \
    -d "$DATADIR" \
    -l "$OUT_DIR"/speech_commands.h5

# quantization and tuning
cnn2snn quantize -m "$OUT_DIR"/speech_commands.h5 -i 8 -w 4 -a 4

python speech_commands_train.py \
    -l "$OUT_DIR"/speech_commands_iq8_wq4_aq4.h5 \
    -s "$OUT_DIR"/speech_commands_qat.h5  \
    -d "$DATADIR" \
    --config "$CONFIG"

python speech_commands_eval.py \
    -d "$DATADIR" \
    -l "$OUT_DIR"/speech_commands_qat.h5

cnn2snn convert -m "$OUT_DIR"/speech_commands_qat.h5

python speech_commands_eval.py \
    -d "$DATADIR" \
    -l "$OUT_DIR"/speech_commands_qat.fbz

if [[ "${BENCHMARK}" == "ON" ]] ; then
    echo "Running benchmarking script"
    python speech_commands_benchmark.py \
        -l "$OUT_DIR"/speech_commands_qat.fbz \
        -d "$DATADIR"
else
    echo "Skipping benchmarking script"
fi
