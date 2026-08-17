#!/usr/bin/env bash
# Full arrhythmia pipeline: build, train, quantize, tune, convert, benchmark.
#
# Usage:
#   ./arrhythmia_train.sh [DATADIR] [SEED]
#
# Both arguments are optional. SEED (default 7, also settable as an environment
# variable) is forwarded to every step with a random component - weight
# initialisation, training shuffle order, the data splits and the benchmark
# sample draw. The figures in the README come from seed 7; results move
# noticeably from seed to seed, most of all for the supraventricular class, so
# re-running with another seed will not reproduce them exactly.
#
# This version uses the pre-defined strict inter-patient data split.

DATADIR="${1:-}"
DATA_ARG=${DATADIR:+-d "$DATADIR"}
SEED="${2:-${SEED:-7}}"

echo "Running arrhythmia pipeline with seed ${SEED} on the inter-patient split"

python arrhythmia_model.py -s models/arrhythmia_classification_untrained.h5 --seed "$SEED"

python arrhythmia_train.py -l models/arrhythmia_classification_untrained.h5 -s models/arrhythmia_classification.h5 -e 80 -lr 3e-3 -reg 2e-7 --seed "$SEED" $DATA_ARG
python arrhythmia_eval.py -l models/arrhythmia_classification.h5 --seed "$SEED" $DATA_ARG 

# 4 bits quantization and tuning
cnn2snn quantize -m models/arrhythmia_classification.h5 -i 8 -w 4 -a 4
python arrhythmia_train.py -l models/arrhythmia_classification_iq8_wq4_aq4.h5 -s models/arrhythmia_classification_qat.h5 -lr 3e-3 -e 50 -reg 2e-7 --seed "$SEED" $DATA_ARG 

python arrhythmia_eval.py -l models/arrhythmia_classification_qat.h5 --seed "$SEED" $DATA_ARG 

# Generate and test Akida model version
cnn2snn convert -m models/arrhythmia_classification_qat.h5
python arrhythmia_eval.py -l models/arrhythmia_classification_qat.fbz --seed "$SEED" $DATA_ARG 

python arrhythmia_benchmark.py -l models/arrhythmia_classification_qat.fbz --seed "$SEED" $DATA_ARG
