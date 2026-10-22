#!/usr/bin/env bash
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
#
# Sparse PlantVillage variants for Akida 2. Runs independently of
# plant_village_train.sh, which produces the dense variants.
#
# Hoyer-Square activity regularization on every ReLU during float training raises
# activation sparsity (fewer events, so lower latency on Akida). Both bit widths are
# quantized from the same regularized float model (strength 2e-6) and fine-tuned with
# QAT *without* the regularizer: for this model, keeping it in QAT cost too much
# accuracy, while the sparsity from float training mostly survives QAT. The 8-bit
# model gets QAT too, because PTQ alone is not reliable on a regularized model.
# The input layer weights are always 8-bit (-i 8).
# Conversion to .fbz is done with `cnn2snn convert` (accepts quantizeml models).
#
# Usage:
#   bash plant_village_train_sparse.sh [DATADIR]
# where the optional DATADIR overrides the default dataset location
# (./data/plant_village).

set -e

DATADIR="${1:-}"
DATA_ARG=${DATADIR:+-d "$DATADIR"}

# Download batch of samples for calibration
wget -N https://data.brainchip.com/dataset-mirror/samples/plantvillage/plantvillage_batch1024.npz \
     -P data/

# 1. Build the model (ImageNet-pretrained backbone + PlantVillage head)
python plant_village_model.py -s models/mobilenet_plant_village_sparse_2e-6.h5

# 2. Fine-tune the model with the activity regularizer
python plant_village_train.py -l models/mobilenet_plant_village_sparse_2e-6.h5 -s models/mobilenet_plant_village_sparse_2e-6.h5 -e 10 -lr 1e-3 -reg 2e-6 $DATA_ARG
python plant_village_eval.py -l models/mobilenet_plant_village_sparse_2e-6.h5 $DATA_ARG

# 8-BIT SPARSE VARIANT (i8/w8/a8) -- QAT, PTQ is a throwaway (_pretmp)
quantizeml quantize -m models/mobilenet_plant_village_sparse_2e-6.h5 -i 8 -w 8 -a 8 \
    -s models/mobilenet_plant_village_sparse_i8_w8_a8_pretmp.h5 \
    --samples data/plantvillage_batch1024.npz -e 2 -bs 100

python plant_village_train.py -l models/mobilenet_plant_village_sparse_i8_w8_a8_pretmp.h5 -s models/mobilenet_plant_village_sparse_i8_w8_a8_qat.h5 -e 20 -lr 1e-4 $DATA_ARG
python plant_village_eval.py -l models/mobilenet_plant_village_sparse_i8_w8_a8_qat.h5 $DATA_ARG

cnn2snn convert -m models/mobilenet_plant_village_sparse_i8_w8_a8_qat.h5
python plant_village_eval.py -l models/mobilenet_plant_village_sparse_i8_w8_a8_qat.fbz $DATA_ARG

python plant_village_benchmark.py -l models/mobilenet_plant_village_sparse_i8_w8_a8_qat.fbz $DATA_ARG || true

# 4-BIT SPARSE VARIANT (i8/w4/a4) -- QAT, PTQ is a throwaway (_pretmp)
quantizeml quantize -m models/mobilenet_plant_village_sparse_2e-6.h5 -i 8 -w 4 -a 4 \
    -s models/mobilenet_plant_village_sparse_i8_w4_a4_pretmp.h5 \
    --samples data/plantvillage_batch1024.npz -e 2 -bs 100

python plant_village_train.py -l models/mobilenet_plant_village_sparse_i8_w4_a4_pretmp.h5 -s models/mobilenet_plant_village_sparse_i8_w4_a4_qat.h5 -e 20 -lr 1e-4 $DATA_ARG
python plant_village_eval.py -l models/mobilenet_plant_village_sparse_i8_w4_a4_qat.h5 $DATA_ARG

cnn2snn convert -m models/mobilenet_plant_village_sparse_i8_w4_a4_qat.h5
python plant_village_eval.py -l models/mobilenet_plant_village_sparse_i8_w4_a4_qat.fbz $DATA_ARG

python plant_village_benchmark.py -l models/mobilenet_plant_village_sparse_i8_w4_a4_qat.fbz $DATA_ARG || true

rm -f models/mobilenet_plant_village_sparse_i8_w8_a8_pretmp.h5 models/mobilenet_plant_village_sparse_i8_w4_a4_pretmp.h5
