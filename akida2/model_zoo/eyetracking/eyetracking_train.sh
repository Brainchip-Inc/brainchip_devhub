#!/usr/bin/env bash
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
#
# End-to-end eyetracking pipeline for Akida 2.
#
# Quantization uses `quantizeml`, 8-bit weights and activations (i8/w8/a8) --
# unlike the VWW example, no 4-bit/QAT variant is produced here: 8-bit PTQ
# already costs very little accuracy for this model (see README "Model"),
# so there was no need to chase a lower precision.
# Conversion to .fbz is done with `cnn2snn convert` (accepts quantizeml models).
#
# Usage:
#   bash eyetracking_train.sh [DATADIR]
# where the optional DATADIR overrides the default processed-dataset location
# (./data/processed). See README.md "Dataset setup" to build that directory
# from the raw GazeBase download first -- unlike the image-classification
# examples in this zoo, there is no single tarball to fetch here.

set -e

DATADIR="${1:-./data/processed}"
DATA_ARG="-d $DATADIR"

# 1. Build untrained float model
python eyetracking_model.py -s models/tinytemporalcnn_eyetracking_untrained.h5

# 2. Float training (from scratch).
python eyetracking_train.py -l models/tinytemporalcnn_eyetracking_untrained.h5 \
    -s models/tinytemporalcnn_eyetracking.h5 -e 30 -lr 1e-3 $DATA_ARG
python eyetracking_eval.py -l models/tinytemporalcnn_eyetracking.h5 $DATA_ARG --save-metrics

# `quantizeml quantize`'s --samples flag wants an .npz of calibration frames
# on disk (unlike eyetracking_eval.py/eyetracking_benchmark.py, which call
# get_samples() directly in-process) -- write one out from the same helper.
python -c "
import numpy as np
from eyetracking_data import get_samples
np.savez('data/calibration_samples.npz', samples=get_samples('$DATADIR', num_samples=1024))
"

# 8-BIT VARIANT (i8/w8/a8) -- PTQ only, no QAT needed.
quantizeml quantize -m models/tinytemporalcnn_eyetracking.h5 -i 8 -w 8 -a 8 \
    -s models/tinytemporalcnn_eyetracking_i8_w8_a8.h5 \
    --samples data/calibration_samples.npz -e 1 -bs 128

python eyetracking_eval.py -l models/tinytemporalcnn_eyetracking_i8_w8_a8.h5 $DATA_ARG --save-metrics

cnn2snn convert -m models/tinytemporalcnn_eyetracking_i8_w8_a8.h5
python eyetracking_eval.py -l models/tinytemporalcnn_eyetracking_i8_w8_a8.fbz $DATA_ARG --save-metrics

python eyetracking_benchmark.py -l models/tinytemporalcnn_eyetracking_i8_w8_a8.fbz $DATA_ARG --save-metrics || true
