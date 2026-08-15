#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""
Convert a QAT ECG Arrhythmia Classification model (.h5) to an Akida model (.fbz),
using this project's input_scaling.py bounds.

`cnn2snn convert`'s CLI only accepts integer -sc/-sh, but this model's input
range isn't [-1, 1] (see input_scaling.py) and needs float precision to match
arrhythmia_eval.py / arrhythmia_sparsity_only.py exactly -- so this calls the
Python API directly instead of shelling out to the CLI.

Example
-------
    python arrhythmia_convert.py -m model/3class_arrhythmia_classification/arrhythmia_classification_qat_model.h5
"""
import argparse
import os

from cnn2snn import convert, load_quantized_model

from input_scaling import SCALE, SHIFT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='QAT model (.h5) to convert')
    parser.add_argument('-o', '--output', default=None,
                         help='Output .fbz path (default: <model_stem>.fbz)')
    args = parser.parse_args()

    out_path = args.output or os.path.splitext(args.model)[0] + '.fbz'

    q_model = load_quantized_model(args.model)
    convert(q_model, input_scaling=(SCALE, SHIFT), file_path=out_path)
    print(f'Model successfully converted and saved as {out_path}.')
