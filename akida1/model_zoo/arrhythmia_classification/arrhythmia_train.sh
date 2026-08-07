DATADIR="${1:-}"
DATA_ARG=${DATADIR:+-d "$DATADIR"}

python arrhythmia_model.py -s models/arrhythmia_classification_untrained.h5

python arrhythmia_train.py -l models/arrhythmia_classification_untrained.h5 -s models/arrhythmia_classification.h5 -e 80 -lr 3e-3 -reg 2e-7 $DATA_ARG
python arrhythmia_eval.py -l models/arrhythmia_classification.h5 $DATA_ARG

# 4 bits quantization and tuning
cnn2snn quantize -m models/arrhythmia_classification.h5 -i 8 -w 4 -a 4
python arrhythmia_train.py -l models/arrhythmia_classification_iq8_wq4_aq4.h5 -s models/arrhythmia_classification_qat.h5 -lr 3e-3 -e 50 -reg 2e-7 $DATA_ARG

python arrhythmia_eval.py -l models/arrhythmia_classification_qat.h5 $DATA_ARG

cnn2snn convert -m models/arrhythmia_classification_qat.h5
python arrhythmia_eval.py -l models/arrhythmia_classification_qat.fbz $DATA_ARG

python arrhythmia_benchmark.py -l models/arrhythmia_classification_qat.fbz $DATA_ARG
