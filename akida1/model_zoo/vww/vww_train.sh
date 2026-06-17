DATADIR="${1:-}"
DATA_ARG=${DATADIR:+-d "$DATADIR"}

python vww_model.py -s models/akidanet_vww_untrained.h5

python vww_train.py -l models/akidanet_vww_untrained.h5 -s models/akidanet_vww.h5 -e 50 -lr 1e-3 $DATA_ARG
python vww_eval.py -l models/akidanet_vww.h5 $DATA_ARG

# 4 bits quantization and tuning
cnn2snn quantize -m models/akidanet_vww.h5 -i 8 -w 4 -a 4
python vww_train.py -l models/akidanet_vww_iq8_wq4_aq4.h5 -s models/akidanet_vww_qat.h5 -lr 1e-4 -e 2 $DATA_ARG

python vww_eval.py -l models/akidanet_vww_qat.h5 $DATA_ARG

cnn2snn convert -m models/akidanet_vww_qat.h5
python vww_eval.py -l models/akidanet_vww_qat.fbz $DATA_ARG

python vww_benchmark.py -l models/akidanet_vww_qat.fbz $DATA_ARG
