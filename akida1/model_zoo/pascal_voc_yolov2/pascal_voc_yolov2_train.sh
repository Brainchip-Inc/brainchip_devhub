DATADIR="${1:-}"
DATA_ARG=${DATADIR:+-d "$DATADIR"}

python pascal_voc_yolov2_model.py -s models/yolo_akidanet_detection.h5

python pascal_voc_yolov2_train.py -l models/yolo_akidanet_detection.h5 -s models/yolo_akidanet_detection.h5 -e 70 -lr 5e-4 $DATA_ARG
python pascal_voc_yolov2_eval.py -l models/yolo_akidanet_detection.h5 $DATA_ARG

# 4 bits quantization and tuning
cnn2snn quantize -m models/yolo_akidanet_detection.h5 -i 8 -w 4 -a 4
python pascal_voc_yolov2_train.py -l models/yolo_akidanet_detection_iq8_wq4_aq4.h5 -s models/yolo_akidanet_detection_qat.h5 -lr 5e-5 -e 5 $DATA_ARG

python pascal_voc_yolov2_eval.py -l models/yolo_akidanet_detection_qat.h5 $DATA_ARG

cnn2snn convert -m models/yolo_akidanet_detection_qat.h5
python pascal_voc_yolov2_eval.py -l models/yolo_akidanet_detection_qat.fbz $DATA_ARG

python pascal_voc_yolov2_benchmark.py -l models/yolo_akidanet_detection_qat.fbz $DATA_ARG
