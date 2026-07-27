import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from html import parser
import json
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from model import build_akida_model, prepare_qat_model, apply_activity_regularizer
from data import ECGDatasetBuilder,ECGDatasetLoader
from sklearn.metrics import classification_report, confusion_matrix
from cnn2snn import quantize, convert, load_quantized_model

TRAIN_RECORDS = [
    '101', '106', '108', '109', '112', '114', '115', '116',
    '118', '119', '122', '124', '201', '203', '205', '207',
    '208', '209', '215', '220', '223', '230'
]
TEST_RECORDS = [
    '100', '103', '105', '111', '113', '117', '121', '123',
    '200', '202', '210', '212', '213', '214', '219', '221',
    '222', '228', '231', '232', '233', '234'
]

# ==============================================================================
# EVALUATION & METRICS METRIC PLOTTERS
# ==============================================================================
def evaluate_and_report(model, X_test, y_test, run_dir, filename_suffix=""):
    """Generates classification metrics report and logs it to a text file."""
    preds = np.argmax(model.predict(X_test), axis=1)
    report = classification_report(y_test, preds, target_names=["N", "S", "V"])
    print(f"\n--- Classification Report {filename_suffix} ---")
    print(report)
    
    report_path = os.path.join(run_dir, f"classification_report{filename_suffix}.txt")
    with open(report_path, "w") as f:
        f.write(report)

def main():
    parser = argparse.ArgumentParser(description="Akida Model Zoo - ECG Training Pipeline")
    parser.add_argument("--data_dir", required=True, help="Path to preprocessed .npy arrays")
    parser.add_argument("--run_dir", default="./akida_ecg_scalogram", help="Target output folder")
    parser.add_argument("--raw_data_dir", required=False, help="Path to raw MIT-BIH mitdb folder")
    parser.add_argument("--lr", type=float, default= 3e-3 ,help=f'Initial learning rate ')
    parser.add_argument("--float_epochs", type=float, default= 80 ,help=f'Float training epochs ')
    parser.add_argument("--qat_epochs", type=float, default= 50 ,help=f'QAT fine-tuning epochs ')
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training (default: 64)")

    args = parser.parse_args()
    input_shape = (36, 32, 1)
    num_classes = 3
    L1L2_reg_value = 5e-7

    os.makedirs(args.data_dir, exist_ok=True)
    # 1. Initialize the dataset builder
    if args.raw_data_dir:
        builder = ECGDatasetBuilder(dataset_path=args.raw_data_dir, data_dir=args.data_dir, window_before=250, window_after=250, img_size=32)
        builder.preprocess_dataset(TRAIN_RECORDS, TEST_RECORDS)
        
    # 2. Load the preprocessed dataset
    loader = ECGDatasetLoader(data_dir = args.data_dir)
    X_train, y_train, X_val, y_val, X_test, y_test = loader.load_dataset()

    # 3. Define class weights for imbalanced dataset
    class_weights = {0: 1.0, 1: 6.0, 2: 3.0}

    # 4. setup_run_directory
    if args.run_dir:
        """Creates a unique path for saving training outputs and configurations."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.run_dir = f"akida_ecg_scalogram/3class_arrhythmia_classification_{timestamp}"
        os.makedirs(args.run_dir, exist_ok=True)
        print(f"Created active Run Directory: {args.run_dir}")
        
    # a. FP32 Stage
    print("\n--- Phase 1: Training FP32 Baseline Model ---")
    model = build_akida_model(input_shape,num_classes,L1L2_reg_value)
    apply_activity_regularizer(model,L1L2_reg_value)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    
    fp32_path = os.path.join(args.run_dir, "arrhythmia_classification_fp32_model.h5")
    fp32_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=fp32_path,monitor="val_loss",save_best_only=True,verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.3,patience=3,verbose=2),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=8,restore_best_weights=False)
    ]

    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=args.float_epochs, batch_size=args.batch_size, class_weight=class_weights,
        callbacks= fp32_callbacks
    )

    # Save training logs and baseline files
    with open(os.path.join(args.run_dir, "arrhythmia_classification_fp32_training_log.json"), "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=4)
    model.save(os.path.join(args.run_dir, "arrhythmia_classification_fp32_last_model.h5"))
   
    #evaluate the FP32 model
    print("\n--- Phase 1: Evaluating FP32 Baseline Model ---")
    # Evaluate FP32 Baseline
    fp32_path = os.path.join(args.run_dir, "arrhythmia_classification_fp32_model.h5")
    best_fp32 = tf.keras.models.load_model(fp32_path)
    evaluate_and_report(best_fp32, X_test, y_test, args.run_dir, filename_suffix="")

    # b. QAT Stage
    print("\n--- Phase 2: Starting Quantization Aware Training (QAT) ---")
    best_fp32.input_names = [tensor.name.split(":")[0] for tensor in best_fp32.inputs]
    q_model = prepare_qat_model(best_fp32)
    q_model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    
    qat_path = os.path.join(args.run_dir, "arrhythmia_classification_qat_model.h5")
    qat_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=qat_path,monitor="val_loss",save_best_only=True,verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.3,patience=3,verbose=2),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=8,restore_best_weights=False)
    ]

    q_history = q_model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=args.qat_epochs, batch_size=args.batch_size, class_weight=class_weights,
        callbacks=qat_callbacks,verbose=1
    )
    
    # Save QAT logs and weights
    with open(os.path.join(args.run_dir, "arrhythmia_classification_qat_training_log.json"), "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in q_history.history.items()}, f, indent=4)
    q_model.save(os.path.join(args.run_dir, "arrhythmia_classification_qat_last_model.h5"))
   
    #evaluate the QAT model
    print("\n--- Phase 2: Evaluating QAT Model ---")
    qat_path = os.path.join(args.run_dir, "arrhythmia_classification_qat_model.h5")
    best_qat = load_quantized_model(qat_path)
    evaluate_and_report(best_qat, X_test, y_test, args.run_dir, filename_suffix="_quant")

    print(f"[SUCCESS] Training cycles terminated. Outputs exported to {args.run_dir}")


    


if __name__ == "__main__":
    main()