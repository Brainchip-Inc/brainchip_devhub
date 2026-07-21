"""
Standalone Hardware Inference Pipeline - 3-Class ECG Arrhythmia Classification
=============================================================================
Replicates the EXACT preprocessing from 3Class_Preprocessing+Training.ipynb
and runs the spiking neural network (SNN) model on physical AKD1500 hardware
(with automatic fallback to CPU simulation if physical device is not found).

Parallelized preprocessing added to measure exact per-beat latencies and throughput.
"""

import os
import argparse
import time
from collections import Counter

import cv2
import pywt
import wfdb
import numpy as np
import random

from scipy.signal import resample
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Model
from joblib import Parallel, delayed  # Added for parallel execution

try:
    import akida
except ImportError:
    print("ERROR: 'akida' package is not installed. Please install it to run SNN models.")
    import sys
    sys.exit(1)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ---------------------------------------------------------------------------
# Constants (must match training notebook exactly)
# ---------------------------------------------------------------------------
WINDOW_BEFORE = 250
WINDOW_AFTER  = 250
NUM_CLASSES   = 3
IMG_SIZE      = 32

TEST_RECORDS = [
    '100', '103', '105', '111', '113', '117', '121', '123',
    '200', '202', '210', '212', '213', '214', '219', '221',
    '222', '228', '231', '232', '233', '234',
]

LABEL_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
}

CLASS_TO_ID = {"N": 0, "S": 1, "V": 2}
TARGET_NAMES = ["N", "S", "V"]

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_DATASET  = "../data/mitdb"
DEFAULT_RUN_DIR  = "../model/3class_run_akida_20260706_133236"
DEFAULT_QAT      = os.path.join(DEFAULT_RUN_DIR, "best_qat_model.h5")
DEFAULT_FBZ      = os.path.join(DEFAULT_RUN_DIR, "ecg_akida1500_3class.fbz")
DEFAULT_SPLITS   = "./data_splits_3class"

# ===========================================================================
# Preprocessing helpers
# ===========================================================================

def extract_beat(signal: np.ndarray, peak: int):
    start = peak - WINDOW_BEFORE
    end   = peak + WINDOW_AFTER

    if start < 0 or end >= len(signal):
        return None

    beat = signal[start:end]

    if len(beat) != 360:
        beat = resample(beat, 360)

    return beat

def compute_rr_features(rpeaks: np.ndarray, i: int) -> np.ndarray:
    curr = rpeaks[i]

    rr_prev = 0 if i == 0 else int(curr - rpeaks[i - 1])
    rr_next = rr_prev if i == len(rpeaks) - 1 else int(rpeaks[i + 1] - curr)

    start      = max(0, i - 10)
    rr_history = np.diff(rpeaks[start:i + 1])
    rr_avg     = np.mean(rr_history) if len(rr_history) > 0 else  rr_prev
    rr_local   = rr_prev / (rr_avg + 1e-8)

    return np.array([rr_prev, rr_next, rr_local, rr_avg], dtype=np.float32)

def beat_to_scalogram(beat: np.ndarray) -> np.ndarray:
    scales = np.arange(1, 65)
    coef, _ = pywt.cwt(beat, scales, "morl")

    img = np.abs(coef)
    img = np.log1p(img)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    return img.astype(np.float32)

def add_rr_rows(X: np.ndarray, RR: np.ndarray) -> np.ndarray:
    out = []
    for img, rr in zip(X, RR):
        rr_img = np.repeat(rr.reshape(4, 1), img.shape[1], axis=1)
        out.append(np.vstack([img, rr_img]))
    return np.array(out)

def process_single_beat(ecg, peaks, peak, symbol, i):
    """Isolated processing framework frame for a single heartbeat to run inside Parallel worker thread."""
    if symbol not in LABEL_MAP:
        return None

    t0 = time.perf_counter()
    beat = extract_beat(ecg, peak)
    t_extract = time.perf_counter() - t0

    if beat is None:
        return None

    t0 = time.perf_counter()
    scalogram = beat_to_scalogram(beat)
    t_scalogram = time.perf_counter() - t0

    t0 = time.perf_counter()
    rr_feat = compute_rr_features(peaks, i)
    t_rr = time.perf_counter() - t0

    return (
        scalogram,
        rr_feat,
        CLASS_TO_ID[LABEL_MAP[symbol]],
        t_extract,
        t_scalogram,
        t_rr
    )

def load_records(records: list, dataset_path: str):
    X, RR, y = [], [], []
    
    all_t_extract = []
    all_t_scalogram = []
    all_t_rr = []

    start_pp_time = time.time()

    for record in records:
        print(f"  Loading record {record} ...")
        signal, _meta = wfdb.rdsamp(os.path.join(dataset_path, record))
        ann           = wfdb.rdann(os.path.join(dataset_path, record), "atr")

        ecg   = signal[:, 0]
        peaks = ann.sample

        # Parallel process beats across all available CPU cores using the loky backend
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(process_single_beat)(ecg, peaks, peak, symbol, i)
            for i, (peak, symbol) in enumerate(zip(ann.sample, ann.symbol))
        )

        for r in results:
            if r is not None:
                scalogram, rr_feat, label_id, t_ex, t_scale, t_r_interval = r
                X.append(scalogram)
                RR.append(rr_feat)
                y.append(label_id)
                
                all_t_extract.append(t_ex)
                all_t_scalogram.append(t_scale)
                all_t_rr.append(t_r_interval)

    total_pp_time = time.time() - start_pp_time
    total_samples = len(X)
    throughput = total_samples / total_pp_time if total_pp_time > 0 else 0

    print(f"\n============ Preprocessing Performance Metrics ============")
    print(f"  Total processed samples: {total_samples}")
    print(f"  Total wall clock time  : {total_pp_time:.4f} sec")
    print(f"  Throughput for PP      : {throughput:.2f} samples/sec")
    print(f"-----------------------------------------------------------")
    print(f"  Core Algorithmic Latency Breakdown (Per Sample Metric):")
    print(f"    Extract/beat         : {np.mean(all_t_extract) * 1000:.4f} ms")
    print(f"    Scalogram/beat       : {np.mean(all_t_scalogram) * 1000:.4f} ms")
    print(f"    RR/beat              : {np.mean(all_t_rr) * 1000:.4f} ms")
    print("===========================================================\n")

    return np.array(X), np.array(RR), np.array(y)

def build_dataset(dataset_path: str):
    print("\n[1/3] Loading TEST records (DS2) with Parallel Processing ...")
    X_test, RR_test, y_test = load_records(TEST_RECORDS, dataset_path)

    print(f"  Raw test  : X={X_test.shape}   RR={RR_test.shape}")

    print("\n[2/3] Normalising RR features (train stats applied to test) ...")
    rr_mean = RR_test.mean(axis=0)
    rr_std  = RR_test.std(axis=0)

    RR_test_n  = (RR_test  - rr_mean) / (rr_std + 1e-8)

    print("\n[3/3] Appending RR rows ...")
    X_test  = add_rr_rows(X_test,   RR_test_n)

    print(f"\n  Final test  : {X_test.shape}   labels: {y_test.shape}")

    return X_test, y_test, rr_mean, rr_std

# ===========================================================================
# Splits management
# ===========================================================================

def save_splits(splits_dir, X_test, y_test, rr_mean, rr_std):
    os.makedirs(splits_dir, exist_ok=True)
    files = {
        "ds2_X_test.npy" : X_test,
        "ds2_y_test.npy" : y_test,
        "rr_mean.npy"    : rr_mean,
        "rr_std.npy"     : rr_std,
    }
    for fname, arr in files.items():
        path = os.path.join(splits_dir, fname)
        np.save(path, arr)
        print(f"  Saved {fname}  {arr.shape}  -> {path}")
    print(f"\nAll splits saved to: {splits_dir}")

def load_splits(splits_dir):
    print(f"\nLoading pre-processed splits from: {splits_dir}")
    X_test  = np.load(os.path.join(splits_dir, "ds2_X_test.npy"))
    y_test  = np.load(os.path.join(splits_dir, "ds2_y_test.npy"))
    rr_mean = np.load(os.path.join(splits_dir, "rr_mean.npy"))
    rr_std = np.load(os.path.join(splits_dir, "rr_std.npy"))

    print(f"  ds2_X_test  : {X_test.shape}")
    print(f"  ds2_y_test  : {y_test.shape}")
    return X_test, y_test, rr_mean, rr_std

# ===========================================================================
# Akida Model Loading & Conversion
# ===========================================================================

def get_akida_model(model_arg):
    if model_arg and os.path.exists(model_arg):
        target_path = model_arg
    else:
        if os.path.exists(DEFAULT_QAT):
            target_path = DEFAULT_QAT
        elif os.path.exists(DEFAULT_FBZ):
            target_path = DEFAULT_FBZ
        else:
            raise FileNotFoundError("Could not find any QAT Keras (.h5) or compiled Akida SNN (.fbz) model.")

    if target_path.endswith('.fbz'):
        print(f"\nLoading compiled Akida SNN model from: {target_path}")
        return akida.Model(target_path)

    print(f"\nCompiling SNN on the fly from QAT model: {target_path}")
    from cnn2snn import quantize, convert, set_akida_version, AkidaVersion
    from quantizeml.model_io import load_model

    dir_name = os.path.dirname(target_path)
    float_path = os.path.join(dir_name, "best_fp32_model.h5")
    if not os.path.exists(float_path):
        raise FileNotFoundError(f"Could not find companion float model structure: {float_path}")

    print("  Loading float model structure...")
    float_model = load_model(float_path)
    
    print("  Building quantized structure...")
    q_model = quantize(
        float_model,
        weight_quantization=4,
        activ_quantization=4,
        input_weight_quantization=8
    )
    
    print("  Loading QAT weights...")
    q_model.load_weights(target_path)

    min_v, max_v = -1.0, 1.0
    scale = 255.0 / (max_v - min_v)
    shift = - min_v * scale
    
    print(f"  Converting with cnn2snn.convert (input_scaling: scale={scale:.2f}, shift={shift:.2f}) ...")
    with set_akida_version(AkidaVersion.v1):
        ak_model = convert(q_model, input_scaling=(scale, shift))

    output_fbz = os.path.join(dir_name, "ecg_akida1500_3class.fbz")
    ak_model.save(output_fbz)
    print(f"  ✓ Compiled SNN saved to: {output_fbz}")
    return ak_model

# ===========================================================================
# SNN Execution Helper
# ===========================================================================

def evaluate_akida(ak_model, X_data, batch_size=128):
    outputs = []
    for idx in range(0, len(X_data), batch_size):
        batch_x = X_data[idx:idx + batch_size]
        batch_out = ak_model.predict(batch_x)
        batch_out = batch_out.squeeze(axis=(1, 2))
        outputs.append(batch_out)
    return np.concatenate(outputs, axis=0)

def silence_output_layer(ak_model):
    try:
        layer = ak_model.layers[-1]
        if len(layer.get_variable_names()) > 0:
            if 'threshold' in layer.get_variable_names():
                th = layer.get_variable('threshold')
                new_th = np.ones_like(th) * 524287
                layer.set_variable('threshold', new_th)
            if 'act_step' in layer.get_variable_names():
                act = layer.get_variable('act_step')
                new_act = np.ones_like(act)
                layer.set_variable('act_step', new_act)
    except Exception as e:
        print(f"Warning: Could not silence layer {ak_model.layers[-1].name}: {e}")
    return ak_model

def remove_final_maxpool(ak_model):
    model_dict = ak_model.to_dict()
    layer_dict = model_dict['layers'][-1]
    if layer_dict['parameters']['layer_type'] in ['Convolutional', 'SeparableConvolutional']:
        if layer_dict['parameters']['pooling_height'] == 2:
            layer_dict['parameters']['pooling_height'] = -1
            layer_dict['parameters']['pooling_width'] = -1
            layer_dict['parameters']['pooling_stride_x'] = -1
            layer_dict['parameters']['pooling_stride_y'] = -1
            layer_dict['parameters']['pool_type'] = 0
            out_h = layer_dict['input_shape'][0]
            out_w = layer_dict['input_shape'][1]
            out_c = layer_dict['output_shape'][2]
            layer_dict['output_shape'] = [out_h, out_w, out_c]

            ak_model = akida.Model.from_dict(model_dict)
    return ak_model

def profile_model_layers(model_akida, device, benchmark_sample, repeats=100):
    print("\n" + "=" * 60)
    print(f"PROFILING PER-LAYER TIMING (repeats={repeats})")
    print("=" * 60)
    num_layers = len(model_akida.layers)
    cumulative_inf_clocks = np.zeros((num_layers, repeats))
    cumulative_inf_times = np.zeros((num_layers, repeats))
    lnames = []
    
    for ll in range(num_layers):
        try:
            cut_model = akida.Model(model_akida.layers[:ll+1])
            cut_model = silence_output_layer(cut_model)
            cut_model = remove_final_maxpool(cut_model)
            
            try:
                cut_model.map(device, mode=akida.MapMode.Minimal, hw_only=True)
            except Exception:
                cut_model.map(device)
                
            lname = cut_model.layers[-1].name
            lnames.append(lname)
            cut_model.forward(benchmark_sample)
            
            for rr in range(repeats):
                start_t = time.perf_counter_ns()
                cut_model.forward(benchmark_sample)
                stop_t = time.perf_counter_ns()
                cumulative_inf_times[ll, rr] = stop_t - start_t
                try:
                    clk = cut_model.metrics['inference_clk']
                except (KeyError, TypeError, AttributeError):
                    clk = 0
                cumulative_inf_clocks[ll, rr] = clk
                
            print(f"  Layer {ll+1}/{num_layers}: {lname:<25} | mapped & profiled successfully.")
        except Exception as e:
            print(f"  ✗ Failed to profile Layer {ll+1} ({model_akida.layers[ll].name}): {e}")
            
    perlayer_inf_clocks = np.copy(cumulative_inf_clocks)
    perlayer_inf_clocks[1:] = cumulative_inf_clocks[1:] - cumulative_inf_clocks[:-1]
    perlayer_inf_clocks = np.clip(perlayer_inf_clocks, 0, None)
    
    clock_freq = 400e6
    perlayer_inf_ms = np.mean(perlayer_inf_clocks / clock_freq * 1000, axis=1)
    
    print("\nPER-LAYER LATENCY SUMMARY (Clock Freq = 400 MHz):")
    print("-" * 60)
    print(f"  {'Layer Name':<25} | {'NPU Clocks':<12} | {'Time (ms)':<10}")
    print("-" * 60)
    for ll in range(len(lnames)):
        avg_clocks = np.mean(perlayer_inf_clocks[ll])
        avg_ms = perlayer_inf_ms[ll]
        print(f"  {lnames[ll]:<25} | {avg_clocks:<12.1f} | {avg_ms:<10.4f} ms")
    print("-" * 60)
    print("=" * 60 + "\n")
    return lnames, perlayer_inf_ms

def compute_activation_sparsity(model, samples, batch_size=64):
    try:
        from akida_models.sparsity import compute_sparsity
        sparsity = compute_sparsity(
            model=model,
            samples=samples,
            batch_size=batch_size,
            verbose=False
        )
        return dict(sparsity)
    except Exception as e:
        print(f"[ERROR] Failed to compute sparsity: {e}")
        return {}

def plot_and_save_metrics(args, model_akida, lnames, perlayer_inf_ms, class_suffix="3class"):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib._color_data as mcd
        if args.model:
            dirname = os.path.dirname(args.model)
        else:
            dirname = DEFAULT_RUN_DIR
    except ImportError:
        print("[WARNING] matplotlib is not installed. Skipping plot generation.")
        return

    try:
        layer_names = []
        layer_nps = []
        pass_ends = []
        seq_ends = []

        for si, seq in enumerate(model_akida.sequences):
            for pi, hwpass in enumerate(seq.passes):
                for layer in hwpass.layers:
                    layer_names.append(layer.name)
                    layer_nps.append(len(layer.mapping.nps))
                pass_end_idx = len(layer_names) - 0.5
                is_last_pass = (pi == len(seq.passes) - 1)
                is_last_seq = (si == len(model_akida.sequences) - 1)
                if not (is_last_pass and is_last_seq):
                    if is_last_pass:
                        seq_ends.append(pass_end_idx)
                    else:
                        pass_ends.append(pass_end_idx)

        fig, ax = plt.subplots(figsize=(max(10, len(layer_names) * 0.5), 5))
        x = range(len(layer_names))
        ax.bar(x, layer_nps, color='steelblue', edgecolor='white')

        for pos in pass_ends:
            ax.axvline(x=pos, color='gray', linestyle='--', linewidth=1.2)
        for pos in seq_ends:
            ax.axvline(x=pos, color='firebrick', linestyle='--', linewidth=1.8)

        ax.set_xticks(list(x))
        ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Number of NPs')
        ax.set_title(f'Akida mapping ({class_suffix}):')
        ax.yaxis.get_major_locator().set_params(integer=True)

        legend_handles = []
        if pass_ends:
            legend_handles.append(mpatches.Patch(color='gray', label='Pass boundary'))
        if seq_ends:
            legend_handles.append(mpatches.Patch(color='firebrick', label='Sequence boundary'))
        if legend_handles:
            ax.legend(handles=legend_handles)

        plt.tight_layout()
        chart_name = f'mapping_chart_{class_suffix}.png'
        plt.savefig(chart_name, dpi=150)
        plt.close(fig)
        print(f"  ✓ Saved mapping chart as '{chart_name}'")
    except Exception as e:
        print(f"  [WARNING] Failed to generate mapping plot: {e}")

    try:
        num_layers = perlayer_inf_ms.shape[0]
        color_offset = 50
        bar_colors = list(mcd.XKCD_COLORS.values())[color_offset:(color_offset+num_layers)]

        fig, ax = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)
        ax.bar(np.arange(num_layers), perlayer_inf_ms, color=bar_colors)
        ax.set_ylabel('Per layer inference (ms)')
        ax.set_xlabel('Layers')
        ax.set_xticks(list(np.arange(num_layers)))
        ax.set_xticklabels(lnames, rotation='vertical')
        fig.suptitle(f"Per-layer latency ({class_suffix})")

        latency_plot_name = f'per_layer_latency_{class_suffix}.png'
        chart_path = os.path.join(dirname, latency_plot_name)
        plt.savefig(chart_path, format='png')
        plt.close(fig)
        print(f"  ✓ Saved per-layer latency plot as '{latency_plot_name}'")
    except Exception as e:
        print(f"  [WARNING] Failed to generate per-layer latency plot: {e}")

def parse_args():
    p = argparse.ArgumentParser(
        description="Akida1500 Hardware ECG Arrhythmia Inference Pipeline (3-Class).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset_path", default=DEFAULT_DATASET, help="MIT-BIH database directory.")
    p.add_argument("--model", default=None, help="Path to compiled SNN model (.fbz) or QAT Keras model (.h5).")
    p.add_argument("--save_splits", action="store_true", default=False, help="Save ds1/ds2 numpy arrays after preprocessing.")
    p.add_argument("--load_splits", action="store_true", default=False, help="Load pre-saved splits (skips raw data preprocessing).")
    p.add_argument("--splits_dir", default=DEFAULT_SPLITS, help="Directory for saving/loading splits.")
    p.add_argument("--batch_size", type=int, default=128, help="Batch size for SNN predictions.")
    p.add_argument("--sparsity", action="store_true", default=True, help="Run the compute sparsity calculation.")
    p.add_argument("--benchmark", action="store_true", default=True, help="Run the deterministic benchmark.")
    p.add_argument("--no_benchmark", action="store_false", dest="benchmark", help="Skip the latency benchmark.")
    p.add_argument("--profile_layers", action="store_true", default=False, help="Profile latency and NPU clock cycles per layer.")
    p.add_argument("--plot", action="store_true", default=False, help="Generate and save mapping and per-layer latency plots.")
    return p.parse_args()

def main():
    args = parse_args()

    print("=" * 60)
    print("  ECG Arrhythmia -- Standalone Hardware SNN Inference Pipeline (3-Class)")
    print("=" * 60)
    print(f"  Dataset      : {args.dataset_path}")
    print(f"  Splits dir   : {args.splits_dir}")

    if args.load_splits:
        X_test, y_test, _, _ = load_splits(args.splits_dir)
    else:
        X_test, y_test, rr_mean, rr_std = build_dataset(args.dataset_path)
        if args.save_splits:
            print("\nSaving splits ...")
            save_splits(args.splits_dir, X_test, y_test, rr_mean, rr_std)

    print("\n" + "=" * 60)
    print("AKD1500 HARDWARE DETECTION")
    print("=" * 60)
    devices = akida.devices()
    if not devices:
        print("  [WARNING] No physical Akida hardware detected! (Falling back to CPU simulation)")
        device = None
    else:
        device = None
        for dd in devices:
            if hasattr(dd, 'ip_version'):
                try:
                    if dd.ip_version == akida.IpVersion.v1:
                        device = dd
                        desc = getattr(dd, 'desc', 'AKD1500')
                        print(f"  ✓ Physical AKD1500 hardware device found: {desc}")
                        break
                except Exception:
                    pass
            if hasattr(dd, 'version'):
                if "1500" in str(dd.version) or "v1" in str(dd.version).lower():
                    device = dd
                    desc = getattr(dd, 'desc', 'AKD1500')
                    print(f"  ✓ Physical AKD1500 hardware device found by version: {desc}")
                    break
        if not device:
            device = devices[0]
            desc = getattr(device, 'desc', 'default')
            print(f"  ✓ Connected to default Akida device: {desc}")

    ak_model = get_akida_model(args.model)

    print("\nMapping model to device ...")
    try:
        ak_model.map(device, mode=akida.MapMode.Minimal, hw_only=True)
        print("  ✓ SNN mapped successfully in Minimal mode with hw_only=True")
    except Exception as e:
        print(f"  [WARNING] Mapping in Minimal mode with hw_only=True failed: {e}")
        print("  Retrying standard mapping...")
        ak_model.map(device)
        print("  ✓ SNN mapped successfully")

    try:
        ak_model.summary()
    except Exception as e:
        print(f"  Could not print summary: {e}")

    print("\nScaling float inputs to SNN uint8 range [0, 255] (optimal range [-1, 1])...")
    if X_test.ndim == 3:
        X_test = X_test[..., np.newaxis]
    min_v, max_v = -1.0, 1.0
    X_uint8 = np.clip(np.round((X_test - min_v) / (max_v - min_v) * 255.0), 0, 255).astype(np.uint8)
    
    t0 = time.time()
    ak_out = remove_final_maxpool(ak_model)
    ak_out = evaluate_akida(ak_model, X_uint8, batch_size=args.batch_size)
    elapsed = time.time() - t0
  
    predictions = np.argmax(ak_out, axis=1)
    n = len(X_uint8)
    print(f"\nInference time : {elapsed:.2f}s  |  {elapsed / n * 1000:.3f} ms/sample  |  Throughput: {n / elapsed:.2f} beats/sec")
    print("\nAccuracy:", accuracy_score(y_test, predictions))
    print("\n--- Classification Report (Akida SNN) ---")
    print(classification_report(y_test, predictions, target_names=TARGET_NAMES, zero_division=0))
    print("--- Confusion Matrix (Akida SNN) ---")
    print(confusion_matrix(y_test, predictions))  
    
    if args.sparsity:
        print("\n" + "=" * 60)
        print("COMPUTING ACTIVATION SPARSITY")
        print("=" * 60)
        num_sparsity_samples = min(1000, len(X_uint8))
        sparsity_samples = X_uint8[:num_sparsity_samples]
        try:
            sparsity_dict = compute_activation_sparsity(ak_model, sparsity_samples)
            print("\nACTIVATION SPARSITY SUMMARY:")
            print("-" * 60)
            print(f"  {'Layer Name':<25} | {'Sparsity (% Zeros)':<20}")
            print("-" * 60)
            for layer_name, sparsity_val in sparsity_dict.items():
                print(f"  {layer_name:<25} | {sparsity_val * 100:.2f}%")
            print("-" * 60)
        except Exception as e:
            print(f"  ? Failed to compute activation sparsity: {e}")
        print("=" * 60 + "\n")

    if args.benchmark:
        print("\n" + "=" * 60)
        print("RUNNING DETERMINISTIC BENCHMARK")
        print("=" * 60)
        ak_model.forward(X_uint8[:1])

        repeats = 100
        inference_clocks = np.zeros((repeats,))
        inference_times = np.zeros((repeats,))
        benchmark_sample = X_uint8[10:11]

        for rr in range(repeats):
            start_t = time.perf_counter_ns()
            ak_model.forward(benchmark_sample)
            stop_t = time.perf_counter_ns()
            inference_times[rr] = stop_t - start_t
            try:
                inference_clocks[rr] = ak_model.metrics['inference_clk']
            except (KeyError, TypeError, AttributeError):
                inference_clocks[rr] = 0

        clock_freq = 400e6
        avg_sys_latency_ms = np.mean(inference_times) / 1e6
        avg_clk_latency_ms = np.mean(inference_clocks / clock_freq) * 1000

        print(f"  Average System-level Latency : {avg_sys_latency_ms:.4f} ms / beat")
        if np.any(inference_clocks > 0):
            print(f"  Average NPU Clock Latency    : {avg_clk_latency_ms:.4f} ms / beat")
            print(f"  Average NPU Clock Cycles     : {np.mean(inference_clocks):.1f} cycles")
        else:
            print("  Average NPU Clock Latency    : N/A (running in CPU simulation)")
        print("=" * 60 + "\n")

        if args.profile_layers:
            lnames, perlayer_inf_ms = profile_model_layers(ak_model, device, benchmark_sample, repeats=100)
            if args.plot:
                plot_and_save_metrics(args, ak_model, lnames, perlayer_inf_ms, class_suffix="3class")

    print("Done.")

if __name__ == "__main__":
    main()