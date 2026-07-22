# Akida SNN Evaluation Framework: MIT-BIH Arrhythmia Classification

This repository provides an end-to-end machine learning and hardware benchmarking pipeline for ECG arrhythmia classification using BrainChip's Akida Neural Processing Unit (NPU).

The framework:

- Processes clinical ECG signals from the MIT-BIH Arrhythmia Database
- Converts 1D heartbeat signals into 2D Continuous Wavelet Transform (CWT) scalograms
- Integrates RR-interval temporal features
- Trains quantized neural networks using Quantization Aware Training (QAT)
- Deploys and evaluates Spiking Neural Networks (SNNs) on Akida hardware
- Automatically falls back to CPU simulation when hardware is unavailable

The final model performs 3-class heartbeat classification:

| Class | Description |
|---------|------------|
| N | Normal |
| S | Supraventricular |
| V | Ventricular |

---

# Dataset Details

The **MIT-BIH Arrhythmia Database (v1.0.0)** is a widely used benchmark dataset for cardiac rhythm analysis.

### Dataset Characteristics

| Metric | Value |
|---------|---------|
| Recordings | 48 |
| Recording Duration | 30 minutes each |
| Sampling Rate | 360 Hz |
| Channels | 2 ECG leads |
| Total Beat Annotations | ~110,000 |

### Target Categories

| Class | Included Beat Types |
|---------|-------------------|
| N | Normal sinus rhythm, bundle branch blocks, escape beats |
| S | Atrial premature beats, aberrated premature beats, nodal premature beats |
| V | Premature ventricular contractions (PVCs), ventricular escape beats |

---

# Environment Setup

## Create Conda Environment

```bash
conda env create -f environment.yml
conda activate akida219
```

## Install Repository Utilities

```bash
pip install -v -e .
```

---

# Repository Structure

## data.py

Core signal-processing pipeline.

Features:

- Beat extraction
- Multi-threaded preprocessing
- Morlet CWT scalogram generation
- RR-interval feature computation

## model.py

Network architecture and quantization pipeline.

Features:

- Akida_ECG_Sparsity_Net
- Depthwise separable convolutions
- L1/L2 activity regularization
- 4-bit QAT configuration

## train.py

Training orchestration.

Features:

- FP32 baseline training
- Quantization-aware fine-tuning
- Model checkpoint generation

## eval.py

Deployment and inference engine.

Features:

- Hardware mapping
- Inference execution
- Throughput measurement
- Layer profiling

## benchmark.py

Benchmarking and visualization utility.

Features:

- Latency profiling
- Sparsity analysis
- Layer mapping statistics
- Performance visualization

---

# Step 1: Download Dataset

```bash
wget -r -np -nH --cut-dirs=3 \
https://physionet.org/files/mitdb/1.0.0/
```

Alternatively, download:

https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip

and extract it into:

```text
./mitdb
```

---

# Step 2: Train the Network

```bash
python train.py \
    --raw_data_dir ./mitdb \
    --data_dir ./processed_data \
    --float_epochs 80 \
    --qat_epochs 50 \
    --batch_size 64
```

Generated artifacts include:

- best_fp32_model.h5
- best_qat_model.h5
- Training logs
- Processed datasets

All outputs are stored in a timestamped experiment directory.

---

# Step 3: Run Hardware Evaluation

```bash
python eval.py \
    --dataset_path ./mitdb \
    --model ./akida_ecg_scalogram/best_qat_model.h5 \
    --batch_size 64
```

Features:

- Akida deployment
- Inference execution
- System latency measurement
- NPU profiling

---

# Step 4: Benchmark the Network

```bash
python benchmark.py \
    --dataset_path ./mitdb \
    --batch_size 64 \
    --model ./akida_ecg_scalogram \
    --sparsity \
    --benchmark \
    --profile_layers \
    --plot
```

Generated figures:

- mapping_chart_3class.png
- per_layer_latency_3class.png

---

# AKD1500 Hardware Benchmark Results

Results obtained from executing the compiled model on a physical AKD1500 PCIe accelerator (16 MB).

---

## Benchmark Highlights

| Metric | Value |
|----------|----------:|
| Classification Accuracy | 95.97% |
| Throughput | 1,118.87 beats/sec |
| Average Latency | 1.1758 ms/beat |
| Average NPU Latency | 1.1535 ms/beat |
| Average NPU Cycles | 461,405 cycles/beat |
| On-Chip Memory Usage | 23,448 Bytes |
| DMA Overhead | 0 Bytes |
| Peak Activation Sparsity | 94.77% |

---

## 1. Preprocessing Performance

### Overall Performance

| Metric | Value |
|----------|----------:|
| Total Processing Time | 15.7635 sec |
| Throughput | 3,126.20 samples/sec |

### Latency Breakdown per Heartbeat

| Operation | Latency (ms) |
|-----------|------------:|
| Beat Extraction | 0.0510 |
| CWT Scalogram Generation | 2.0686 |
| RR Interval Feature Extraction | 0.0173 |

**Observation:** CWT generation is the dominant preprocessing bottleneck.

---

## 2. Hardware Mapping Summary

| Metric | Value |
|----------|----------:|
| On-Chip Memory | 23,448 Bytes |
| Network Layers | 6 |
| Neural Processors Used | 6 |
| Mapping Passes | 1 |
| DMA Overhead | 0 Bytes |

### Key Observation

✅ Entire network executes on-chip with zero external memory transfers.

---

## 3. Classification Performance

**Overall Evaluation Accuracy: 95.97%**

### Class-wise Metrics

| Class | Precision | Recall | F1-Score | Support |
|:------|----------:|--------:|---------:|---------:|
| N | 0.98 | 0.98 | 0.98 | 44,224 |
| S | 0.64 | 0.69 | 0.66 | 1,837 |
| V | 0.84 | 0.88 | 0.86 | 3,219 |

### Aggregate Metrics

| Metric | Precision | Recall | F1-Score | Support |
|---------|----------:|--------:|---------:|---------:|
| Macro Average | 0.82 | 0.85 | 0.83 | 49,280 |
| Weighted Average | 0.96 | 0.96 | 0.96 | 49,280 |

---

## 4. Spiking Network Activation Sparsity

The L1/L2 activity regularizers successfully induced high activation sparsity across the network, significantly reducing computational activity and expected power consumption.

### Layer-wise Sparsity

| Layer | Sparsity (%) |
|---------|------------:|
| stem_conv | 93.99 |
| block1_sepconv | 72.66 |
| block2_sepconv | 94.77 |
| block3_sepconv | 77.80 |
| dense | 78.81 |
| dense_1 | 0.00 |

### Key Observations

- Peak sparsity: **94.77%** (`block2_sepconv`)
- Stem layer achieved **93.99%** sparsity
- Most intermediate layers maintained **>70%** sparsity
- Output classification layer remained fully active

---

## 5. NPU Latency Profile

### Overall Metrics

| Metric | Value |
|----------|----------:|
| Average System Latency | 1.1758 ms |
| Average NPU Latency | 1.1535 ms |
| Average NPU Cycles | 461,405.2 |
| Throughput | 1,118.87 beats/sec |

### Per-Layer Timing Breakdown (400 MHz)

| Layer | NPU Clocks | Time (ms) |
|---------|-----------:|----------:|
| stem_conv | 37,789.0 | 0.0945 |
| block1_sepconv | 46,879.1 | 0.1172 |
| block2_sepconv | 301,932.9 | 0.7548 |
| block3_sepconv | 67,361.8 | 0.1684 |
| dense | 4,392.2 | 0.0110 |
| dense_1 | 2,695.0 | 0.0067 |

### Key Observation

`block2_sepconv` is the dominant compute stage, accounting for the majority of NPU execution latency.

---

# Generated Artifacts

The benchmarking pipeline generates:

| File | Description |
|---------|-------------|
| mapping_chart_3class.png | NPU node allocation visualization |
| per_layer_latency_3class.png | Layer-wise latency breakdown |

---

# Summary

The proposed Akida-compatible ECG classifier achieves:

- **95.97% classification accuracy**
- **1,118.87 beats/sec throughput**
- **1.18 ms inference latency**
- **100% on-chip execution**
- **0-byte DMA overhead**
- **94.77% peak activation sparsity**

These results demonstrate an efficient low-power deployment path for real-time edge-based cardiac arrhythmia detection.