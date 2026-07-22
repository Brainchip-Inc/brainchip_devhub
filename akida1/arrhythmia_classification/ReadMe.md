# Akida SNN Evaluation Framework: MIT-BIH Arrhythmia Classification

This repository contains a specialized machine learning and hardware-benchmarking pipeline designed to process clinical ECG signals, train optimized neural networks, and execute Spiking Neural Networks (SNNs) on physical Akida NPU hardware (with automatic fallback to CPU simulation). 

The architecture converts 1D heartbeats into 2D Continuous Wavelet Transform (CWT) scalograms and embeds chronologically tracked RR-intervals to perform highly efficient 3-class arrhythmia classification (Normal, Supraventricular, and Ventricular beats).

## Dataset Details & Target Classes

The **MIT-BIH Arrhythmia Database** (v1.0.0) is a gold-standard clinical corpus used to evaluate cardiac detection systems.
* **Data Volume:** 48 half-hour two-channel ambulatory ECG recordings digitized at 360 samples per second.
* **Annotations:** Approximately 110,000 independent cardiologist-verified beat labels.
* **Pipeline Grouping:** The scripts automatically map diverse clinical sub-diagnoses into 3 key categories:
  * **`N` (Normal):** Normal sinus rhythm, left/right bundle branch blocks, and escape beats.
  * **`S` (Supraventricular):** Atrial premature beats, aberrated premature beats, and nodal junctional premature beats.
  * **`V` (Ventricular):** Premature ventricular contractions (PVCs) and ventricular escape beats.

## Environment Setup

1. **Replicate the Conda Environment:**
   Create and initialize the virtual workspace using the project’s specific dependency configuration:
   conda env create -f environment.yml
   conda activate akida219
2. **Install Local Utilities:**
  Install the repository's native tools in editable mode at the top level of the repository:
  pip install -v -e .

## Script Architecture & Explanations
* **data.py:** Core digital signal processing and multi-threaded feature preprocessing. Extracts beats in window arrays, generates 2D Morlet CWT scalograms, and computes statistical RR-intervals via multi-core parallelization.  
* **model.py:** Network architecture assembly and quantization setups. Builds Akida_ECG_Sparsity_Net using standard Depthwise Separable convolutions with L1L2 activity regularizers for 4-bit weight/activation QAT.  
* **train.py:** Multi-stage optimization and network training loop orchestration. Trains the FP32 model baseline, transitions to a Phase 2 Quantization Aware Training (QAT) fine-tuning flow, and saves performance weights.  
* **eval.py:** Direct deployment-level hardware inference execution engine. Maps .fbz network files to hardware, runs inference, measures throughput/NPU clocks, and profiles per-layer time distributions.  
* **benchmark.py:** Independent edge-profiling and visualization script. Analyzes overall internal activation sparsity distributions across testing frames and generates layer mapping distributions or latency performance bar charts.  

**Step 1: Download the Source Dataset**
wget -r -np -nH --cut-dirs=3 [https://physionet.org/files/mitdb/1.0.0/](https://physionet.org/files/mitdb/1.0.0/)

(Alternatively, download the .zip archive from https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip and extract it to a directory named ./mitdb).

**Step 2: Run the Training Pipeline**
python train.py --raw_data_dir ./mitdb --data_dir ./processed_data --run_dir --float_epochs 80 --qat_epochs 50 --batch_size 64

Key artifacts like best_fp32_model.h5, best_qat_model.h5, and training logs will be written to a timestamped folder inside akida_ecg_scalogram/.

Processed data: Exports float-normalized 3D/4D numpy files to disk arrays.

**Step 3: Run Hardware Inference & Profiling**
python eval.py --dataset_path ./mitdb --model .akida_ecg_scalogram/<timestamped folder>/best_qat_model.h5 --batch_size 64 

Processed data: Standardizes the inputs into scaled uint8 bounds [0, 255] for immediate SNN injection.

**Step 4: Standalone Network Benchmarking**
python benchmark.py --dataset_path ./mitdb --batch_size 64 --model .akida_ecg_scalogram/<timestamped folder> --sparsity --benchmark --profile_layers --plot

This runs inference, calculates system latencies, and maps performance. It generates visual diagnostics (mapping_chart_3class.png and per_layer_latency_3class.png) in your runtime path.

**AKD1500 Hardware Benchmarking Results**
Below are the verified execution results obtained from running the compiled 3-class model pipeline directly on a physical PCIe Akida AKD1500 NPU (16MB) hardware accelerator.

1. Preprocessing Framework Performance
* Leveraging the parallelized joblib backend on the 49,280 test samples of DS2 yields the following computational footprints:
* Total Preprocessing Wall Clock Time: 15.7635 seconds
* Algorithmic Pipeline Throughput: 3,126.20 samples / second

  Core Latency Breakdown per Heartbeat:
  * Beat Extraction Window: 0.0510 ms
  * CWT Scalogram Transformation: 2.0686 ms (Main algorithmic bottleneck)
  * RR-Interval Feature Tracking: 0.0173 ms

2. Hardware Mapping Summary
* Total On-Chip Memory Utilized: 23,448 Bytes
* Physical Hardware Utilization: 6 Layers mapped onto 6 Neural Processors (NPs) via 1 Sequence Pass.
* External Memory Overhead (DMAs): 0 Bytes (100% On-Chip Execution).

3. Classification Performance (Akida SNN)
Overall Evaluation Accuracy: 95.97%
--- Classification Report (Akida SNN) ---
              precision    recall  f1-score   support

           N       0.98      0.98      0.98     44224
           S       0.64      0.69      0.66      1837
           V       0.84      0.88      0.86      3219
    accuracy                           0.96     49280
   macro avg       0.82       0.85     0.83     49280
weighted avg       0.96      0.96      0.96     49280

4. Spiking Network Activation Sparsity
The L1L2 activity regularizers successfully produced high activation sparsity across the network nodes, dramatically minimizing the system power envelope:

* stem_conv: 93.99% Sparsity (% Zeros)
* block1_sepconv: 72.66% Sparsity
* block2_sepconv: 94.77% Sparsity
* block3_sepconv: 77.80% Sparsity
* dense: 78.81% Sparsity
* dense_1: 0.00% Sparsity

5. NPU Latency & Per-Layer Timing Breakdown
* Average System-level Latency: 1.1758 ms / beat
* Average NPU Clock Latency: 1.1535 ms / beat
* Average NPU Clock Cycles: 461,405.2 cycles / beat
* Total Pipeline Throughput: 1,118.87 beats / second

Per-Layer Overhead Summary (Clock Freq = 400 MHz)

<table border="1" cellpadding="6" cellspacing="0" style="border-collapse: collapse; text-align: left;">
    <thead>
        <tr>
            <th>Layer Name</th>
            <th>NPU Clocks</th>
            <th>Time (ms)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>stem_conv</td>
            <td>37789.0</td>
            <td>0.0945</td>
        </tr>
        <tr>
            <td>block1_sepconv</td>
            <td>46879.1</td>
            <td>0.1172</td>
        </tr>
        <tr>
            <td>block2_sepconv</td>
            <td>301932.9</td>
            <td>0.7548</td>
        </tr>
        <tr>
            <td>block3_sepconv</td>
            <td>67361.8</td>
            <td>0.1684</td>
        </tr>
        <tr>
            <td>dense</td>
            <td>4392.2</td>
            <td>0.0110</td>
        </tr>
        <tr>
            <td>dense_1</td>
            <td>2695.0</td>
            <td>0.0067</td>
        </tr>
    </tbody>
</table>

<p><strong>Clock Frequency:</strong> 400 MHz</p>


**Generated Output Artifacts**

Upon running the full evaluation suite, your active folder will contain:
* mapping_chart_3class.png (NPU Node allocation distributions) 
* per_layer_latency_3class.png (Visual layer-by-layer latency graph charts)