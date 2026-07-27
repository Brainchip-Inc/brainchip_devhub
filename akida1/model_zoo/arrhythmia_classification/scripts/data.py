import os
import cv2
import pywt
import wfdb
import time
import numpy as np
from scipy.signal import resample
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

class ECGDatasetBuilder:
    def __init__(self, dataset_path, data_dir, window_before=250, window_after=250, img_size=32):
        self.dataset_path = dataset_path
        self.data_dir = data_dir
        self.window_before = window_before
        self.window_after = window_after
        self.beat_target_length = 360
        self.img_size = img_size
        self.num_classes = 3
        
        self.label_map = {
            "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
            "A": "S", "a": "S", "J": "S", "S": "S",
            "V": "V", "E": "V"
        }
        self.class_to_id = {"N": 0, "S": 1, "V": 2}
    # ==============================================================================
    # 2. SIGNAL PROCESSING CORE MODULES
    # ==============================================================================
    def extract_beat(self,signal, peak):
        """Extracts a fixed-window ECG beat around a given peak location."""
        start = peak - self.window_before
        end = peak + self.window_after

        if start < 0 or end >= len(signal):
            return None

        beat = signal[start:end]

        if len(beat) != self.beat_target_length:
            beat = resample(beat, self.beat_target_length)

        return beat


    def compute_rr_features(self,rpeaks, i):
        """Calculates chronological RR-interval metrics relative to adjacent beats."""
        curr = rpeaks[i]
        rr_prev = 0 if i == 0 else curr - rpeaks[i - 1]
        rr_next = rr_prev if i == len(rpeaks) - 1 else rpeaks[i + 1] - curr

        start = max(0, i - 10)
        rr_history = np.diff(rpeaks[start:i + 1])
        rr_avg = np.mean(rr_history) if len(rr_history) > 0 else rr_prev
        rr_local = rr_prev / (rr_avg + 1e-8)

        return np.array([rr_prev, rr_next, rr_local, rr_avg])


    def beat_to_scalogram(self,beat):
        """Transforms raw time-series 1D vectors into scaled 2D Continuous Wavelet scalograms."""
        beat = beat.astype(np.float32)
        scales = np.arange(1, 33)
        coef, _ = pywt.cwt(beat, scales, 'morl')

        img = np.abs(coef)
        img = np.log1p(img)
        
        # Min-Max Normalization
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        return img.astype(np.float32)


    def process_beat(self, ecg, peaks, peak, symbol, i, record):
        """Pipeline coordinator for a single heartbeat sample execution frame."""
        if symbol not in self.label_map:
            return None

        t0 = time.perf_counter()
        beat = self.extract_beat(ecg, peak)
        t_extract = time.perf_counter() - t0

        if beat is None:
            return None

        t0 = time.perf_counter()
        scalogram = self.beat_to_scalogram(beat)
        t_scalogram = time.perf_counter() - t0

        t0 = time.perf_counter()
        rr_feat = self.compute_rr_features(peaks, i)
        t_rr = time.perf_counter() - t0

        return (
            scalogram,
            self.class_to_id[self.label_map[symbol]],
            rr_feat,
            symbol,
            record,
            t_extract,
            t_scalogram,
            t_rr
        )

    # ==============================================================================
    # 3. MULTI-PROCESSING DATA LOADING OPERATIONS
    # ==============================================================================
    def load_records(self,records):
        """Reads clinical record subsets from disk and processes features in parallel."""
        X, y, RR = [], [], []
        beat_symbol, beat_record = [], []
        t_load = 0

        for record in records:
            print(f"Processing Record: {record}")
            t0 = time.time()
            signal, _ = wfdb.rdsamp(os.path.join(self.dataset_path, record))
            ann = wfdb.rdann(os.path.join(self.dataset_path, record), "atr")
            t_load += time.time() - t0

            ecg = signal[:, 0]
            peaks = ann.sample

            results = Parallel(n_jobs=-1, backend="loky")(
                delayed(self.process_beat)(ecg, peaks, peak, symbol, i, record)
                for i, (peak, symbol) in enumerate(zip(ann.sample, ann.symbol))
            )
            
            valid_results = [r for r in results if r is not None]
            print(f"Annotations: {len(results)} | Valid Beats: {len(valid_results)}")

            for r in valid_results:
                scalogram, label, rr_feat, symbol, record_name, _, _, _ = r
                X.append(scalogram)
                y.append(label)
                RR.append(rr_feat)
                beat_symbol.append(symbol)
                beat_record.append(record_name)

        return (
            np.array(X),
            np.array(RR),
            np.array(y),
            np.array(beat_symbol),
            np.array(beat_record),
            {"load": t_load}
        )


    def add_rr_rows(self,X, RR):
        """Appends physical RR interval state representations onto the bottom axes of scalogram imagery matrix representations."""
        out = []
        for img, rr in zip(X, RR):
            rr_img = np.repeat(rr.reshape(4, 1), img.shape[1], axis=1)
            merged = np.vstack([img, rr_img])
            out.append(merged)
        return np.array(out)
    
    # ==============================================================================
    # 4. MONITORING AND DIAGNOSTICS HELPERS
    # ==============================================================================
    def print_performance_metrics(self, dataset_label, total_samples, total_time, timing_dict, t_norm, t_embed):
        """Logs processing framework execution diagnostics."""
        print(f"\n============ {dataset_label} Performance Metrics ============")
        print(f"Signal loading time:     {timing_dict['load']:.4f} sec")
        print(f"RR normalization time:  {t_norm:.4f} sec")
        print(f"RR embedding time:      {t_embed:.4f} sec")
        print(f"Total processing time:  {total_time:.4f} sec")
        print(f"Total processed samples: {total_samples}")
        
        throughput = total_samples / total_time if total_time > 0 else 0
        time_per_sample = (total_time / total_samples) * 1000 if total_samples > 0 else 0
        print(f"Throughput:              {throughput:.2f} samples/sec")
        print(f"Latency per sample:      {time_per_sample:.4f} ms")
        print("=" * 50)
        
    
    def preprocess_dataset(self, train_records, test_records):
        # --- A. Process Training Set ---
        print("\nExecuting Train Dataset Phase...")
        start_train_time = time.time()
        
        X1_train, RR_train, y1_train, _, _, train_timing = self.load_records(train_records)
        
        # Track statistics based exclusively on training space bounds
        rr_mean = RR_train.mean(axis=0)
        rr_std = RR_train.std(axis=0)

        t0 = time.time()
        RR_train = (RR_train - rr_mean) / (rr_std + 1e-8)
        t_train_norm = time.time() - t0

        t0 = time.time()
        X_train = self.add_rr_rows(X1_train, RR_train)
        t_train_embed = time.time() - t0

        self.print_performance_metrics(
            "Train Set", len(X_train), time.time() - start_train_time, 
            train_timing, t_train_norm, t_train_embed
        )
        
        # --- B. Process Testing Set ---
        print("\nExecuting Test Dataset Phase...")
        start_test_time = time.time()
        
        X_test, RR_test, y_test, _, _, test_timing = self.load_records(test_records)

        t0 = time.time()
        RR_test = (RR_test - rr_mean) / (rr_std + 1e-8)
        t_test_norm = time.time() - t0

        t0 = time.time()
        X_test = self.add_rr_rows(X_test, RR_test)
        t_test_embed = time.time() - t0

        self.print_performance_metrics(
            "Test Set", len(X_test), time.time() - start_test_time, 
            test_timing, t_test_norm, t_test_embed
        )
        
        # --- C. Expand Channel Dimensions ---
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        # --- D. Stratified Validation Splitting ---
        train_idx, val_idx = train_test_split(
            np.arange(len(y1_train)),
            test_size=0.2,
            stratify=y1_train,
            random_state=42
        )

        X_val = X_train[val_idx]
        y_val = y1_train[val_idx]
        X_train_final = X_train[train_idx]
        y_train_final = y1_train[train_idx]

        # --- E. Export Outputs to Disk ---
        #os.makedirs(self.data_dir, exist_ok=True)
        datasets = {
            "X_train.npy": X_train_final,
            "y_train.npy": y_train_final,
            "X_val.npy": X_val,
            "y_val.npy": y_val,
             "X_test.npy": X_test,
            "y_test.npy": y_test
        }

        for filename, array in datasets.items():
            np.save(os.path.join(self.data_dir, filename), array)

        print(f"\n[SUCCESS] Preprocessing completed. Datasets saved to: {self.data_dir}")

class ECGDatasetLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    def load_dataset(self):
        """Loads preprocessed training, validation, and testing numpy arrays."""
        print(f"Loading datasets from: {self.data_dir}...")
        X_train = np.load(os.path.join(self.data_dir, "X_train.npy"))
        y_train = np.load(os.path.join(self.data_dir, "y_train.npy"))
        X_val = np.load(os.path.join(self.data_dir, "X_val.npy"))
        y_val = np.load(os.path.join(self.data_dir, "y_val.npy"))
        X_test = np.load(os.path.join(self.data_dir, "X_test.npy"))
        y_test = np.load(os.path.join(self.data_dir, "y_test.npy"))
        
        print(f"Data successfully loaded. Train shape: {X_train.shape}")
        return X_train, y_train, X_val, y_val, X_test, y_test