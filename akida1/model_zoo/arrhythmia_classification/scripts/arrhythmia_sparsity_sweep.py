#!/usr/bin/env python3
"""Sweep the ECG Arrhythmia Classification training pipeline over activity-
regularization strengths (L1L2 or Hoyer-Square) and collect accuracy / sparsity
for each point, to scope the accuracy-vs-sparsity trade-off (Asana: "Akida 1:
Sparsify VWW model" -- same investigation applied to the Arrhythmia example; see
../vww/SPARSITY_EXPERIMENT.md, ../plant_village/SPARSITY_EXPERIMENT.md and
../speech_commands/SPARSITY_EXPERIMENT.md for the original findings this mirrors).

Unlike the other three examples, this project's train.py runs float training AND
QAT fine-tuning (plus both evals) in a single process/call -- there's no separate
--qat invocation. So each sweep point is:
  1. float train + float eval + QAT fine-tune + QAT eval   (train.py, one call;
     writes <tag_dir>/accuracies.json with float_accuracy/qat_accuracy)
  2. convert to Akida                                       (arrhythmia_convert.py
     -- NOT the `cnn2snn convert` CLI, which only accepts integer scale/shift;
     this model's input range needs float precision, see input_scaling.py)
  3. Akida eval                                              (arrhythmia_eval.py)
  4. sparsity                                                 (arrhythmia_sparsity_only.py
     -- software backend; this machine has no physical Akida hardware attached)

Run from inside akida1/model_zoo/arrhythmia_classification/scripts/, with the
preprocessed dataset already built (--data_dir, per data.py's
ECGDatasetBuilder.preprocess_dataset -- pass --raw-data-dir once to build it from
a downloaded MIT-BIH mitdb/ directory; subsequent sweep points reuse --data_dir).

Example:
    python arrhythmia_sparsity_sweep.py \\
        --data-dir ../data/processed \\
        --reg-values 0 1e-5 1e-3 \\
        --reg-type l1l2 \\
        --out-dir ../sweep_results
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ACC_RE = re.compile(r"(?:Akida accuracy|Test accuracy):\s*([0-9.]+)")
SPARSITY_RE = re.compile(r"Mean sparsity:\s*([0-9.]+)")


def check_prereqs(args):
    data_dir = Path(args.data_dir)
    if not (data_dir / "X_train.npy").is_file():
        print(f"Preprocessed dataset not found at: {data_dir}\n"
              f"Build it first with a --raw-data-dir pass, e.g.:\n"
              f"  python train.py --raw_data_dir <mitdb dir> --data_dir {data_dir} "
              f"--float_epochs 1 --qat_epochs 1", file=sys.stderr)
        sys.exit(1)


def detect_akida_hardware():
    """Return True if a physical Akida device is attached."""
    result = subprocess.run(
        [sys.executable, "-c", "import akida; print(len(akida.devices()))"],
        capture_output=True, text=True)
    if result.returncode != 0:
        return False
    return int(result.stdout.strip().splitlines()[-1]) > 0


def run(cmd):
    print(f"$ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}")
    return result.stdout


def parse_last_match(pattern, stdout):
    matches = pattern.findall(stdout)
    return float(matches[-1]) if matches else None


def sweep_point(reg, args, tag_dir):
    tag_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"reg": reg, "reg_type": args.reg_type if reg else "none"}

    reg_args = ["-reg", str(reg), "--reg-type", args.reg_type] if reg else []

    # 1. float train + float eval + QAT fine-tune + QAT eval (all one process)
    run([
        "python", "train.py",
        "--data_dir", args.data_dir, "--run_dir", str(tag_dir),
        "--float_epochs", str(args.float_epochs), "--qat_epochs", str(args.qat_epochs),
        "--lr", str(args.lr), "--batch_size", str(args.batch_size),
        *reg_args,
    ])
    accuracies = json.loads((tag_dir / "accuracies.json").read_text())
    metrics["float_accuracy"] = accuracies["float_accuracy"]
    metrics["qat_accuracy"] = accuracies["qat_accuracy"]

    # 2. convert to Akida
    qat_model = tag_dir / "arrhythmia_classification_qat_model.h5"
    akida_model = tag_dir / "arrhythmia_classification_qat_model.fbz"
    run(["python", "arrhythmia_convert.py", "-m", str(qat_model), "-o", str(akida_model)])

    # 3. Akida eval
    out = run(["python", "arrhythmia_eval.py", "-l", str(akida_model), "-d", args.data_dir])
    metrics["akida_accuracy"] = parse_last_match(ACC_RE, out)

    # 4. sparsity (software backend -- no physical Akida device on this machine)
    out = run(["python", "arrhythmia_sparsity_only.py", "-l", str(akida_model), "-d", args.data_dir])
    metrics["sparsity"] = parse_last_match(SPARSITY_RE, out)
    metrics["latency_ms"] = None  # not measurable without real Akida hardware

    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="../data/processed",
                    help="Preprocessed .npy arrays dir (ECGDatasetBuilder output)")
    p.add_argument("--reg-values", nargs="+", type=float, default=[0, 1e-5, 1e-3],
                    help="Activity-regularization strengths to sweep (0 = baseline, no -reg flag)")
    p.add_argument("--reg-type", choices=["l1l2", "hoyer_square", "hoyer_square_norm"],
                    default="l1l2",
                    help="Regularizer type to pass through to train.py's --reg-type")
    p.add_argument("--float-epochs", type=int, default=80,
                    help="Epochs for float training (matches docs/config.json's production default)")
    p.add_argument("--qat-epochs", type=int, default=50,
                    help="Epochs for QAT fine-tune (matches docs/config.json's production default)")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-dir", default="../sweep_results", help="Where per-reg-value models/metrics are written")
    p.add_argument("--csv", default=None, help="Summary CSV path (default: <out-dir>/sweep_summary.csv)")
    args = p.parse_args()

    check_prereqs(args)

    have_hardware = detect_akida_hardware()
    if have_hardware:
        print("Akida hardware detected, but this sweep only measures sparsity via the "
              "software backend (arrhythmia_sparsity_only.py) -- no latency/power benchmark "
              "script exists for this project yet.")
    else:
        print("No Akida hardware detected -- using arrhythmia_sparsity_only.py "
              "(sparsity only, no latency/power numbers).")

    out_dir = Path(args.out_dir)
    csv_path = Path(args.csv) if args.csv else out_dir / "sweep_summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for reg in args.reg_values:
        tag = f"{args.reg_type}_reg_{reg:.1e}" if reg else "reg_0_baseline"
        print(f"\n==== {tag} ====\n")
        try:
            rows.append(sweep_point(reg, args, out_dir / tag))
        except RuntimeError as e:
            print(f"SKIPPING {tag}: {e}", file=sys.stderr)
            rows.append({"reg": reg, "error": str(e)})

        # write incrementally so a crash mid-sweep doesn't lose earlier points
        fieldnames = sorted({k for row in rows for k in row})
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nSummary written to {csv_path}")


if __name__ == "__main__":
    main()
