#!/usr/bin/env python3
"""Sweep the PlantVillage training pipeline over activity-regularization
strengths (L1L2 or Hoyer-Square) and collect accuracy / sparsity (/ latency,
if Akida hardware is present) for each point, to scope the accuracy-vs-sparsity
trade-off (Asana: "Akida 1: Sparsify VWW model" -- same investigation applied
to the PlantVillage example; see ../vww/SPARSITY_EXPERIMENT.md for the
original VWW findings this mirrors).

Mirrors the stage sequence in plant_village_train.sh:
  1. float train        (plant_village_train.py, -reg applied here)
  2. float eval          (plant_village_eval.py)
  3. quantize            (cnn2snn quantize -i 8 -w 4 -a 4)
  4. QAT fine-tune       (plant_village_train.py, -reg re-applied here)
  5. QAT eval             (plant_village_eval.py)
  6. convert to Akida    (cnn2snn convert)
  7. Akida eval           (plant_village_eval.py)
  8. sparsity/benchmark  (plant_village_sparsity_only.py if no Akida device is
                          attached, else plant_village_benchmark.py
                          --save-metrics for real latency/power)

Run from inside akida1/model_zoo/plant_village/ (same cwd plant_village_train.sh
expects), with the untrained model already built
(models/akidanet_plant_village_untrained.h5, per step 1 of
plant_village_train.sh). The PlantVillage tfds dataset will auto-download to
--data on first use if not already present (~1-2GB).

Note: plant_village_benchmark.py requires a physical Akida device (AKD1500) --
it hard-exits without one. Training/eval run fine on GPU alone; this script
auto-detects hardware and falls back to plant_village_sparsity_only.py
(software-backend sparsity, no latency/power) when no device is found.

Example:
    python plant_village_sparsity_sweep.py \\
        --untrained models/akidanet_plant_village_untrained.h5 \\
        --data ./data/plant_village \\
        --reg-values 0 1e-6 1e-5 1e-4 1e-3 \\
        --float-epochs 10 --qat-epochs 2 \\
        --out-dir sweep_results
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ACC_RE = re.compile(r"(?:Akida accuracy|Test accuracy):\s*([0-9.]+)")
SPARSITY_RE = re.compile(r"Mean sparsity:\s*([0-9.]+)")

DOCS_METRICS_PATH = Path(__file__).parent / 'docs' / 'metrics.json'


def check_prereqs(args):
    untrained = Path(args.untrained)
    if not untrained.is_file():
        print(f"Untrained base model not found at: {untrained}\n"
              f"Build it first with: python plant_village_model.py -s {untrained}",
              file=sys.stderr)
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


def eval_model(model_path, data_dir):
    out = run(["python", "plant_village_eval.py", "-l", str(model_path), "-d", data_dir])
    return parse_last_match(ACC_RE, out)


def sweep_point(reg, args, tag_dir, have_hardware):
    tag_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"reg": reg, "reg_type": args.reg_type if reg else "none"}

    float_model = tag_dir / "akidanet_plant_village.h5"
    reg_args = ["-reg", str(reg), "--reg-type", args.reg_type] if reg else []

    # 1. float train
    run([
        "python", "plant_village_train.py",
        "-l", args.untrained, "-s", str(float_model),
        "-e", str(args.float_epochs), "-lr", str(args.float_lr),
        "-d", args.data, "-b", str(args.batch_size),
        *reg_args,
    ])
    # 2. float eval
    metrics["float_accuracy"] = eval_model(float_model, args.data)

    # 3. quantize
    run(["cnn2snn", "quantize", "-m", str(float_model), "-i", "8", "-w", "4", "-a", "4"])
    quantized_model = float_model.with_name(float_model.stem + "_iq8_wq4_aq4.h5")

    # 4. QAT fine-tune (re-apply the same regularizer so sparsity survives quantization)
    qat_model = tag_dir / "akidanet_plant_village_qat.h5"
    run([
        "python", "plant_village_train.py",
        "-l", str(quantized_model), "-s", str(qat_model),
        "-e", str(args.qat_epochs), "-lr", str(args.qat_lr),
        "-d", args.data, "-b", str(args.batch_size),
        *reg_args,
    ])
    # 5. QAT eval
    metrics["qat_accuracy"] = eval_model(qat_model, args.data)

    # 6. convert to Akida
    run(["cnn2snn", "convert", "-m", str(qat_model)])
    akida_model = qat_model.with_suffix(".fbz")

    # 7. Akida eval
    metrics["akida_accuracy"] = eval_model(akida_model, args.data)

    # 8. sparsity (+ latency/power if hardware is present)
    if have_hardware:
        # plant_village_benchmark.py only writes to the repo-shared docs/metrics.json
        # (used to generate the README tables), not a per-run file. Snapshot/restore
        # it around the call so the sweep doesn't leave that shared file clobbered
        # with the last reg value's numbers.
        backup = DOCS_METRICS_PATH.read_text() if DOCS_METRICS_PATH.exists() else None
        try:
            run(["python", "plant_village_benchmark.py", "-l", str(akida_model), "-d", args.data,
                 "--save-metrics"])
            bench = json.loads(DOCS_METRICS_PATH.read_text())
            shutil.copy(DOCS_METRICS_PATH, tag_dir / "metrics.json")
            metrics["sparsity"] = bench.get("sparsity")
            for mode in ("minimal", "allnps"):
                for key in ("latency_ms", "total_P", "total_E", "dyn_P", "dyn_E",
                            "nps", "passes", "cycles"):
                    bench_key = f"{mode}_{key}"
                    if bench_key in bench:
                        metrics[bench_key] = bench[bench_key]
        finally:
            if backup is not None:
                DOCS_METRICS_PATH.write_text(backup)
            elif DOCS_METRICS_PATH.exists():
                DOCS_METRICS_PATH.unlink()
    else:
        out = run(["python", "plant_village_sparsity_only.py", "-l", str(akida_model),
                   "-d", args.data])
        metrics["sparsity"] = parse_last_match(SPARSITY_RE, out)
        metrics["latency_ms"] = None  # not measurable without real Akida hardware

    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--untrained", default="models/akidanet_plant_village_untrained.h5",
                    help="Path to the untrained base model (plant_village_model.py -s output)")
    p.add_argument("--data", default="./data/plant_village", help="PlantVillage tfds data directory")
    p.add_argument("--reg-values", nargs="+", type=float, default=[0, 1e-6, 1e-5, 1e-4, 1e-3],
                    help="Activity-regularization strengths to sweep (0 = baseline, no -reg flag)")
    p.add_argument("--reg-type", choices=["l1l2", "hoyer_square", "hoyer_square_norm"],
                    default="l1l2",
                    help="Regularizer type to pass through to plant_village_train.py's --reg-type")
    p.add_argument("--float-epochs", type=int, default=10,
                    help="Epochs for float training (matches plant_village_train.sh's default)")
    p.add_argument("--qat-epochs", type=int, default=2,
                    help="Epochs for QAT fine-tune (matches plant_village_train.sh's default)")
    p.add_argument("--float-lr", type=float, default=1e-3)
    p.add_argument("--qat-lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out-dir", default="sweep_results", help="Where per-reg-value models/metrics are written")
    p.add_argument("--csv", default=None, help="Summary CSV path (default: <out-dir>/sweep_summary.csv)")
    args = p.parse_args()

    check_prereqs(args)

    have_hardware = detect_akida_hardware()
    if have_hardware:
        print("Akida hardware detected -- will run full plant_village_benchmark.py for real latency/power.")
    else:
        print("No Akida hardware detected -- falling back to plant_village_sparsity_only.py "
              "(sparsity only, no latency/power numbers).")

    out_dir = Path(args.out_dir)
    csv_path = Path(args.csv) if args.csv else out_dir / "sweep_summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for reg in args.reg_values:
        tag = f"{args.reg_type}_reg_{reg:.1e}" if reg else "reg_0_baseline"
        print(f"\n==== {tag} ====\n")
        try:
            rows.append(sweep_point(reg, args, out_dir / tag, have_hardware))
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
