#!/usr/bin/env python3
"""Sweep the Speech Commands training pipeline over activity-regularization
strengths (L1L2 or Hoyer-Square) and collect accuracy / sparsity (/ latency,
if Akida hardware is present) for each point, to scope the accuracy-vs-sparsity
trade-off (Asana: "Akida 1: Sparsify VWW model" -- same investigation applied
to the Speech Commands example; see ../vww/SPARSITY_EXPERIMENT.md and
../plant_village/SPARSITY_EXPERIMENT.md for the original findings this mirrors).

Unlike VWW/PlantVillage, this example already ships with a tuned production
config (configs/training_cfg.yml) that applies raw Hoyer-Square activity
regularization with *different* strengths for the float and QAT stages
(activity_reg_hoyer_strength / activity_reg_hoyer_strength_qat). This sweep
applies the *same* reg value to both stages (matching how VWW/PlantVillage's
sweeps work, and how this script's baseline/L1L2/raw-Hoyer/normalized-Hoyer
sweeps are structured), rather than reusing the production per-stage ratio --
see SPARSITY_EXPERIMENT.md for discussion.

Mirrors the stage sequence in speech_commands_train.sh:
  1. float train        (speech_commands_train.py, generated temp config)
  2. float eval          (speech_commands_eval.py)
  3. quantize            (cnn2snn quantize -i 8 -w 4 -a 4)
  4. QAT fine-tune       (speech_commands_train.py --qat, generated temp config)
  5. QAT eval             (speech_commands_eval.py)
  6. convert to Akida    (cnn2snn convert)
  7. Akida eval           (speech_commands_eval.py)
  8. sparsity/benchmark  (speech_commands_sparsity_only.py if no Akida device
                          is attached, else speech_commands_benchmark.py
                          --save-metrics for real latency/power)

Run from inside akida1/model_zoo/speech_commands/ (same cwd
speech_commands_train.sh expects), with the untrained model already built
(models/speech_commands_untrained.h5, per speech_commands_model.py). The
Speech Commands tfds dataset will auto-download to --data on first use if not
already present (~2.4GB).

Example:
    python speech_commands_sparsity_sweep.py \\
        --untrained models/speech_commands_untrained.h5 \\
        --base-config configs/training_cfg.yml \\
        --data ./data/sc10 \\
        --reg-values 0 1e-6 1e-5 1e-4 1e-3 \\
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

import yaml

ACC_RE = re.compile(r"(?:Akida accuracy|Test accuracy):\s*([0-9.]+)")
SPARSITY_RE = re.compile(r"Mean sparsity:\s*([0-9.]+)")

DOCS_METRICS_PATH = Path(__file__).parent / 'docs' / 'metrics.json'


def check_prereqs(args):
    untrained = Path(args.untrained)
    if not untrained.is_file():
        print(f"Untrained base model not found at: {untrained}\n"
              f"Build it first with: python speech_commands_model.py --config "
              f"{args.base_config} -s {untrained}", file=sys.stderr)
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
    out = run(["python", "speech_commands_eval.py", "-l", str(model_path), "-d", data_dir])
    return parse_last_match(ACC_RE, out)


def make_sweep_config(base_config, reg, reg_type, tag_dir):
    """Write a temp config with activity_reg_hoyer_strength(_qat) and reg_type
    overridden to `reg`/`reg_type`, keeping every other base_config key as-is."""
    cfg = dict(base_config)
    cfg["activity_reg_hoyer_strength"] = reg
    cfg["activity_reg_hoyer_strength_qat"] = reg
    cfg["reg_type"] = reg_type
    cfg_path = tag_dir / "sweep_cfg.yml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    return cfg_path


def sweep_point(reg, args, base_config, tag_dir, have_hardware):
    tag_dir.mkdir(parents=True, exist_ok=True)
    metrics = {"reg": reg, "reg_type": args.reg_type if reg else "none"}

    cfg_path = make_sweep_config(base_config, reg, args.reg_type, tag_dir)

    float_model = tag_dir / "speech_commands.h5"

    # 1. float train
    run([
        "python", "speech_commands_train.py",
        "-l", args.untrained, "-s", str(float_model),
        "-d", args.data, "--config", str(cfg_path),
    ])
    # 2. float eval
    metrics["float_accuracy"] = eval_model(float_model, args.data)

    # 3. quantize
    run(["cnn2snn", "quantize", "-m", str(float_model), "-i", "8", "-w", "4", "-a", "4"])
    quantized_model = float_model.with_name(float_model.stem + "_iq8_wq4_aq4.h5")

    # 4. QAT fine-tune (re-apply the same regularizer so sparsity survives quantization)
    qat_model = tag_dir / "speech_commands_qat.h5"
    run([
        "python", "speech_commands_train.py",
        "-l", str(quantized_model), "-s", str(qat_model),
        "-d", args.data, "--config", str(cfg_path), "--qat",
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
        # speech_commands_benchmark.py only writes to the repo-shared docs/metrics.json
        # (used to generate the README tables), not a per-run file. Snapshot/restore
        # it around the call so the sweep doesn't leave that shared file clobbered
        # with the last reg value's numbers.
        backup = DOCS_METRICS_PATH.read_text() if DOCS_METRICS_PATH.exists() else None
        try:
            run(["python", "speech_commands_benchmark.py", "-l", str(akida_model), "-d", args.data,
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
        out = run(["python", "speech_commands_sparsity_only.py", "-l", str(akida_model),
                   "-d", args.data])
        metrics["sparsity"] = parse_last_match(SPARSITY_RE, out)
        metrics["latency_ms"] = None  # not measurable without real Akida hardware

    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--untrained", default="models/speech_commands_untrained.h5",
                    help="Path to the untrained base model (speech_commands_model.py -s output)")
    p.add_argument("--base-config", default="configs/training_cfg.yml",
                    help="Base config to copy non-regularization settings from "
                         "(filters, dropout, epochs, lr, augmentation, etc.)")
    p.add_argument("--data", default="./data/sc10", help="Speech Commands tfds data directory")
    p.add_argument("--reg-values", nargs="+", type=float, default=[0, 1e-6, 1e-5, 1e-4, 1e-3],
                    help="Activity-regularization strengths to sweep (0 = baseline)")
    p.add_argument("--reg-type", choices=["l1l2", "hoyer_square", "hoyer_square_norm"],
                    default="hoyer_square",
                    help="Regularizer type (passed through the generated config's reg_type key)")
    p.add_argument("--out-dir", default="sweep_results", help="Where per-reg-value models/metrics are written")
    p.add_argument("--csv", default=None, help="Summary CSV path (default: <out-dir>/sweep_summary.csv)")
    args = p.parse_args()

    check_prereqs(args)

    with open(args.base_config) as f:
        base_config = yaml.safe_load(f)

    have_hardware = detect_akida_hardware()
    if have_hardware:
        print("Akida hardware detected -- will run full speech_commands_benchmark.py for real latency/power.")
    else:
        print("No Akida hardware detected -- falling back to speech_commands_sparsity_only.py "
              "(sparsity only, no latency/power numbers).")

    out_dir = Path(args.out_dir)
    csv_path = Path(args.csv) if args.csv else out_dir / "sweep_summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for reg in args.reg_values:
        tag = f"{args.reg_type}_reg_{reg:.1e}" if reg else "reg_0_baseline"
        print(f"\n==== {tag} ====\n")
        try:
            rows.append(sweep_point(reg, args, base_config, out_dir / tag, have_hardware))
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
