"""
Discover pretrained models in the repository and classify them for CI.

Model artifacts live in ``akida*/model_zoo/<example>/pretrained_models/`` and
follow the naming convention:

    <name>.h5       float Keras model
    <name>_qat.h5   quantized Keras model
    <name>.fbz      Akida model

The architecture is derived from the top-level directory: ``akida1`` -> v1,
``akida2`` -> v2.

Used in two ways:
- imported by ``test/conftest.py`` to parametrize the pytest suites;
- run as a script by the workflows' scope job to compute which models a PR
  touches (``--diff``) or list everything (``--all``), emitting
  ``key=value`` lines suitable for ``$GITHUB_OUTPUT``.

A full run of all models is triggered when the diff touches:
- the ``akida_models`` pin in ``pyproject.toml``,
- anything under ``test/`` or ``.github/workflows/`` (the CI harness itself).
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]

ARCH_BY_DIR = {"akida1": "v1", "akida2": "v2"}
HARNESS_PREFIXES = ("test/", ".github/workflows/")
PYPROJECT = "pyproject.toml"


@dataclass(frozen=True)
class ModelSpec:
    """A single pretrained model artifact."""
    path: str   # repo-relative posix path
    kind: str   # float | quantized | akida
    arch: str   # v1 | v2

    @property
    def abs_path(self):
        return REPO_ROOT / self.path


def classify(rel_path):
    """Classify a repo-relative path into a ModelSpec, or None if it is not a
    pretrained model artifact."""
    parts = PurePosixPath(Path(rel_path).as_posix()).parts
    if len(parts) < 5:
        return None
    arch = ARCH_BY_DIR.get(parts[0])
    if arch is None or parts[1] != "model_zoo" or "pretrained_models" not in parts[:-1]:
        return None
    name = parts[-1]
    if name.endswith(".fbz"):
        kind = "akida"
    elif name.endswith(".h5"):
        kind = "quantized" if name[:-len(".h5")].endswith(("_qat", "_i8_w8_a8")) else "float"
    else:
        return None
    return ModelSpec(path=str(PurePosixPath(*parts)), kind=kind, arch=arch)


def discover_all():
    """Glob the repository for every pretrained model artifact."""
    specs = []
    for arch_dir in ARCH_BY_DIR:
        base = REPO_ROOT / arch_dir / "model_zoo"
        if not base.is_dir():
            continue
        for f in sorted(base.glob("*/pretrained_models/**/*")):
            if f.is_file():
                spec = classify(f.relative_to(REPO_ROOT))
                if spec is not None:
                    specs.append(spec)
    return specs


def _git_diff_lines(base_ref, *paths):
    cmd = ["git", "diff", "--name-status", f"{base_ref}...HEAD", "--", *paths]
    out = subprocess.run(cmd, cwd=REPO_ROOT, check=True,
                         capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def _akida_models_pin_changed(base_ref):
    cmd = ["git", "diff", f"{base_ref}...HEAD", "--", PYPROJECT]
    out = subprocess.run(cmd, cwd=REPO_ROOT, check=True,
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            if "akida_models" in line:
                return True
    return False


def discover_changed(base_ref):
    """Return (specs, run_all) for the diff against base_ref.

    Only added/modified/renamed model artifacts are considered (deletions are
    ignored). run_all is set when the akida_models pin or the CI harness
    changed, in which case all models are returned.
    """
    run_all = False
    specs = []
    for line in _git_diff_lines(base_ref):
        fields = line.split("\t")
        status, path = fields[0], fields[-1]
        if status.startswith("D"):
            continue
        posix = PurePosixPath(Path(path).as_posix())
        if str(posix) == PYPROJECT:
            if _akida_models_pin_changed(base_ref):
                run_all = True
        elif str(posix).startswith(HARNESS_PREFIXES):
            run_all = True
        else:
            spec = classify(path)
            if spec is not None:
                specs.append(spec)
    if run_all:
        return discover_all(), True
    return specs, False


def emit_github_output(specs, run_all, stream=sys.stdout):
    """Emit key=value lines for $GITHUB_OUTPUT: run_all + one space-separated
    path list per (arch, kind) group."""
    print(f"run_all={'true' if run_all else 'false'}", file=stream)
    for arch in ARCH_BY_DIR.values():
        for kind in ("float", "quantized", "akida"):
            paths = [s.path for s in specs if s.arch == arch and s.kind == kind]
            print(f"{arch}_{kind}={' '.join(paths)}", file=stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true",
                       help="List every model in the repository.")
    group.add_argument("--diff", metavar="BASE_REF",
                       help="List models added/modified vs BASE_REF "
                            "(e.g. origin/main).")
    parser.add_argument("--github-output", action="store_true",
                        help="Emit key=value lines for $GITHUB_OUTPUT.")
    args = parser.parse_args()

    if args.diff:
        specs, run_all = discover_changed(args.diff)
    else:
        specs, run_all = discover_all(), args.all

    if args.github_output:
        emit_github_output(specs, run_all)
    else:
        print(f"run_all: {run_all}")
        for spec in specs:
            print(f"{spec.arch:3} {spec.kind:9} {spec.path}")


if __name__ == "__main__":
    main()
