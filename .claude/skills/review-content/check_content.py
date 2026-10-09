#!/usr/bin/env python
# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""Mechanical content checks for brainchip_devhub. Read-only: never writes to the repo.

Checks:
  readme-drift   generated READMEs differ from template.format_map(metrics.json)
  broken-link    relative link / image in a markdown file points at a missing path
  version-badge  akida_models badge disagrees with the pin in pyproject.toml
  machine-path   /mnt/... or /home/... in committed code or docs
  copyright      new model_zoo .py file without the Brainchip copyright header
  lfs            weight file committed as a blob instead of an LFS pointer
  runtime-dir    file committed under an example's data/ or models/ directory
  notebook-noise notebook outputs left by a headless run: split stream outputs, progress-bar
                 \r residue, nbclient timing metadata

Usage:
  python .claude/skills/review-content/check_content.py [--base origin/main]

Findings on lines changed since --base (committed or not) are listed first; the rest are
pre-existing. Exit status is 1 if any finding is on a changed line.
"""
import argparse
import json
import re
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from brainchip_utils.notebooks import noise_findings  # noqa: E402  (stdlib-only module)
TEXT_SUFFIXES = {".py", ".sh", ".md", ".template", ".yml", ".yaml", ".toml"}
LINK_RE = re.compile(r"\]\(([^)\s]+)\)|(?:src|href|srcset)=\"([^\"]+)\"")
PATH_RE = re.compile(r"/(?:mnt|home)/\w[\w.-]*")
COPYRIGHT = "Brainchip Holdings Ltd."


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def repo_files():
    """Tracked files plus untracked, non-ignored ones (new files not yet committed)."""
    listed = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
    return sorted(p for p in set(listed) if (ROOT / p).is_file())


def check_readme_drift():
    for template in sorted(ROOT.glob("akida*/model_zoo/*/docs/README.md.template")):
        example = template.parents[1]
        readme = example / "README.md"
        rel = readme.relative_to(ROOT).as_posix()
        try:
            metrics = json.loads((example / "docs" / "metrics.json").read_text())
            expected = template.read_text().format_map(metrics)
        except (KeyError, AttributeError, ValueError, IndexError) as e:
            yield "readme-drift", rel, f"template does not render: {type(e).__name__}: {e}"
            continue
        if not readme.exists() or readme.read_text() != expected:
            yield "readme-drift", rel, "out of sync with docs/README.md.template + docs/metrics.json"

    landing = ROOT / "akida1" / "README.md"
    try:
        build_table = runpy.run_path(str(ROOT / "akida1" / "update_readme.py"))["build_table"]
    except SyntaxError as e:
        yield "readme-drift", "akida1/README.md", f"not checked: update_readme.py needs Python >= 3.12 ({e.msg})"
        return
    expected = (ROOT / "akida1" / "docs" / "README.md.template").read_text().replace("{model_zoo_table}", build_table())
    if landing.read_text() != expected:
        yield "readme-drift", "akida1/README.md", "out of sync; run akida1/update_readme.py"


def check_links(files):
    for rel in files:
        if not rel.endswith(".md"):
            continue
        path = ROOT / rel
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for match in LINK_RE.finditer(line):
                for target in (match.group(1) or match.group(2)).split(","):
                    target = target.strip().split()[0] if target.strip() else ""
                    target = target.split("#")[0].split("?")[0]
                    if not target or re.match(r"^[a-z]+:", target) or "{" in target:
                        continue
                    resolved = (path.parent / target).resolve()
                    if ROOT not in resolved.parents and resolved != ROOT:
                        continue  # GitHub-relative URLs such as ../../issues
                    if not resolved.exists():
                        yield "broken-link", f"{rel}:{lineno}", target


def check_version_badges():
    pin = re.search(r"akida_models==([\w.]+)", (ROOT / "pyproject.toml").read_text())
    if not pin:
        return
    for rel in ("README.md", "akida1/docs/README.md.template"):
        for lineno, line in enumerate((ROOT / rel).read_text().splitlines(), 1):
            for badge in re.findall(r"akida__models-([\d.]+)-", line):
                if badge != pin.group(1):
                    yield "version-badge", f"{rel}:{lineno}", f"badge {badge} != pyproject pin {pin.group(1)}"


def check_text(files, added):
    for rel in files:
        path = ROOT / rel
        if path.suffix == ".ipynb":
            cells = json.loads(path.read_text()).get("cells", [])
            lines = [l for c in cells for l in c.get("source", [])]
        elif path.suffix in TEXT_SUFFIXES or path.name.endswith(".md.template"):
            lines = path.read_text(errors="replace").splitlines()
        else:
            continue
        for lineno, line in enumerate(lines, 1):
            if PATH_RE.search(line):
                where = rel if path.suffix == ".ipynb" else f"{rel}:{lineno}"
                yield "machine-path", where, PATH_RE.search(line).group(0)
        is_new_zoo_script = rel in added and "model_zoo" in rel and path.suffix == ".py"
        if is_new_zoo_script and not any(COPYRIGHT in l for l in lines[:3]):
            yield "copyright", rel, "missing copyright header in first 3 lines"


def check_binaries(files):
    for rel in files:
        parts = Path(rel).parts
        if "model_zoo" in parts and len(parts) > 4 and parts[3] in ("data", "models") and parts[-1] != ".gitignore":
            yield "runtime-dir", rel, "runtime-only directory; should not be committed"
        if rel.endswith((".h5", ".fbz")):
            if not git("cat-file", "-p", f":{rel}").startswith("version https://git-lfs"):
                yield "lfs", rel, "committed as a regular blob, not an LFS pointer"


def check_notebooks(files):
    for rel in files:
        if not rel.endswith(".ipynb"):
            continue
        findings = list(noise_findings(json.loads((ROOT / rel).read_text())))
        if findings:
            cells = sorted({index for index, _ in findings})
            problems = sorted({problem.split(" into ")[0] for _, problem in findings})
            yield ("notebook-noise", rel,
                   f"{'; '.join(problems)} in {len(cells)} cell(s); "
                   f"run python -m brainchip_utils.notebooks --tidy-only {rel}")


def changed_lines(base):
    """{path: set of changed line numbers, or None for a new file} vs the merge base,
    covering committed and uncommitted changes, numbered as in the working tree."""
    merge_base = git("merge-base", base, "HEAD").strip()
    changed, current = {}, None
    for line in git("diff", "-U0", merge_base).splitlines():
        if line.startswith("+++ "):
            current = line[6:] if line.startswith("+++ b/") else None
            if current:
                changed.setdefault(current, set())
        elif line.startswith("@@") and current:
            start, _, count = re.search(r"\+(\d+)(,(\d+))?", line).groups()
            start, count = int(start), int(count) if count is not None else 1
            changed[current].update(range(start, start + count))
    added = set(git("diff", "--name-only", "--diff-filter=A", merge_base).splitlines())
    added |= set(git("ls-files", "--others", "--exclude-standard").splitlines())
    changed.update({f: None for f in added})
    return changed, added


def in_change(finding, changed):
    rel, _, lineno = finding[1].partition(":")
    if rel not in changed:
        return False
    return not lineno or changed[rel] is None or int(lineno) in changed[rel]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="origin/main", help="ref to diff against (default: origin/main)")
    args = parser.parse_args()

    files = repo_files()
    changed, added = changed_lines(args.base)

    findings = [*check_readme_drift(), *check_links(files), *check_version_badges(),
                *check_text(files, added), *check_binaries(files), *check_notebooks(files)]
    in_diff = [f for f in findings if in_change(f, changed)]
    pre_existing = [f for f in findings if f not in in_diff]

    for title, group in (("In this change", in_diff), ("Pre-existing", pre_existing)):
        print(f"## {title}: {len(group)}")
        for check, where, msg in group:
            print(f"- [{check}] {where}: {msg}")
    raise SystemExit(1 if in_diff else 0)


if __name__ == "__main__":
    main()
