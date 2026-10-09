---
name: review-content
description: Review a brainchip_devhub branch, PR or path against the content Definition of done (generated-README sync, number provenance, Akida 1 vs 2 correctness, dataset licensing, links, house style) and report findings. Use before opening any content PR, when asked to review a PR or example, and from the GitHub Action on content PRs. Does not edit files unless --fix is passed.
---

# Review content

Review content changes the way the repo owner would, so their review is a final check
rather than a first pass. The standard is the **Definition of done** in the root
`CLAUDE.md`. Read it first.

## Usage

```
/review-content [<PR number> | <branch> | <path>] [--base <ref>] [--fix]
```

- No target: the current branch against `origin/main` (including uncommitted changes).
- PR number: `gh pr checkout <n>` first (or `gh pr diff <n>` if checkout isn't possible),
  then review against the PR's base branch.
- Path: review that example or file as it stands, even if unchanged. This is useful for audits.
- `--fix`: after reporting, apply fixes that are mechanical and unambiguous (regenerate
  READMEs, fix links, bump badges). Never change numbers, prose claims or `metrics.json`.

## Step 1: Scope

List the changed files (`git diff --name-only <base>...HEAD` plus uncommitted changes) and
group them:

| Group | Examples |
| --- | --- |
| Example code | `<example>_*.py`, `*.sh`, `colab_setup.py` |
| Source of numbers | `docs/metrics.json`, `docs/zoo_card.json`, `pretrained_models/*` |
| Templates | `docs/README.md.template`, `akida1/docs/README.md.template` |
| Generated | example `README.md`, `akida1/README.md` |
| Notebooks | `*_notebook_*.ipynb` |
| Landing / nav | `README.md`, `akida*/README.md` |
| Tooling | `test/`, `.github/`, `brainchip_utils/`, `.claude/` |

Then read the linked issue (`gh pr view --json body,closingIssuesReferences`, or the branch
name) and pull out its acceptance criteria. Those are part of the review.

## Step 2: Mechanical checks

```bash
python .claude/skills/review-content/check_content.py --base <base>
```

The script is read-only. CI runs the same script on every PR (`content-checks` workflow).
Findings under "In this change" are blocking. Mention "Pre-existing" ones only if they're
in files this change touches.

Also check, by hand:

- **Generated READMEs edited directly.** A generated `README.md` changed but its
  template/metrics didn't: blocking, even if the drift check passes.
- **New weights.** New or changed `pretrained_models/*` files must come with updated
  `metrics.json`. CI (`models-float`, `models-hardware`) must be green; check with
  `gh pr checks`.
- **New example wiring.** The example is linked from the root `README.md` platform table;
  Akida 1 examples have `docs/zoo_card.json` with a valid `domain`; the README has
  "Requirements", "Reference Models" and "Contributing and Maintenance" sections that link
  to the top-level README rather than repeating it.
- **Notebooks.** The Colab badge URL path matches the notebook's location on `main`. The
  benchmark notebook is committed *with* outputs from a hardware run. Training notebooks
  need not be. Notebooks executed headlessly should go through
  `python -m brainchip_utils.notebooks`; a bare nbclient run leaves split progress-bar
  output (the `notebook-noise` check).

## Step 3: Editorial review

Read the changed prose and notebooks as the target reader would: an ML engineer who knows
Keras but is new to Akida.

1. **Every number has provenance.** For each accuracy, latency, power, energy, sparsity or
   parameter count in prose (not just in templated fields), find where it comes from:
   `metrics.json`, a cited paper, or a figure generated from metrics. A hard-coded number
   that duplicates a metrics key is a should-fix: replace it with the `{key}`
   placeholder. An untraceable number is blocking.
2. **Platform correctness.** Check claims about hardware behaviour against
   `.claude/knowledge/akida.md`. Akida 1 text must not describe Akida 2 behaviour or the reverse
   (8-bit vs 4-bit defaults, quantizeml vs cnn2snn, fused vs split separable convs,
   HWPR is available on AKD1500 but not AKD1000). Hardware is named wherever a number appears.
3. **Claims match code.** The README's pipeline table (epochs, learning rates, bit widths)
   matches the `.sh` / training script. Commands shown actually exist with those flags.
4. **Dataset.** Licence and citation are stated; setup instructions work from a clean clone;
   nothing non-redistributable is committed.
5. **Reader fit.** Explains the Akida-specific *why*, skips generic ML tutorials, is runnable
   top to bottom, and points to existing sections instead of duplicating them. Follows the
   Style section of `CLAUDE.md`.
6. **Acceptance criteria.** Each criterion from the issue is met, unmet, or can't be
   verified here.

## Step 4: Say what you could not verify

Be explicit. Unless training was run as part of this change (with its logs recorded in
the PR), Claude cannot confirm that it reproduces the published accuracy. Nor can it confirm
that hardware numbers were measured as described. List these under "Needs human
verification" with what the reviewer should check, rather than implying they passed.

## Output

Post one report (as a PR comment when run from the GitHub Action, otherwise in the
terminal):

```markdown
## Content review: <target>

**Verdict:** Ready | Changes needed | Needs human verification only

### Blocking
- `path:line`: problem → what would fix it

### Should fix
- ...

### Nits
- ...

### Acceptance criteria
- [x] criterion (evidence)
- [ ] criterion (what's missing)

### Needs human verification
- ...

<sub>Mechanical checks: N in this change, M pre-existing (not shown).</sub>
```

Keep findings concrete and anchored to a file and line. Leave out praise, restated diffs,
and nits that the house style doesn't actually require. If there are no findings, say so in
one line.
