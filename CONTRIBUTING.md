# Contributing

This repository is run like a production codebase even though it is research code. The reason is
simple: every number in `RESEARCH.md` will eventually be believed by someone, so the code that
produces it has to be right, reproducible, and reviewed.

## The workflow

1. **Branch from `main`.** Names: `feat/…`, `fix/…`, `exp/…` (experiment code), `infra/…`,
   `docs/…`. `main` accepts squash merges only, through pull requests only; the repository
   ruleset blocks direct pushes, force pushes and deletions for everyone, including admins.
2. **One concern per PR.** A game engine, a tokenizer, an experiment runner are separate PRs.
   Keep PRs under ~800 changed lines unless it is generated data or a spec.
3. **Run the preflight before every submission:** `scripts/preflight.sh`. It syncs the
   environment against the lockfile (`uv sync --locked`; a stale `uv.lock` fails), format-checks,
   lints, runs the regular test suite, then runs the **ground-truth gate** `scripts/ground_truth.sh`
   (`scripts/check_truth.py` plus every test marked `ground_truth`, values in `truth/`) and prints
   one line `PREFLIGHT OK  commit=<12 hex>  truth=<hash>  truth_vs_main=unchanged|CHANGED  <time>`.
   On a tree with uncommitted or untracked paths it prints `PREFLIGHT DIRTY (...)` and fails
   instead: the results would not describe the commit named in the line. Paste the `PREFLIGHT OK`
   line into the PR body. The `ground-truth` CI job fails unless the last such line has a
   `commit=` that is a prefix of the PR head and a `truth=` equal to
   `scripts/check_truth.py --hash` at that commit, so re-run the preflight on every head commit
   you push and update the PR body. The pre-push hook runs it automatically once you enable
   hooks: `git config core.hooksPath .githooks`.
4. **Open the PR with the template.** Fill every section. Reference the spec section you
   implemented; list deviations explicitly and add the `spec-deviation` label.
5. **Independent review.** Every PR is reviewed by someone (or some agent) who did not write it,
   following `docs/CODE_REVIEW.md`. Blocking findings must be fixed and re-reviewed. Approval is
   recorded by a PR comment containing the line `REVIEW-APPROVED <full 40-character head sha>`
   plus the label `review: approved` (with `review: changes requested` removed); the
   `review-gate` check reads the live labels and comments and passes only when all of that holds
   for the current head. Pushing new commits removes `review: approved` and fails the check until
   the review is redone.
6. **Merge** when the four required checks `lint`, `tests`, `ground-truth` and `review-gate` are
   green. Squash merge; the PR title becomes the commit subject, the PR body the commit body.

## Ground truth

`truth/*.json` holds values the code must reproduce that were derived **independently of the
code**: analytic results (Kuhn's 1950 solution), hand-computed best responses with the
derivation written out, and numbers from an external reference implementation with its version
recorded and the run's script and output committed under `truth/derivations/`. Each file carries
a `status` that is exactly `verified` or `unverified`, and provenance. `scripts/check_truth.py`
rejects an `unverified` file that contains `game_value_seat0` or `hand_computed_best_responses`:
tests enforce those numerically, so a value the code is held to must have verified provenance.
Changing a truth file is itself a reviewed event: add the `ground-truth` label (the `review-gate`
check requires it whenever `truth/` or `tests/ground_truth/` changes) and put the new derivation
in the PR. Never "fix" a truth value to match the code.

The engine modules the ground-truth tests import are listed in
`tests/ground_truth/pending_engine.txt` while they do not exist yet; a test module whose
`require_module(...)` (see `tests/conftest.py`) hits a listed module skips, visibly, instead of
failing. **The engine PR must delete the entries it implements** (and may delete the file once it
is empty). When the file is empty or absent, `scripts/ground_truth.sh` runs pytest with
`--forbid-skips`, which turns every skip into a failure, so the suite cannot pass vacuously once
the engine exists. A missing module that is not listed fails collection at any time.

## Trust model

The gate protects against accidents and process drift, not against the repository owner, who can
edit labels, comments, workflows and rulesets. Workflows run from the PR branch, so a PR can
change the very checks that judge it: any change under `.github/`, `scripts/`, `truth/` or
`tests/ground_truth/` is a blocking-review item that the reviewer reads line by line
(`docs/CODE_REVIEW.md`). The repository ruleset pins the four required checks (`lint`, `tests`,
`ground-truth`, `review-gate`) to the GitHub Actions app, so a same-named status reported by
anything else does not count, and allows squash merges only.

## Rules of thumb

- No new runtime dependencies without a sentence of justification in the PR.
- Determinism: every stochastic path takes a `numpy.random.Generator` or a torch seed; every
  reported number records its seed and config.
- Nothing under `runs/`, no data shards, no checkpoints in git.
- Commit messages end with the co-author trailer used in this repo's history.
