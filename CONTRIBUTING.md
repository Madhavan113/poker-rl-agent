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
3. **Run the preflight before every submission:** `scripts/preflight.sh`. It formats-checks,
   lints, runs the full test suite, then runs the **ground-truth suite** (`tests/ground_truth/`,
   values in `truth/`) and prints a `PREFLIGHT` block. Paste that block into the PR. The pre-push
   hook runs it automatically once you enable hooks: `git config core.hooksPath .githooks`.
4. **Open the PR with the template.** Fill every section. Reference the spec section you
   implemented; list deviations explicitly and add the `spec-deviation` label.
5. **Independent review.** Every PR is reviewed by someone (or some agent) who did not write it,
   following `docs/CODE_REVIEW.md`. Blocking findings must be fixed and re-reviewed. Approval is
   recorded by adding the label `review: approved`; pushing new commits removes it and fails the
   `review-gate` check until the review is redone.
6. **Merge** when `lint`, `tests`, `ground-truth` and `review-gate` are green. Squash merge; the
   PR title becomes the commit subject, the PR body the commit body.

## Ground truth

`truth/*.json` holds values the code must reproduce that were derived **independently of the
code**: analytic results (Kuhn's 1950 solution), hand-computed best responses with the
derivation written out, and numbers from an external reference implementation with its version
recorded. Each file carries a `status` (`verified`, `corroborated`, `unverified`) and provenance.
Changing a truth file is itself a reviewed event: add the `ground-truth` label and put the new
derivation in the PR. Never "fix" a truth value to match the code.

## Rules of thumb

- No new runtime dependencies without a sentence of justification in the PR.
- Determinism: every stochastic path takes a `numpy.random.Generator` or a torch seed; every
  reported number records its seed and config.
- Nothing under `runs/`, no data shards, no checkpoints in git.
- Commit messages end with the co-author trailer used in this repo's history.
