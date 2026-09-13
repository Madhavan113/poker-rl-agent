#!/usr/bin/env bash
# Pre-submission check: locked sync, format, lint, regular tests, then the ground-truth gate
# (scripts/ground_truth.sh). Prints one PREFLIGHT line to paste into the PR body; the ground-truth
# CI job recomputes the commit and truth hash from the PR head and fails if the line is stale.
# Exits non-zero on any failure, and prints PREFLIGHT DIRTY instead of OK when the tree has
# uncommitted or untracked paths (the results would not describe the commit named in the line).
set -euo pipefail
export UV_LOCKED=1
cd "$(git rev-parse --show-toplevel)"

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

step "sync (locked)";        uv sync --locked --quiet
step "format check";         uv run ruff format --check src tests scripts
step "lint";                 uv run ruff check src tests scripts
step "tests";                uv run pytest -q -m "not ground_truth" --ignore=tests/ground_truth
step "ground-truth gate";    scripts/ground_truth.sh

commit=$(git rev-parse HEAD | cut -c1-12)
truth_hash=$(uv run python scripts/check_truth.py --hash)

truth_vs_main=unknown
if git remote get-url origin >/dev/null 2>&1; then
  git fetch --quiet origin main \
    || echo "warning: could not fetch origin/main; comparing against the local remote-tracking ref"
  if git rev-parse --verify --quiet origin/main >/dev/null; then
    if git diff --quiet origin/main -- truth/ tests/ground_truth/; then
      truth_vs_main=unchanged
    else
      truth_vs_main=CHANGED
    fi
  fi
fi

n_dirty=$(git status --porcelain | wc -l | tr -d ' ')
stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
if [ "$n_dirty" -ne 0 ]; then
  printf '\nPREFLIGHT DIRTY (%s uncommitted/untracked paths)  commit=%s  truth=%s  truth_vs_main=%s  %s\n' \
    "$n_dirty" "$commit" "$truth_hash" "$truth_vs_main" "$stamp"
  exit 1
fi
printf '\nPREFLIGHT OK  commit=%s  truth=%s  truth_vs_main=%s  %s\n' \
  "$commit" "$truth_hash" "$truth_vs_main" "$stamp"
