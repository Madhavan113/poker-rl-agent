#!/usr/bin/env bash
# Pre-submission check: format, lint, tests, and the ground-truth suite.
# Prints a PREFLIGHT block to paste into the PR. Exits non-zero on any failure.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

step "sync (frozen lock)";      uv sync --frozen --quiet
step "format check";            uv run ruff format --check src tests scripts
step "lint";                    uv run ruff check src tests scripts
step "tests";                   uv run pytest -q -m "not ground_truth"
step "truth files";             uv run python scripts/check_truth.py
step "ground-truth suite";      uv run pytest -q -m ground_truth -rs tests/ground_truth

commit=$(git rev-parse --short HEAD)
dirty=$(git status --porcelain --untracked-files=no | wc -l | tr -d ' ')
truth_hash=$(cat truth/*.json | shasum -a 256 | cut -c1-12)
printf '\nPREFLIGHT OK  commit=%s  dirty_tracked_files=%s  truth=%s  %s\n' \
  "$commit" "$dirty" "$truth_hash" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
