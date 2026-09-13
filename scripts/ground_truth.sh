#!/usr/bin/env bash
# Ground-truth gate: the single command run by CI (.github/workflows/ci.yml, job ground-truth)
# and by scripts/preflight.sh. Validates truth/*.json, then runs every test marked ground_truth
# with NO path argument, so a marked test placed outside tests/ground_truth/ is still collected
# (and rejected by the placement check in tests/conftest.py).
#
# While tests/ground_truth/pending_engine.txt lists engine modules that do not exist yet, the
# tests importing them skip (shown by -rs). Once that file is empty or absent, --forbid-skips
# turns any skip into a failure, so the suite cannot pass vacuously.
set -euo pipefail
export UV_LOCKED=1
cd "$(git rev-parse --show-toplevel)"

pending=tests/ground_truth/pending_engine.txt
uv run python scripts/check_truth.py

pytest_args=(-q -m ground_truth -rs)
if [ -f "$pending" ] && grep -qvE '^[[:space:]]*(#|$)' "$pending"; then
  echo "pending engine modules ($pending); tests importing them may skip:"
  grep -vE '^[[:space:]]*(#|$)' "$pending" | sed 's/^/  /'
else
  echo "no pending engine modules: running with --forbid-skips"
  pytest_args+=(--forbid-skips)
fi
uv run pytest "${pytest_args[@]}"
