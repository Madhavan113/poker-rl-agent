"""The truth files themselves are well-formed (engine independent)."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.ground_truth
ROOT = Path(__file__).resolve().parents[2]


def test_check_truth_script_passes():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_truth.py")], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_kuhn_truth_param_order_matches_spec(truth):
    data = truth("kuhn")
    assert data["param_order"] == [
        "0:J|", "0:Q|", "0:K|", "0:J|cb", "0:Q|cb", "0:K|cb",
        "1:J|c", "1:Q|c", "1:K|c", "1:J|b", "1:Q|b", "1:K|b",
    ]  # fmt: skip
    assert all(len(t["theta"]) == 12 for t in data["nash_family"]["thetas"])
    assert all(len(b["opponent_theta"]) == 12 for b in data["hand_computed_best_responses"])
