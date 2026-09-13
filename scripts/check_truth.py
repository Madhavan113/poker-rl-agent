"""Validate the ground-truth files in truth/ and print a summary.

A truth file is JSON with: ``game``, ``status`` in {verified, corroborated, unverified},
``provenance`` (non-empty list of strings), ``tolerances`` (dict). Everything else is
game-specific and consumed by tests/ground_truth/.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

STATUSES = {"verified", "corroborated", "unverified"}
REQUIRED = ("game", "status", "provenance", "tolerances")


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"{path}: invalid JSON: {exc}"]
    for key in REQUIRED:
        if key not in data:
            errors.append(f"{path}: missing key {key!r}")
    if data.get("status") not in STATUSES:
        errors.append(f"{path}: status must be one of {sorted(STATUSES)}")
    prov = data.get("provenance")
    if not isinstance(prov, list) or not prov or not all(isinstance(p, str) and p for p in prov):
        errors.append(f"{path}: provenance must be a non-empty list of strings")
    if not isinstance(data.get("tolerances"), dict):
        errors.append(f"{path}: tolerances must be a dict")
    for item in data.get("hand_computed_best_responses", []):
        for key in ("name", "opponent_theta", "br_seat", "value", "derivation"):
            if key not in item:
                errors.append(f"{path}: hand-computed BR {item.get('name', '?')!r} missing {key!r}")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "truth").glob("*.json"))
    if not files:
        print("no truth files found", file=sys.stderr)
        return 1
    errors: list[str] = []
    print(f"{'file':<20} {'game':<8} {'status':<13} sha256[:12]")
    for path in files:
        errors.extend(validate(path))
        data = json.loads(path.read_text()) if not errors else {}
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        print(f"{path.name:<20} {data.get('game', '?'):<8} {data.get('status', '?'):<13} {digest}")
    for err in errors:
        print(err, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
