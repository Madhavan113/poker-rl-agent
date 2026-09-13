"""Validate the ground-truth files in truth/ and print a summary, or with ``--hash`` one digest.

A truth file is JSON with ``game``, ``status`` (exactly ``verified`` or ``unverified``),
``provenance`` (non-empty list of strings) and ``tolerances`` (dict). Everything else is
game-specific and consumed by tests/ground_truth/.

Schema rule: an ``unverified`` file must not contain ``game_value_seat0`` or
``hand_computed_best_responses``, because the ground-truth tests enforce those numerically; a
number the code is held to must have verified provenance.

``--hash`` prints sha256[:12] over the sorted concatenation of truth/*.json. It is the only kind of
truth hash in this repo: the ``truth=`` field of the PREFLIGHT line, the value CI recomputes, and
the table's per-file digests all come from :func:`truth_digest`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

STATUSES = ("verified", "unverified")
REQUIRED = ("game", "status", "provenance", "tolerances")
VERIFIED_ONLY = ("game_value_seat0", "hand_computed_best_responses")
BR_KEYS = ("name", "opponent_theta", "br_seat", "value", "derivation")
ROOT = Path(__file__).resolve().parents[1]
TRUTH_DIR = ROOT / "truth"


def truth_files() -> list[Path]:
    return sorted(TRUTH_DIR.glob("*.json"))


def truth_digest(paths: list[Path]) -> str:
    """sha256[:12] of the bytes of ``paths`` concatenated in sorted order."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def validate(path: Path) -> tuple[dict, list[str]]:
    """Return ``(data, errors)`` for one truth file; ``data`` is ``{}`` if it could not be parsed."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"{path.name}: invalid JSON: {exc}"]
    if not isinstance(data, dict):
        return {}, [f"{path.name}: top level must be a JSON object"]
    errors: list[str] = []
    for key in REQUIRED:
        if key not in data:
            errors.append(f"{path.name}: missing key {key!r}")
    status = data.get("status")
    if status not in STATUSES:
        errors.append(f"{path.name}: status must be one of {list(STATUSES)}, got {status!r}")
    prov = data.get("provenance")
    if not isinstance(prov, list) or not prov or not all(isinstance(p, str) and p for p in prov):
        errors.append(f"{path.name}: provenance must be a non-empty list of strings")
    if not isinstance(data.get("tolerances"), dict):
        errors.append(f"{path.name}: tolerances must be a dict")
    if status == "unverified":
        for key in VERIFIED_ONLY:
            if key in data:
                errors.append(
                    f"{path.name}: status is 'unverified' but {key!r} is present; tests enforce it "
                    "numerically, so it needs verified provenance (or must be removed)"
                )
    for item in data.get("hand_computed_best_responses", []):
        if not isinstance(item, dict):
            errors.append(f"{path.name}: hand-computed BR entries must be objects, got {item!r}")
            continue
        for key in BR_KEYS:
            if key not in item:
                errors.append(
                    f"{path.name}: hand-computed BR {item.get('name', '?')!r} missing {key!r}"
                )
    return data, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate truth/*.json.")
    parser.add_argument(
        "--hash",
        action="store_true",
        help="print only sha256[:12] over the sorted concatenation of truth/*.json",
    )
    args = parser.parse_args(argv)
    files = truth_files()
    if not files:
        print("no truth files found", file=sys.stderr)
        return 1
    if args.hash:
        print(truth_digest(files))
        return 0
    errors: list[str] = []
    print(f"{'file':<20} {'game':<8} {'status':<12} sha256[:12]")
    for path in files:
        data, file_errors = validate(path)
        errors.extend(file_errors)
        game = str(data.get("game", "?"))
        status = str(data.get("status", "?"))
        print(f"{path.name:<20} {game:<8} {status:<12} {truth_digest([path])}")
    print(f"{'all (sorted concat)':<20} {'':<8} {'':<12} {truth_digest(files)}")
    for err in errors:
        print(err, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
