import json
from pathlib import Path

import pytest

TRUTH_DIR = Path(__file__).resolve().parents[2] / "truth"


@pytest.fixture(scope="session")
def truth():
    def load(name: str) -> dict:
        return json.loads((TRUTH_DIR / f"{name}.json").read_text())

    return load
