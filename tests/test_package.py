"""Package smoke test; also guarantees the non-ground-truth suite is never empty."""

import exsolver


def test_package_imports():
    assert exsolver.__doc__
