"""Root test configuration: engine-module gating, ``--forbid-skips`` and ground-truth placement.

The ground-truth suite (tests/ground_truth/) must never pass vacuously. Engine modules its tests
import may be missing only while they are listed in tests/ground_truth/pending_engine.txt; the
engine PR deletes those entries, after which a missing module fails collection and
scripts/ground_truth.sh adds ``--forbid-skips`` so that any remaining skip fails the run.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from types import ModuleType

import pytest

TESTS_DIR = Path(__file__).resolve().parent
GROUND_TRUTH_DIR = TESTS_DIR / "ground_truth"
PENDING_ENGINE_FILE = GROUND_TRUTH_DIR / "pending_engine.txt"


def pending_engine_modules() -> set[str]:
    """Dotted module names allowed to be missing: one per line, blank lines and ``#`` ignored."""
    if not PENDING_ENGINE_FILE.is_file():
        return set()
    names: set[str] = set()
    for raw in PENDING_ENGINE_FILE.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            names.add(line)
    return names


def require_module(name: str) -> ModuleType:
    """Import ``name`` and return it.

    On ImportError the calling test module is skipped (at module level) only if ``name`` is
    listed in tests/ground_truth/pending_engine.txt; otherwise the ImportError propagates and
    collecting that module fails.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        if name in pending_engine_modules():
            caller = inspect.stack()[1].filename
            pytest.skip(
                f"{Path(caller).name}: engine module {name!r} not importable yet; listed in "
                f"{PENDING_ENGINE_FILE.relative_to(TESTS_DIR.parent)}",
                allow_module_level=True,
            )
        raise


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--forbid-skips",
        action="store_true",
        default=False,
        help="turn every skipped test or module into a failure (scripts/ground_truth.sh adds this "
        "once tests/ground_truth/pending_engine.txt is empty or absent)",
    )


def _fail_if_skipped(
    report: pytest.TestReport | pytest.CollectReport, config: pytest.Config
) -> None:
    if not config.getoption("--forbid-skips") or not report.skipped:
        return
    if hasattr(report, "wasxfail"):  # an expected failure is not a skip
        return
    longrepr = report.longrepr
    reason = longrepr[2] if isinstance(longrepr, tuple) else str(longrepr)
    report.outcome = "failed"
    report.longrepr = f"skips are forbidden (--forbid-skips): {reason}"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    report = yield
    _fail_if_skipped(report, item.config)
    return report


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector: pytest.Collector):
    # Module-level skips (pytest.skip(allow_module_level=True), importorskip) surface here, not
    # in pytest_runtest_makereport.
    report = yield
    _fail_if_skipped(report, collector.config)
    return report


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Every test in tests/ground_truth/ carries the ``ground_truth`` marker and nothing else does.

    Runs before ``-m`` deselection so a misplaced test is caught however the suite is invoked.
    """
    problems: list[str] = []
    for item in items:
        if item.path is None:
            continue
        inside = Path(item.path).resolve().is_relative_to(GROUND_TRUTH_DIR)
        marked = item.get_closest_marker("ground_truth") is not None
        if marked and not inside:
            problems.append(
                f"{item.nodeid}: marked ground_truth but lives outside tests/ground_truth/"
            )
        elif inside and not marked:
            problems.append(
                f"{item.nodeid}: in tests/ground_truth/ but lacks the ground_truth marker"
            )
    if problems:
        raise pytest.UsageError("ground-truth placement check failed:\n  " + "\n  ".join(problems))
