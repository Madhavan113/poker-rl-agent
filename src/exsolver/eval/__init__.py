"""Exact per-hand session metrics, aggregation over sessions and plots."""

from exsolver.eval.aggregate import aggregate
from exsolver.eval.session import (
    KUHN_PROBES,
    FixedStrategyAgent,
    SessionMetrics,
    entropy,
    kl_divergence,
    run_session,
)

__all__ = [
    "KUHN_PROBES",
    "FixedStrategyAgent",
    "SessionMetrics",
    "aggregate",
    "entropy",
    "kl_divergence",
    "run_session",
]
