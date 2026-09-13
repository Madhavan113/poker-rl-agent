"""The ``Agent`` protocol every agent (and the session runner) agrees on."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from exsolver.data.records import HandRecord, SessionContext
from exsolver.strategy import TabularStrategy


class Agent(Protocol):
    """An agent playing a session of hands against one unknown opponent.

    The runner calls ``reset`` once per session, then for every hand ``strategy_for_hand`` (and
    ``belief``) *before* the hand and ``observe`` *after* it. ``ctx.hands`` holds every completed
    hand of the session, oldest first, and is the only information about the opponent an agent
    may use.
    """

    name: str

    def reset(self, rng: np.random.Generator) -> None:
        """Start of a session; ``rng`` is the agent's private source of randomness."""
        ...

    def observe(self, hand: HandRecord) -> None:
        """Called after every completed hand."""
        ...

    def strategy_for_hand(self, ctx: SessionContext, seat: int) -> TabularStrategy:
        """Full strategy over *all* of the agent's infosets at ``seat`` for the upcoming hand."""
        ...

    def belief(self, ctx: SessionContext) -> np.ndarray | None:
        """Posterior over the discrete population given ``ctx``, or None if the agent has none."""
        ...


def check_seat(seat: int) -> int:
    if seat not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {seat}")
    return int(seat)
