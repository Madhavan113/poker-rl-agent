"""Agents that play a fixed profile: equilibrium, oracle best response, uniform random."""

from __future__ import annotations

import numpy as np

from exsolver.agents.base import check_seat
from exsolver.data.records import HandRecord, SessionContext
from exsolver.games.base import Game
from exsolver.solvers.best_response import best_response
from exsolver.solvers.cfr import cfr_plus
from exsolver.strategy import TabularStrategy, merge_seats, seat_entries, uniform_strategy


class FixedStrategyAgent:
    """Plays ``profile`` (entries for both seats) regardless of context; no belief."""

    def __init__(self, profile: TabularStrategy, name: str) -> None:
        self.profile = profile
        self.name = name

    def reset(self, rng: np.random.Generator) -> None:
        pass

    def observe(self, hand: HandRecord) -> None:
        pass

    def strategy_for_hand(self, ctx: SessionContext, seat: int) -> TabularStrategy:
        return {k: v.copy() for k, v in seat_entries(self.profile, check_seat(seat)).items()}

    def belief(self, ctx: SessionContext) -> np.ndarray | None:
        return None


class EquilibriumAgent(FixedStrategyAgent):
    """CFR+ average strategy, computed once."""

    def __init__(self, game: Game, iterations: int = 2000, name: str = "equilibrium") -> None:
        super().__init__(cfr_plus(game, iterations), name)
        self.iterations = iterations


class OracleBRAgent(FixedStrategyAgent):
    """Exact best response to the true opponent profile at both seats (the EV upper bound).

    This is the **only** agent that is handed the opponent's profile.
    """

    def __init__(self, game: Game, opp_profile: TabularStrategy, name: str = "oracle_br") -> None:
        br0, v0 = best_response(game, opp_profile, 0)
        br1, v1 = best_response(game, opp_profile, 1)
        super().__init__(merge_seats(br0, br1), name)
        self.values: tuple[float, float] = (v0, v1)  # BR value for the agent at seat 0 / seat 1


class RandomAgent(FixedStrategyAgent):
    """Uniform over legal actions (data collection only)."""

    def __init__(self, game: Game, name: str = "random") -> None:
        super().__init__(uniform_strategy(game), name)
