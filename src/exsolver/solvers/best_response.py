"""Exact best response, expected value and exploitability by tree traversal."""

from __future__ import annotations

import numpy as np

from exsolver.games.base import N_ACTIONS, Game
from exsolver.games.tree import compile_tree
from exsolver.strategy import TabularStrategy


def _cf_reach(game: Game, opponent: TabularStrategy, br_seat: int) -> tuple[np.ndarray, np.ndarray]:
    tree = compile_tree(game)
    sigma = tree.dense_from_strategy(opponent, (1 - br_seat,))
    rho = tree.realization(sigma)
    return sigma, tree.reach_chance * tree.reach(rho, 1 - br_seat)


def best_response_value(game: Game, opponent: TabularStrategy, br_seat: int) -> float:
    """Expected value (for ``br_seat``) of the exact best response to ``opponent``."""
    tree = compile_tree(game)
    sigma, cf = _cf_reach(game, opponent, br_seat)
    u, _ = tree.best_response_pass(sigma, br_seat, cf)
    return float(u[tree.root]) * (1.0 if br_seat == 0 else -1.0)


def best_response(
    game: Game, opponent: TabularStrategy, br_seat: int
) -> tuple[TabularStrategy, float]:
    """Exact pure best response of ``br_seat`` against ``opponent``'s other-seat entries.

    Returns the BR strategy (one-hot at every infoset of ``br_seat``) and its expected value for
    ``br_seat``. Ties resolve to the lowest-index maximising action; at infosets the opponent
    makes unreachable the BR plays CALL when legal, otherwise the lowest legal action.
    """
    tree = compile_tree(game)
    sigma, cf = _cf_reach(game, opponent, br_seat)
    u, a_star = tree.best_response_pass(sigma, br_seat, cf)
    rows = tree.infoset_rows[br_seat]
    br = np.zeros((tree.n_infosets, N_ACTIONS))
    br[rows, a_star[rows]] = 1.0
    value = float(u[tree.root]) * (1.0 if br_seat == 0 else -1.0)
    return tree.strategy_from_dense(br, (br_seat,)), value


def expected_value(game: Game, strat_seat0: TabularStrategy, strat_seat1: TabularStrategy) -> float:
    """Exact expected chips for seat 0 when seat 0 plays ``strat_seat0`` and seat 1 ``strat_seat1``."""
    tree = compile_tree(game)
    sigma = tree.dense_from_strategy(strat_seat0, (0,))
    sigma += tree.dense_from_strategy(strat_seat1, (1,))
    return float(tree.values(sigma)[tree.root])


def exploitability(game: Game, profile: TabularStrategy) -> float:
    """NashConv / 2: mean over seats of the best-response value against ``profile``; 0 at a Nash."""
    return 0.5 * (best_response_value(game, profile, 0) + best_response_value(game, profile, 1))


def seat_exploitability(
    game: Game, strat: TabularStrategy, seat: int, game_value_seat0: float
) -> float:
    """Gain of an omniscient opponent over their equilibrium value against ``strat`` at ``seat``."""
    if seat == 0:
        return best_response_value(game, strat, 1) + game_value_seat0
    return best_response_value(game, strat, 0) - game_value_seat0
