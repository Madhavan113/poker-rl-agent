"""Tabular strategies: infoset key -> length-3 probability vector over (FOLD, CALL, RAISE).

A *profile* is a strategy covering both seats' infosets; per-seat strategies only cover one
seat. Keys start with the seat (``"0:K|"``), so ``seat_of_key`` is a cheap split.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from exsolver.games.base import N_ACTIONS, Action, Game
from exsolver.games.tree import compile_tree

TabularStrategy = dict[str, np.ndarray]


def seat_of_key(key: str) -> int:
    """Seat that owns the infoset ``key`` (``"1:J|c"`` -> 1)."""
    return int(key.partition(":")[0])


@lru_cache(maxsize=8)
def _infoset_table(game: Game) -> tuple[tuple[str, tuple[Action, ...]], ...]:
    tree = compile_tree(game)
    return tuple(
        (key, tuple(int(a) for a in np.flatnonzero(tree.infoset_legal[i])))
        for i, key in enumerate(tree.infoset_keys)
    )


def enumerate_infosets(game: Game) -> dict[str, list[Action]]:
    """Every infoset key of both seats -> its legal actions (memoised; fixed depth-first order).

    The order matches ``compile_tree(game).infoset_keys`` and is the row order used by
    ``strategy_to_array`` / ``array_to_strategy``.
    """
    return {key: list(legal) for key, legal in _infoset_table(game)}


def seat_infosets(game: Game, seat: int) -> list[str]:
    """Infoset keys owned by ``seat`` in enumeration order."""
    return [key for key, _ in _infoset_table(game) if seat_of_key(key) == seat]


def uniform_strategy(game: Game, seats: tuple[int, ...] = (0, 1)) -> TabularStrategy:
    """Uniform-over-legal-actions strategy for the given seats."""
    out: TabularStrategy = {}
    for key, legal in _infoset_table(game):
        if seat_of_key(key) in seats:
            v = np.zeros(N_ACTIONS)
            v[list(legal)] = 1.0 / len(legal)
            out[key] = v
    return out


def strategy_to_array(
    game: Game, strat: TabularStrategy, seats: tuple[int, ...] = (0, 1)
) -> np.ndarray:
    """Dense ``[n_infosets, 3]`` array in enumeration order; rows of other seats are zero.

    Raises ``KeyError`` for a missing infoset and ``ValueError`` (naming the key) for rows with
    negative or non-finite entries, mass on illegal actions or a sum different from one.
    """
    return compile_tree(game).dense_from_strategy(strat, seats)


def array_to_strategy(
    game: Game, arr: np.ndarray, seats: tuple[int, ...] = (0, 1)
) -> TabularStrategy:
    """Inverse of ``strategy_to_array`` (copies the rows)."""
    return compile_tree(game).strategy_from_dense(np.asarray(arr, dtype=np.float64), seats)


def validate_strategy(game: Game, strat: TabularStrategy, seats: tuple[int, ...] = (0, 1)) -> None:
    """Raise if ``strat`` is not a valid strategy for ``seats`` (see ``strategy_to_array``)."""
    strategy_to_array(game, strat, seats)


def seat_entries(strat: TabularStrategy, seat: int) -> TabularStrategy:
    """The sub-dictionary of ``strat`` owned by ``seat``."""
    return {k: v for k, v in strat.items() if seat_of_key(k) == seat}


def merge_seats(seat0: TabularStrategy, seat1: TabularStrategy) -> TabularStrategy:
    """Profile made of ``seat0``'s seat-0 entries and ``seat1``'s seat-1 entries."""
    return {**seat_entries(seat0, 0), **seat_entries(seat1, 1)}
