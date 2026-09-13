"""Shared helpers for the game-engine and solver tests."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from exsolver.games.base import N_ACTIONS, Game, State
from exsolver.strategy import TabularStrategy, enumerate_infosets, seat_of_key


def walk(game: Game, s: State | None = None) -> Iterator[State]:
    """Every state of the game tree in depth-first order."""
    stack = [game.root() if s is None else s]
    while stack:
        s = stack.pop()
        yield s
        if game.is_terminal(s):
            continue
        if game.is_chance(s):
            stack.extend(s2 for s2, _ in game.chance_outcomes(s))
        else:
            stack.extend(game.apply(s, a) for a in game.legal_actions(s))


def random_strategy(
    game: Game,
    rng: np.random.Generator,
    seats: tuple[int, ...] = (0, 1),
    concentration: float = 1.0,
) -> TabularStrategy:
    """Dirichlet-random strategy over legal actions at every infoset of ``seats``."""
    out: TabularStrategy = {}
    for key, legal in enumerate_infosets(game).items():
        if seat_of_key(key) in seats:
            v = np.zeros(N_ACTIONS)
            v[legal] = rng.dirichlet(np.full(len(legal), concentration))
            out[key] = v
    return out


def pure_strategy(game: Game, choose, seats: tuple[int, ...] = (0, 1)) -> TabularStrategy:
    """One-hot strategy: ``choose(key, legal) -> action`` at every infoset of ``seats``."""
    out: TabularStrategy = {}
    for key, legal in enumerate_infosets(game).items():
        if seat_of_key(key) in seats:
            v = np.zeros(N_ACTIONS)
            a = choose(key, legal)
            assert a in legal, (key, a, legal)
            v[a] = 1.0
            out[key] = v
    return out
