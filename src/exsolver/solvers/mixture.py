"""Realization-equivalent behavioural mixture of strategies (sequence-form average)."""

from __future__ import annotations

import numpy as np

from exsolver.games.base import Game
from exsolver.games.tree import CompiledTree, compile_tree
from exsolver.strategy import TabularStrategy


class StrategyMixer:
    """Pre-computes the ``seat`` entries and own reach of ``profiles`` so ``mix`` is cheap.

    Use this when mixing the same population repeatedly with changing weights (a posterior).
    """

    def __init__(self, game: Game, profiles: list[TabularStrategy], seat: int) -> None:
        if not profiles:
            raise ValueError("need at least one profile")
        self.tree: CompiledTree = compile_tree(game)
        self.seat = seat
        self.rows = self.tree.infoset_rows[seat]
        theta = np.stack([self.tree.dense_from_strategy(p, (seat,)) for p in profiles])  # [M,I,3]
        self.theta = theta[:, self.rows]  # [M, I_seat, 3]
        rho = self.tree.realization_batch(theta)  # [M, S]
        self.own_reach = rho[:, self.tree.infoset_parent_seq[self.rows]]  # [M, I_seat]

    def mix_dense(self, weights: np.ndarray) -> np.ndarray:
        """[I_seat, 3] mixture at this seat's infosets (enumeration order)."""
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != self.theta.shape[0]:
            raise ValueError("one weight per profile expected")
        if np.any(w < 0.0) or w.sum() <= 0.0:
            raise ValueError("weights must be non-negative with a positive sum")
        w = w / w.sum()
        wr = w[:, None] * self.own_reach  # [M, I_seat]
        num = np.einsum("mi,mia->ia", wr, self.theta)
        den = wr.sum(axis=0)
        fallback = np.einsum("m,mia->ia", w, self.theta)  # plain average where unreachable
        reachable = den > 0.0
        out = np.where(reachable[:, None], num / np.where(reachable, den, 1.0)[:, None], fallback)
        return out

    def mix(self, weights: np.ndarray) -> TabularStrategy:
        keys = self.tree.infoset_keys
        dense = self.mix_dense(weights)
        return {keys[i]: dense[j] for j, i in enumerate(self.rows)}


def mix_strategies(
    game: Game, weights: np.ndarray, profiles: list[TabularStrategy], seat: int
) -> TabularStrategy:
    """Behavioural strategy for ``seat`` realization-equivalent to playing profile ``i`` w.p. ``w_i``.

    At each infoset ``I`` of ``seat``, ``sigma(a|I) = sum_i w_i r_i(I) theta_i(a|I) /
    sum_i w_i r_i(I)`` where ``r_i(I)`` is the product of ``theta_i``'s own action probabilities
    along the path to ``I``. Infosets no profile can reach get the plain weighted average. For
    any opponent strategy ``s``: ``expected_value(s, mix) == sum_i w_i expected_value(s, theta_i)``.
    Weights are normalised to sum to one.
    """
    return StrategyMixer(game, profiles, seat).mix(weights)
