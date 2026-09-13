"""CFR+ (regret matching+, alternating updates, linear averaging) on a compiled game tree."""

from __future__ import annotations

import numpy as np

from exsolver.games.base import N_ACTIONS, Game
from exsolver.games.tree import CompiledTree, compile_tree
from exsolver.strategy import TabularStrategy


def regret_matching_plus(regrets: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """Strategy proportional to positive regrets; uniform over legal actions when all are zero."""
    pos = np.where(legal, np.maximum(regrets, 0.0), 0.0)
    tot = pos.sum(axis=1, keepdims=True)
    uniform = legal / legal.sum(axis=1, keepdims=True)
    return np.where(tot > 0.0, pos / np.where(tot > 0.0, tot, 1.0), uniform)


class CFRPlus:
    """Incremental CFR+ solver; call ``run`` then ``average_strategy``.

    Each iteration ``t`` updates the players alternately: traverse with the current profile,
    accumulate ``t * own_reach * sigma`` into the average (linear averaging), add the
    counterfactual regrets and floor them at zero (regret matching+).
    """

    def __init__(self, game: Game) -> None:
        self.game = game
        self.tree: CompiledTree = compile_tree(game)
        tree = self.tree
        self.legal = tree.infoset_legal.astype(np.float64)
        self.regrets = np.zeros((tree.n_infosets, N_ACTIONS))
        self.cum_strategy = np.zeros((tree.n_infosets, N_ACTIONS))
        self.sigma = tree.uniform_dense()
        self.iteration = 0
        self._rows = tree.infoset_rows
        self._dec = tree.dec_nodes
        self._dec_inf = tuple(tree.infoset[d] for d in tree.dec_nodes)
        self._signs = (1.0, -1.0)

    def step(self) -> None:
        """One CFR+ iteration (both players, alternating)."""
        tree, sigma = self.tree, self.sigma
        self.iteration += 1
        t = float(self.iteration)
        for p in (0, 1):
            rows = self._rows[p]
            rho = tree.realization(sigma)
            cf = tree.reach_chance * tree.reach(rho, 1 - p)
            u = tree.values(sigma)
            dec, inf = self._dec[p], self._dec_inf[p]
            gain = (u[tree.child[dec]] - u[dec][:, None]) * (self._signs[p] * cf[dec])[:, None]
            inst = np.empty((tree.n_infosets, N_ACTIONS))
            for a in range(N_ACTIONS):
                inst[:, a] = np.bincount(inf, weights=gain[:, a], minlength=tree.n_infosets)
            own_reach = rho[tree.infoset_parent_seq[rows]]
            self.cum_strategy[rows] += (t * own_reach)[:, None] * sigma[rows]
            # illegal actions carry no regret (their "child" is the zero sentinel node)
            self.regrets[rows] = np.maximum(self.regrets[rows] + inst[rows] * self.legal[rows], 0.0)
            sigma[rows] = regret_matching_plus(self.regrets[rows], self.legal[rows])

    def run(self, iterations: int) -> CFRPlus:
        for _ in range(iterations):
            self.step()
        return self

    def average_dense(self) -> np.ndarray:
        """Normalised average strategy as an [I, 3] array (uniform where never reached)."""
        tot = self.cum_strategy.sum(axis=1, keepdims=True)
        uniform = self.legal / self.legal.sum(axis=1, keepdims=True)
        return np.where(tot > 0.0, self.cum_strategy / np.where(tot > 0.0, tot, 1.0), uniform)

    def average_strategy(self) -> TabularStrategy:
        return self.tree.strategy_from_dense(self.average_dense())

    def current_strategy(self) -> TabularStrategy:
        return self.tree.strategy_from_dense(self.sigma)


def cfr_plus(game: Game, iterations: int) -> TabularStrategy:
    """Average strategy of both seats after ``iterations`` CFR+ iterations."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    return CFRPlus(game).run(iterations).average_strategy()
