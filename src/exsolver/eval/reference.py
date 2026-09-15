"""Evaluator-side reference for the policy-level KL: ``pi*_t`` and the reach weights ``w_t``.

``BRActionReference`` is built by the runner from the **population** and read with the
**evaluator's exact posterior** (docs/experiments/e2-decision-relevant-inference.md, "New exact
quantities"). It is never handed to an agent: agents receive the population (their prior) and
the hands they played, nothing else, and the runner keeps this object on its own side.

For the agent's seat ``s`` and the posterior ``p_t`` before hand ``t``:

* ``pi*_t(a | I) = sum_i p_t(i) 1[BR(theta_i)(I) = a]`` at every infoset ``I`` of seat ``s``
  (``exsolver.agents.bayes.br_action_posterior`` over the cached exact best responses);
* ``w_t(I)`` is the probability of reaching ``I`` when the agent plays ``pi*_t`` as a mixed
  behavioural strategy against the posterior-mixture opponent ``mix(p_t)``
  (``solvers.mixture.StrategyMixer`` at the opponent's seat), normalised to sum to one over the
  infosets of seat ``s``::

      reach(I) = sum_{nodes n in I} P_chance(n) * P_own(n; pi*_t) * P_opp(n; mix(p_t))
      w_t(I)   = reach(I) / sum_J reach(J)

  ``P_chance`` is the deal (uniform over the agent's card, then the opponent's card), ``P_own``
  the product of ``pi*_t``'s probabilities of the agent's own actions on the path and ``P_opp``
  the same for the mixture opponent. Infosets with ``reach(I) = 0`` (a history ``pi*_t`` never
  produces, or one the mixture opponent never allows) get ``w_t(I) = 0`` and therefore do not
  enter the weighted KL; they still count in the unweighted mean. The normaliser is always
  positive: the seat's first decision points are reached with the card probability whatever the
  strategies. Before normalisation ``sum_J reach(J)`` is the expected number of decisions the
  agent takes per hand, so the weighted KL is "KL per decision, decisions weighted by how often
  ``pi*_t`` faces them".
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from exsolver.agents.base import check_seat
from exsolver.agents.bayes import BestResponseTable
from exsolver.games.base import Game
from exsolver.games.tree import CompiledTree, compile_tree
from exsolver.population.population import Population
from exsolver.solvers.mixture import StrategyMixer
from exsolver.strategy import TabularStrategy


def infoset_reach(tree: CompiledTree, sigma: np.ndarray, seat: int) -> np.ndarray:
    """Unnormalised reach probability of every infoset of ``seat`` under the profile ``sigma``.

    ``sigma`` is dense ``[n_infosets, 3]`` with both seats' rows filled. Returns ``[I_seat]`` in
    enumeration order (``tree.infoset_rows[seat]``): ``sum_{n in I} reach_chance(n) *
    reach_0(n) * reach_1(n)``.
    """
    seat = check_seat(seat)
    rho = tree.realization(np.asarray(sigma, dtype=np.float64))
    nodes = tree.dec_nodes[seat]
    reach = tree.reach_chance[nodes] * rho[tree.node_seq[0][nodes]] * rho[tree.node_seq[1][nodes]]
    per_infoset = np.bincount(tree.infoset[nodes], weights=reach, minlength=tree.n_infosets)
    return per_infoset[tree.infoset_rows[seat]]


class BRActionReference:
    """``policy_target(seat, probs) -> (pi_star [I_seat, 3], weights [I_seat])`` (see module doc)."""

    def __init__(self, game: Game, population: Population | Sequence[TabularStrategy]) -> None:
        profiles = list(population.profiles if isinstance(population, Population) else population)
        self.game = game
        self.tree = compile_tree(game)
        self.br_table = BestResponseTable(game, profiles)
        self._mixers = (StrategyMixer(game, profiles, 0), StrategyMixer(game, profiles, 1))

    @property
    def m(self) -> int:
        return self.br_table.m

    def action_posterior(self, probs: np.ndarray, seat: int) -> np.ndarray:
        """``pi*`` rows ``[I_seat, 3]`` of ``seat`` under the posterior ``probs``."""
        seat = check_seat(seat)
        return self.br_table.action_posterior(probs)[self.tree.infoset_rows[seat]]

    def action_posterior_strategy(self, probs: np.ndarray, seat: int) -> TabularStrategy:
        """``pi*`` at ``seat`` as a tabular (mixed) strategy."""
        return self.br_table.action_posterior_for_seat(probs, seat)

    def weights(self, probs: np.ndarray, pi_star: np.ndarray, seat: int) -> np.ndarray:
        """``w_t`` over the infosets of ``seat``: reach under ``(pi_star, mix(probs))``, sum 1."""
        seat = check_seat(seat)
        sigma = np.zeros((self.tree.n_infosets, 3))
        sigma[self.tree.infoset_rows[seat]] = pi_star
        sigma[self.tree.infoset_rows[1 - seat]] = self._mixers[1 - seat].mix_dense(probs)
        reach = infoset_reach(self.tree, sigma, seat)
        total = float(reach.sum())
        if not total > 0.0:  # cannot happen for a well-formed game (root decisions are reached)
            raise RuntimeError("no infoset of the seat is reachable")
        return reach / total

    def policy_target(self, seat: int, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(pi*_t rows, w_t)`` for ``seat`` given the exact posterior ``probs`` (both ``[I_seat, ...]``)."""
        p = np.asarray(probs, dtype=np.float64).reshape(-1)
        if p.shape[0] != self.m:
            raise ValueError(f"posterior has {p.shape[0]} entries but the population has {self.m}")
        pi_star = self.action_posterior(p, seat)
        return pi_star, self.weights(p, pi_star, seat)
