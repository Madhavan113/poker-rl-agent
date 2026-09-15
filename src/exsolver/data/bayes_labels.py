"""Bayes-BR labels for the DPT generator (E2 condition E): best response to the posterior mixture.

``labels="oracle_br"`` (the E1 recipe, RESEARCH.md 2.5) labels every agent decision with
``BR(theta)(I)`` for the *true* sampled opponent. ``labels="bayes_br"`` replaces that with the
myopic Bayes-optimal action: with ``p_t`` the exact posterior over the discrete population given
the ``t`` completed hands of the session (``exsolver.bayes.ExactPosterior``, prior = the
population weights, exactly the object the evaluator builds), the label at hand ``t`` is ::

    BR( mix(p_t, population profiles at the opponent's seat), agent seat )(I)

where ``mix`` is the realization-equivalent behavioural mixture (``StrategyMixer``) and ``BR``
the exact solver (``exsolver.solvers.best_response``, ties to the lowest index). This is the
action ``BayesBRAgent`` would take at ``I`` with the same context, so a model trained on these
labels has ``BayesBRAgent`` as its exact reference.

Information flow, by construction: ``BayesBRLabeller`` is built from the *population* only and
is updated with completed ``HandRecord``s, the agent-side observations of the hands. Neither the
sampled opponent's id nor its theta is an input anywhere in this module; the label depends on the
true opponent only through what the hands reveal about it.

Cost: the posterior, the mixture and its best response change once per completed hand, never
per decision, so ``label_table`` is computed at most once per (hand, seat) and cached until the
next ``observe``. Kuhn, ``M = 256``: about 80 us per hand (update 20, mix 20, BR 40).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from exsolver.bayes.posterior import ExactPosterior
from exsolver.data.records import HandRecord
from exsolver.games.base import Game
from exsolver.games.tree import compile_tree
from exsolver.population.population import Population
from exsolver.solvers.best_response import best_response
from exsolver.solvers.mixture import StrategyMixer
from exsolver.strategy import TabularStrategy


class BayesBRLabeller:
    """Labels of the ``bayes_br`` variant along one session; reusable across sessions via ``reset``.

    ``population`` is a ``Population`` (prior = its ``weights``) or a sequence of full profiles
    (uniform prior), optionally overridden by ``prior_weights`` -- the same constructor contract
    as ``ExactPosterior`` / ``BayesBRAgent``.
    """

    def __init__(
        self,
        game: Game,
        population: Population | Sequence[TabularStrategy],
        prior_weights: np.ndarray | None = None,
    ) -> None:
        profiles: list[TabularStrategy] = list(
            population.profiles if isinstance(population, Population) else population
        )
        self.game = game
        self.tree = compile_tree(game)
        self.posterior = ExactPosterior(game, population, prior_weights)
        # indexed by the *opponent's* seat
        self._mixers = (StrategyMixer(game, profiles, 0), StrategyMixer(game, profiles, 1))
        self._tables: dict[int, np.ndarray] = {}  # agent seat -> BR table for the current posterior
        self._probs: np.ndarray | None = None

    @property
    def m(self) -> int:
        return self.posterior.m

    @property
    def n_hands(self) -> int:
        """Number of completed hands the current posterior is conditioned on."""
        return self.posterior.n_hands

    @property
    def probs(self) -> np.ndarray:
        """Posterior probabilities ``[M]`` given the observed hands (a copy)."""
        return self.posterior.probs

    def reset(self) -> None:
        """Start a new session: back to the prior, caches dropped."""
        self.posterior.reset()
        self._invalidate()

    def observe(self, hand: HandRecord) -> None:
        """Condition on one completed hand (the agent's observations only)."""
        self.posterior.update(hand)
        self._invalidate()

    def observe_many(self, hands: Sequence[HandRecord]) -> None:
        for hand in hands:
            self.observe(hand)

    def _invalidate(self) -> None:
        self._tables.clear()
        self._probs = None

    def _current_probs(self) -> np.ndarray:
        if self._probs is None:
            self._probs = self.posterior.probs
        return self._probs

    def mixture(self, opp_seat: int) -> TabularStrategy:
        """Posterior-mixture opponent strategy at ``opp_seat`` given the observed hands."""
        if opp_seat not in (0, 1):
            raise ValueError(f"opp_seat must be 0 or 1, got {opp_seat}")
        return self._mixers[opp_seat].mix(self._current_probs())

    def label_table(self, seat: int) -> np.ndarray:
        """``[n_infosets]`` int8 table of Bayes-BR actions at ``seat``'s infosets (-1 elsewhere).

        Rows are ``compile_tree(game).infoset_index`` ids. Cached until the next ``observe`` /
        ``reset``; the returned array is shared, do not modify it.
        """
        if seat not in (0, 1):
            raise ValueError(f"seat must be 0 or 1, got {seat}")
        table = self._tables.get(seat)
        if table is None:
            br, _ = best_response(self.game, self.mixture(1 - seat), seat)
            table = np.full(self.tree.n_infosets, -1, dtype=np.int8)
            index = self.tree.infoset_index
            for key, row in br.items():
                table[index[key]] = int(np.argmax(row))  # one-hot rows: argmax is the BR action
            self._tables[seat] = table
        return table

    def label(self, seat: int, infoset_key: str) -> int:
        """Bayes-BR action for ``seat`` at ``infoset_key`` given the observed hands."""
        a = int(self.label_table(seat)[self.tree.infoset_index[infoset_key]])
        if a < 0:
            raise KeyError(f"infoset {infoset_key!r} does not belong to seat {seat}")
        return a


def bayes_br_label_table(
    game: Game,
    population: Population | Sequence[TabularStrategy],
    hands: Sequence[HandRecord],
    seat: int,
    prior_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Pure function form: the Bayes-BR label table for ``seat`` after observing ``hands``.

    Its inputs are the whole information budget of the ``bayes_br`` label: the population, the
    completed hands as the agent recorded them and the seat. Convenient for tests and one-off
    checks; the generator uses ``BayesBRLabeller`` incrementally.
    """
    labeller = BayesBRLabeller(game, population, prior_weights)
    labeller.observe_many(hands)
    return labeller.label_table(seat).copy()


__all__ = ["BayesBRLabeller", "bayes_br_label_table"]
