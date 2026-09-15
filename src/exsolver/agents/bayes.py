"""Exact-Bayes agents over a discrete population: Thompson sampling, Bayes best response, plurality BR.

All three share the exact posterior ``p_t`` over the ``M`` population members and a lazily
filled table of exact best responses ``BR(theta_i)`` (both seats, one per member). The
posterior over best-response *actions* (docs/experiments/e2-decision-relevant-inference.md)::

    pi*_t(a | I) = sum_i p_t(i) 1[BR(theta_i)(I) = a]

is the DPT limit of the transformer's action head. ``ThompsonAgent`` samples ``i ~ p_t`` and
plays ``BR(theta_i)`` (a sample from ``pi*_t`` at every infoset, correlated across infosets);
``PluralityBRAgent`` plays ``argmax_a pi*_t(a | I)`` (the exact analogue of Transformer(argmax));
``BayesBRAgent`` best-responds to the posterior mixture (myopic Bayes-optimal, not a function of
``pi*_t``). None of them sees anything but the population (their prior) and the hands played.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from exsolver.agents.base import check_seat
from exsolver.bayes.posterior import ExactPosterior
from exsolver.data.records import HandRecord, SessionContext
from exsolver.games.base import N_ACTIONS, Game
from exsolver.games.tree import compile_tree
from exsolver.population.population import Population
from exsolver.solvers.best_response import best_response
from exsolver.solvers.mixture import StrategyMixer
from exsolver.strategy import TabularStrategy, merge_seats, seat_entries

PROB_SUM_TOL = 1e-6


def br_action_posterior(br_onehot: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """``pi*(a | I) = sum_i probs[i] * br_onehot[i, I, a]`` -> ``[n_infosets, 3]``.

    Pure and vectorised. ``br_onehot`` is ``[M, n_infosets, 3]`` with every row a one-hot
    action (a pure best response at every infoset of both seats); ``probs`` is a length-``M``
    probability vector. Raises ``ValueError`` on a shape mismatch, on rows that are not one-hot
    or on ``probs`` that are not a probability vector (non-finite, negative, or not summing to
    one within ``PROB_SUM_TOL``).
    """
    br = np.asarray(br_onehot, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if br.ndim != 3 or br.shape[2] != N_ACTIONS:
        raise ValueError(f"br_onehot must have shape [M, n_infosets, {N_ACTIONS}], got {br.shape}")
    if br.shape[0] != p.shape[0]:
        raise ValueError(f"br_onehot has {br.shape[0]} members but probs has {p.shape[0]}")
    if not np.all(np.isfinite(p)) or np.any(p < 0.0) or abs(float(p.sum()) - 1.0) > PROB_SUM_TOL:
        raise ValueError(f"probs must be a probability vector, got {p}")
    if not np.all((br == 0.0) | (br == 1.0)) or not np.all(br.sum(axis=2) == 1.0):
        raise ValueError("every row of br_onehot must be a one-hot action")
    return np.einsum("m,mia->ia", p, br)


class BestResponseTable:
    """Lazily filled ``[M, n_infosets, 3]`` table of exact best responses to ``profiles``.

    ``BR(theta_i)`` at both seats is computed the first time member ``i`` is needed and kept
    (one-hot rows, enumeration order). ``action_posterior(probs)`` fills the rows of every
    member with positive mass and returns ``pi*`` for all infosets of both seats.
    """

    def __init__(self, game: Game, profiles: Sequence[TabularStrategy]) -> None:
        if len(profiles) == 0:
            raise ValueError("need at least one profile")
        self.game = game
        self.profiles = list(profiles)
        self.tree = compile_tree(game)
        self.dense = np.zeros((len(self.profiles), self.tree.n_infosets, N_ACTIONS))
        self.ready = np.zeros(len(self.profiles), dtype=bool)
        self._tabular: dict[int, TabularStrategy] = {}

    @property
    def m(self) -> int:
        return len(self.profiles)

    def ensure(self, ids: np.ndarray | Sequence[int]) -> None:
        """Compute (once) the best responses to the members in ``ids``."""
        for i in np.asarray(ids, dtype=np.int64).reshape(-1):
            i = int(i)
            if self.ready[i]:
                continue
            br0, _ = best_response(self.game, self.profiles[i], 0)
            br1, _ = best_response(self.game, self.profiles[i], 1)
            self.dense[i] = self.tree.dense_from_strategy(merge_seats(br0, br1))
            self.ready[i] = True

    def best_response_to(self, i: int) -> TabularStrategy:
        """Cached exact best response (both seats) to member ``i``; callers get a shared dict."""
        br = self._tabular.get(i)
        if br is None:
            self.ensure([i])
            br = self.tree.strategy_from_dense(self.dense[i])
            self._tabular[i] = br
        return br

    def action_posterior(self, probs: np.ndarray) -> np.ndarray:
        """``pi*(a | I)`` over every infoset of both seats -> ``[n_infosets, 3]``."""
        p = np.asarray(probs, dtype=np.float64).reshape(-1)
        if p.shape[0] != self.m:
            raise ValueError(f"expected {self.m} probabilities, got {p.shape[0]}")
        support = np.flatnonzero(p > 0.0)
        self.ensure(support)
        return br_action_posterior(self.dense[support], p[support])

    def action_posterior_for_seat(self, probs: np.ndarray, seat: int) -> TabularStrategy:
        """``pi*`` restricted to the infosets of ``seat`` as a (mixed) tabular strategy."""
        return self.tree.strategy_from_dense(self.action_posterior(probs), (check_seat(seat),))


class _PosteriorAgent:
    """Shared machinery: an ``ExactPosterior`` kept in sync with the session context.

    The posterior is conditioned on ``self._observed``, the hands it has been updated with, in
    order. ``observe`` is the fast path (one incremental update). ``strategy_for_hand`` and
    ``belief`` first check that ``ctx.hands`` *extends* the observed list -- element-wise
    identity, falling back to equality for equal copies -- and update on the new hands only; if
    the context diverges (fewer hands, or a different hand at some position) the posterior is
    recomputed from the prior over ``ctx.hands``. The belief is therefore always the exact
    posterior given exactly ``ctx.hands``, whatever the call order.
    """

    def __init__(
        self,
        game: Game,
        population: Population | Sequence[TabularStrategy],
        prior_weights: np.ndarray | None,
        name: str,
    ) -> None:
        self.game = game
        self.name = name
        self.profiles: list[TabularStrategy] = list(
            population.profiles if isinstance(population, Population) else population
        )
        self.posterior = ExactPosterior(game, population, prior_weights)
        self.br_table = BestResponseTable(game, self.profiles)
        self._observed: list[HandRecord] = []

    def reset(self, rng: np.random.Generator) -> None:
        self.posterior.reset()
        self._observed = []

    def observe(self, hand: HandRecord) -> None:
        self.posterior.update(hand)
        self._observed.append(hand)

    def _extends_observed(self, hands: Sequence[HandRecord]) -> bool:
        """True if ``hands`` starts with exactly the hands the posterior was conditioned on."""
        seen = self._observed
        if len(hands) < len(seen):
            return False
        return all(h is o or h == o for h, o in zip(hands[: len(seen)], seen, strict=True))

    def _sync(self, ctx: SessionContext) -> np.ndarray:
        """Posterior probabilities given exactly ``ctx.hands``."""
        hands = list(ctx.hands)
        if not self._extends_observed(hands):
            self.posterior.reset()
            self._observed = []
        for hand in hands[len(self._observed) :]:
            self.posterior.update(hand)
            self._observed.append(hand)
        return self.posterior.probs

    def belief(self, ctx: SessionContext) -> np.ndarray | None:
        return self._sync(ctx)

    def br_action_posterior_for_seat(self, seat: int) -> TabularStrategy:
        """``pi*_t(a | I)`` at every infoset of ``seat`` under the agent's *current* posterior.

        The posterior is the one conditioned on the hands observed so far (``observe`` / the last
        ``strategy_for_hand`` or ``belief`` call); rows are mixed in general.
        """
        return self.br_table.action_posterior_for_seat(self.posterior.probs, seat)


class ThompsonAgent(_PosteriorAgent):
    """Sample an opponent from the posterior and play the best response to it.

    Best responses (both seats) are computed lazily once per opponent id and cached. The
    sampling generator comes from the constructor or, preferably, from ``reset(rng)``.
    """

    def __init__(
        self,
        game: Game,
        population: Population | Sequence[TabularStrategy],
        prior_weights: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        name: str = "thompson",
    ) -> None:
        super().__init__(game, population, prior_weights, name)
        self.rng = rng
        self.last_sample: int | None = None

    def reset(self, rng: np.random.Generator) -> None:
        super().reset(rng)
        self.rng = rng
        self.last_sample = None

    def best_response_to(self, opp_id: int) -> TabularStrategy:
        """Cached exact best response (both seats) to population member ``opp_id``."""
        return self.br_table.best_response_to(int(opp_id))

    def strategy_for_hand(self, ctx: SessionContext, seat: int) -> TabularStrategy:
        seat = check_seat(seat)
        if self.rng is None:
            raise RuntimeError("ThompsonAgent needs a generator: pass rng= or call reset(rng)")
        probs = self._sync(ctx)
        i = int(self.rng.choice(probs.size, p=probs))
        self.last_sample = i
        return {k: v.copy() for k, v in seat_entries(self.best_response_to(i), seat).items()}


class BayesBRAgent(_PosteriorAgent):
    """Best response to the posterior mixture of the population (myopic Bayes-optimal play).

    The mixture is the realization-equivalent behavioural strategy of "opponent ``i`` with
    probability ``posterior_i``" at the opponent's seat (``StrategyMixer``), so the best response
    to it maximises exact expected value under the current belief.
    """

    def __init__(
        self,
        game: Game,
        population: Population | Sequence[TabularStrategy],
        prior_weights: np.ndarray | None = None,
        name: str = "bayes_br",
    ) -> None:
        super().__init__(game, population, prior_weights, name)
        # indexed by the *opponent's* seat
        self._mixers = (
            StrategyMixer(game, self.profiles, 0),
            StrategyMixer(game, self.profiles, 1),
        )
        self.last_value: float | None = None

    def mixture(self, ctx: SessionContext, opp_seat: int) -> TabularStrategy:
        """Posterior-mixture opponent strategy at ``opp_seat`` given ``ctx``."""
        return self._mixers[check_seat(opp_seat)].mix(self._sync(ctx))

    def strategy_for_hand(self, ctx: SessionContext, seat: int) -> TabularStrategy:
        seat = check_seat(seat)
        br, value = best_response(self.game, self.mixture(ctx, 1 - seat), seat)
        self.last_value = value
        return br


class PluralityBRAgent(_PosteriorAgent):
    """Play the posterior-plurality best-response action ``argmax_a pi*_t(a | I)`` at every infoset.

    ``pi*_t`` is ``br_action_posterior`` of the cached per-member best responses under the exact
    posterior (both seats cached lazily, like ``ThompsonAgent``). Ties resolve to the lowest
    action index, matching ``solvers.best_response``. The strategy is pure at every infoset;
    ``last_action_posterior`` keeps the ``pi*_t`` rows of the last ``strategy_for_hand`` call.
    """

    def __init__(
        self,
        game: Game,
        population: Population | Sequence[TabularStrategy],
        prior_weights: np.ndarray | None = None,
        name: str = "plurality_br",
    ) -> None:
        super().__init__(game, population, prior_weights, name)
        self.last_action_posterior: TabularStrategy | None = None

    def strategy_for_hand(self, ctx: SessionContext, seat: int) -> TabularStrategy:
        seat = check_seat(seat)
        probs = self._sync(ctx)
        pi_star = self.br_table.action_posterior_for_seat(probs, seat)
        self.last_action_posterior = pi_star
        out: TabularStrategy = {}
        for key, row in pi_star.items():
            pure = np.zeros(N_ACTIONS)
            pure[int(np.argmax(row))] = 1.0  # argmax: first maximal index on ties
            out[key] = pure
        return out
