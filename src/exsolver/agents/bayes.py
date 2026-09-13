"""Exact-Bayes agents over a discrete population: Thompson sampling and Bayes best response."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from exsolver.agents.base import check_seat
from exsolver.bayes.posterior import ExactPosterior
from exsolver.data.records import HandRecord, SessionContext
from exsolver.games.base import Game
from exsolver.population.population import Population
from exsolver.solvers.best_response import best_response
from exsolver.solvers.mixture import StrategyMixer
from exsolver.strategy import TabularStrategy, merge_seats, seat_entries


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
        self._br_cache: dict[int, TabularStrategy] = {}

    def reset(self, rng: np.random.Generator) -> None:
        super().reset(rng)
        self.rng = rng
        self.last_sample = None

    def best_response_to(self, opp_id: int) -> TabularStrategy:
        """Cached exact best response (both seats) to population member ``opp_id``."""
        br = self._br_cache.get(opp_id)
        if br is None:
            br0, _ = best_response(self.game, self.profiles[opp_id], 0)
            br1, _ = best_response(self.game, self.profiles[opp_id], 1)
            br = merge_seats(br0, br1)
            self._br_cache[opp_id] = br
        return br

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
