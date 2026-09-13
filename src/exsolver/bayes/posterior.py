"""Exact Bayesian posterior over a discrete opponent population.

For a hand in which the agent held ``c_me`` and the opponent took actions ``a_1..a_k`` at
histories ``I_1..I_k`` (RESEARCH.md 2.2)::

    L(hand | theta) = sum_{c_opp} P(c_opp | c_me, board) * prod_k theta(a_k | I_k(c_opp))

The sum over the opponent's unknown private card is *joint* across all of their actions in the
hand (they share one card). If a showdown revealed ``c_opp`` the sum collapses to that single
term, which keeps its deal weight ``P(c_opp | c_me, board)`` (a constant across opponents, so
posteriors are unaffected; the returned value is then the log-probability of the observation up
to the factors below). Two theta-independent factors are deliberately left out:

* the agent's own actions, which is exactly why the posterior does not depend on the policy that
  generated the hands;
* the probability of the board, ``P(board | c_me)``: likelihoods are *conditional on the board*.
  Candidate weights use the engine's deal and board chance probabilities but are normalised over
  the candidates consistent with the agent's cards and the recorded board, i.e. they are
  ``P(c_opp | c_me, board)``; the dropped normaliser is the same for every opponent.

Two stages keep an update cheap regardless of the population size:

* ``replay_hand`` (theta-independent) walks the game engine once per candidate opponent card,
  starting from the root deal, recording the opponent's infoset ids and the deal/board chance
  weights; with a revealed card only that candidate is kept;
* ``log_likelihoods_from_replay`` (theta-dependent) gathers from a dense
  ``[M, n_infosets, 3]`` log-probability array and log-sum-exps over candidates.

Everything is in log space; opponents that cannot have produced the hand get ``-inf`` (never
NaN) and a posterior of exactly zero.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from exsolver.data.records import ACT, BOARD, HandRecord
from exsolver.games.base import N_ACTIONS, Game
from exsolver.games.tree import compile_tree
from exsolver.population.population import Population
from exsolver.strategy import TabularStrategy, strategy_to_array


def safe_log(x: np.ndarray) -> np.ndarray:
    """``log(x)`` as float64 with ``log(0) = -inf`` and no warning."""
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(x, dtype=np.float64))


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable log-sum-exp along ``axis``; ``-inf`` (not NaN) where every entry is ``-inf``."""
    x = np.asarray(x, dtype=np.float64)
    mx = np.max(x, axis=axis, keepdims=True)
    mx = np.where(np.isfinite(mx), mx, 0.0)  # all -inf: shift by 0, log(0) below gives -inf
    with np.errstate(divide="ignore"):
        s = np.log(np.exp(x - mx).sum(axis=axis))
    return s + np.squeeze(mx, axis=axis)


@dataclass(frozen=True)
class HandReplay:
    """Theta-independent replay of one hand from the agent's point of view.

    ``infosets[c, k]`` is the id (``compile_tree(game).infoset_index``) of the infoset at which
    the opponent took its ``k``-th action if it holds candidate card(s) ``c``; ``actions[k]`` is
    that action (identical for every candidate: the history is public); ``log_weights[c]`` is
    ``log P(candidate c | agent's cards, board)`` over the candidates consistent with the agent's
    observations. With a revealed opponent card there is exactly one candidate.
    """

    infosets: np.ndarray  # [C, K] int64
    actions: np.ndarray  # [K] int64
    log_weights: np.ndarray  # [C] float64

    @property
    def n_candidates(self) -> int:
        return int(self.log_weights.shape[0])


def replay_hand(game: Game, hand: HandRecord) -> HandReplay:
    """Replay ``hand`` for every opponent card consistent with the agent's cards and the board.

    Candidates are the outcomes of the root deal that give the agent ``hand.my_cards``; each is
    weighted by its deal probability times the probability of the recorded board cards (a
    candidate holding a board card is impossible and dropped). Weights are normalised over the
    consistent candidates -- giving ``P(c_opp | c_me, board)``, so the theta-independent
    ``P(board | c_me)`` is dropped -- before a revealed card (``hand.opp_cards``) selects one of
    them.

    Raises ``ValueError`` for a record the engine cannot replay (wrong actor, illegal action,
    board event at a decision node, revealed card inconsistent with the deal, ...).
    """
    if hand.seat not in (0, 1):
        raise ValueError(f"hand.seat must be 0 or 1, got {hand.seat}")
    agent, opp = hand.seat, 1 - hand.seat
    index = compile_tree(game).infoset_index
    my_cards = tuple(int(c) for c in hand.my_cards)
    events = list(hand.events)
    root = game.root()
    if not game.is_chance(root):
        raise ValueError("expected the root of the game to be the deal chance node")

    candidates: list[tuple[tuple[int, ...], float, list[int]]] = []
    actions: list[int] | None = None
    for s, p_deal in game.chance_outcomes(root):
        if tuple(int(c) for c in game.private_cards(s, agent)) != my_cards:
            continue
        opp_cards = tuple(int(c) for c in game.private_cards(s, opp))
        weight = float(p_deal)
        infs: list[int] = []
        acts: list[int] = []
        alive = True
        i = 0
        while i < len(events):
            if game.is_terminal(s):
                raise ValueError(f"event {events[i]!r} recorded after the hand ended")
            if game.is_chance(s):
                before = len(game.board_cards(s))
                outcomes = game.chance_outcomes(s)
                n_new = len(game.board_cards(outcomes[0][0])) - before
                if n_new < 1:
                    raise ValueError("a chance node inside the hand reveals no board card")
                want = events[i : i + n_new]
                if len(want) < n_new or any(ev[0] != BOARD for ev in want):
                    raise ValueError(
                        f"expected {n_new} board event(s) at position {i}, got {want!r}"
                    )
                cards = tuple(int(ev[1]) for ev in want)
                for s2, p2 in outcomes:
                    if tuple(game.board_cards(s2)[before:]) == cards:
                        s, weight = s2, weight * float(p2)
                        break
                else:
                    alive = False  # this candidate holds one of the board cards
                    break
                i += n_new
                continue
            ev = events[i]
            if ev[0] != ACT:
                raise ValueError(f"expected an action event at a decision node, got {ev!r}")
            _, actor, a = ev
            to_act = game.current_player(s)
            if actor != (0 if to_act == agent else 1):
                raise ValueError(f"event {ev!r} attributes the action to the wrong player")
            if actor == 1:
                infs.append(index[game.infoset_key(s, opp)])
                acts.append(int(a))
            s = game.apply(s, int(a))
            i += 1
        if not alive:
            continue
        if actions is None:
            actions = acts
        elif actions != acts:
            raise ValueError("opponent action sequence differs across candidate cards")
        candidates.append((opp_cards, weight, infs))

    if not candidates or actions is None:
        raise ValueError(f"no deal is consistent with my_cards={my_cards} and the recorded board")
    total = sum(w for _, w, _ in candidates)
    if hand.opp_cards is not None:
        revealed = tuple(int(c) for c in hand.opp_cards)
        candidates = [c for c in candidates if c[0] == revealed]
        if len(candidates) != 1:
            raise ValueError(f"revealed opponent cards {revealed} are inconsistent with the deal")
    k = len(actions)
    infosets = np.array([infs for _, _, infs in candidates], dtype=np.int64).reshape(
        len(candidates), k
    )
    weights = np.array([w for _, w, _ in candidates], dtype=np.float64) / total
    return HandReplay(infosets, np.asarray(actions, dtype=np.int64), np.log(weights))


def log_likelihoods_from_replay(log_pop: np.ndarray, replay: HandReplay) -> np.ndarray:
    """``[M]`` log-likelihoods of a replayed hand under each row of ``log_pop`` (``[M, I, 3]``)."""
    m = log_pop.shape[0]
    if replay.actions.size == 0:
        return np.zeros(m)  # the opponent never acted: every theta explains the hand equally
    ll = log_pop[:, replay.infosets, replay.actions]  # [M, C, K]
    return logsumexp(ll.sum(axis=2) + replay.log_weights, axis=1)


def population_array(game: Game, profiles: Sequence[TabularStrategy]) -> np.ndarray:
    """Dense ``[M, n_infosets, 3]`` array of full profiles (both seats) in enumeration order."""
    if len(profiles) == 0:
        raise ValueError("need at least one profile")
    return np.stack([strategy_to_array(game, p) for p in profiles])


def hand_log_likelihoods(game: Game, pop_array: np.ndarray, hand: HandRecord) -> np.ndarray:
    """``[M]`` log-likelihoods of ``hand`` under the population ``pop_array`` (``[M, I, 3]``).

    Pure function of its inputs; see the module docstring for the definition. Use
    ``ExactPosterior`` (which caches ``log(pop_array)``) for repeated updates.
    """
    pop = np.asarray(pop_array, dtype=np.float64)
    tree = compile_tree(game)
    if pop.ndim != 3 or pop.shape[1:] != (tree.n_infosets, N_ACTIONS):
        raise ValueError(
            f"pop_array must have shape [M, {tree.n_infosets}, {N_ACTIONS}], got {pop.shape}"
        )
    return log_likelihoods_from_replay(safe_log(pop), replay_hand(game, hand))


class ExactPosterior:
    """Exact posterior over ``M`` opponents, updated one ``HandRecord`` at a time.

    ``population`` is a ``Population`` (its ``weights`` are the default prior) or a sequence of
    full profiles (uniform prior by default). Probabilities are kept as normalised
    log-probabilities; ``-inf`` marks opponents that cannot have produced the observed hands.
    """

    def __init__(
        self,
        game: Game,
        population: Population | Sequence[TabularStrategy],
        prior_weights: np.ndarray | None = None,
    ) -> None:
        if isinstance(population, Population):
            profiles = list(population.profiles)
            default_weights = population.weights
        else:
            profiles = list(population)
            default_weights = None
        self.game = game
        self.pop_array = population_array(game, profiles)
        self.log_pop = safe_log(self.pop_array)
        m = self.pop_array.shape[0]
        if prior_weights is None:
            w = np.full(m, 1.0 / m) if default_weights is None else default_weights
        else:
            w = prior_weights
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        if w.shape != (m,):
            raise ValueError(f"prior_weights must have shape ({m},), got {w.shape}")
        if not np.all(np.isfinite(w)) or np.any(w < 0.0) or w.sum() <= 0.0:
            raise ValueError("prior_weights must be finite, non-negative and not all zero")
        self.log_prior = safe_log(w / w.sum())
        self._log_post = self.log_prior.copy()
        self.n_hands = 0
        self.log_evidence = 0.0  # log P(hands) up to the agent's own action factors

    @property
    def m(self) -> int:
        return int(self.log_prior.shape[0])

    def reset(self) -> None:
        """Back to the prior."""
        self._log_post = self.log_prior.copy()
        self.n_hands = 0
        self.log_evidence = 0.0

    def update(self, hand: HandRecord) -> None:
        """Condition on one completed hand.

        Raises ``ValueError`` if no opponent with positive posterior mass can have produced it.
        """
        ll = log_likelihoods_from_replay(self.log_pop, replay_hand(self.game, hand))
        joint = self._log_post + ll
        lz = float(logsumexp(joint))
        if not np.isfinite(lz):
            raise ValueError(
                "the observed hand has zero probability under every opponent with posterior mass"
            )
        self._log_post = joint - lz
        self.log_evidence += lz
        self.n_hands += 1

    def update_many(self, hands: Sequence[HandRecord]) -> None:
        for hand in hands:
            self.update(hand)

    @property
    def log_post(self) -> np.ndarray:
        """Normalised log posterior ``[M]`` (a copy)."""
        return self._log_post.copy()

    @property
    def probs(self) -> np.ndarray:
        """Posterior probabilities ``[M]``; exact zeros for impossible opponents."""
        return np.exp(self._log_post)

    def entropy(self) -> float:
        """Shannon entropy of the posterior in nats (``0 log 0 = 0``)."""
        lp = self._log_post
        p = np.exp(lp)
        nz = p > 0.0
        h = float(-(p[nz] * lp[nz]).sum())
        return h if h > 0.0 else 0.0  # a point mass gives -0.0 otherwise

    def map_id(self) -> int:
        """Most probable opponent (ties resolved to the lowest index)."""
        return int(np.argmax(self._log_post))

    def copy(self) -> ExactPosterior:
        """Independent copy of the posterior state (the population arrays are shared)."""
        new = copy.copy(self)
        new._log_post = self._log_post.copy()
        return new
