"""Online opponent inference for node locking (E5 spec §Inference).

Two estimators of the opponent's behavioural profile from the hands seen so far, both with an
optional per-hand weight (``gamma ** age_in_blocks``) so that old hands fade when the opponent
adapts:

* **Population posterior** (``discounted_posterior``): ``log P(i | h) = log prior_i +
  sum_t w_t log L(hand_t | theta_i)`` over a discrete population; the estimate is the
  sequence-form mixture of the population under that posterior (``StrategyMixer``). With all
  weights 1 this is ``ExactPosterior``.
* **Prior-free posterior mode** (``em_fit``): ``strength`` pseudo-observations of a reference
  profile (the equilibrium) at every opponent infoset, fitted by EM over the opponent's hidden
  card when there was no showdown.

``expected_visits`` gives the (weighted) number of times the opponent acted at each infoset
under an estimate -- the ``n_I`` that sets the lock confidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from exsolver.bayes.posterior import HandReplay, logsumexp
from exsolver.games.base import N_ACTIONS


@dataclass(frozen=True)
class PaddedHands:
    """``H`` replays padded to ``C`` candidates and ``K`` opponent actions.

    ``infosets [H, C, K]`` (0 where padded), ``actions [H, K]``, ``act_mask [H, K]``,
    ``log_weights [H, C]`` (``-inf`` for padded candidates).
    """

    infosets: np.ndarray
    actions: np.ndarray
    act_mask: np.ndarray
    log_weights: np.ndarray

    @property
    def n_hands(self) -> int:
        return int(self.actions.shape[0])


def pad_replays(replays: Sequence[HandReplay]) -> PaddedHands:
    h = len(replays)
    c_max = max((r.n_candidates for r in replays), default=1)
    k_max = max((int(r.actions.size) for r in replays), default=1) or 1
    infosets = np.zeros((h, c_max, k_max), dtype=np.int64)
    actions = np.zeros((h, k_max), dtype=np.int64)
    mask = np.zeros((h, k_max))
    lw = np.full((h, c_max), -np.inf)
    for i, r in enumerate(replays):
        c, k = r.infosets.shape
        infosets[i, :c, :k] = r.infosets
        actions[i, :k] = r.actions
        mask[i, :k] = 1.0
        lw[i, :c] = r.log_weights
    return PaddedHands(infosets, actions, mask, lw)


def _responsibilities(log_theta: np.ndarray, hands: PaddedHands) -> np.ndarray:
    """``[H, C]`` posterior over each hand's opponent card under the profile ``log_theta``."""
    acts = np.broadcast_to(hands.actions[:, None, :], hands.infosets.shape)
    terms = log_theta[hands.infosets, acts]
    ll = np.where(hands.act_mask[:, None, :] > 0.0, terms, 0.0).sum(axis=2)
    ll = ll + hands.log_weights
    norm = logsumexp(ll, axis=1)
    norm = np.where(np.isfinite(norm), norm, 0.0)
    return np.exp(ll - norm[:, None])


def _expected_counts(
    log_theta: np.ndarray, hands: PaddedHands, weights: np.ndarray, n_infosets: int
) -> np.ndarray:
    """``[I, 3]`` expected (weighted) opponent action counts per infoset."""
    resp = _responsibilities(log_theta, hands) * weights[:, None]  # [H, C]
    w = resp[:, :, None] * hands.act_mask[:, None, :]  # [H, C, K]
    acts = np.broadcast_to(hands.actions[:, None, :], hands.infosets.shape)
    flat = hands.infosets.ravel() * N_ACTIONS + acts.ravel()
    counts = np.bincount(flat, weights=w.ravel(), minlength=n_infosets * N_ACTIONS)
    return counts.reshape(n_infosets, N_ACTIONS)


def _check_weights(weights: np.ndarray | None, hands: PaddedHands) -> np.ndarray:
    if weights is None:
        return np.ones(hands.n_hands)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape != (hands.n_hands,) or np.any(~np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must be finite, non-negative, one per hand")
    return w


def _log(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(p)


def expected_visits(
    theta: np.ndarray, hands: PaddedHands, weights: np.ndarray | None = None
) -> np.ndarray:
    """``[I]`` weighted expected number of opponent decisions at each infoset under ``theta``."""
    w = _check_weights(weights, hands)
    return _expected_counts(_log(theta), hands, w, theta.shape[0]).sum(axis=1)


def em_fit(
    hands: PaddedHands,
    reference: np.ndarray,
    strength: float,
    weights: np.ndarray | None = None,
    init: np.ndarray | None = None,
    iterations: int = 30,
) -> np.ndarray:
    """Posterior mode ``[I, 3]`` with ``strength * reference`` pseudo-observations per infoset.

    Each EM step sets ``theta = (expected counts + strength * reference) / total``: the MAP under
    a Dirichlet(``strength * reference + 1``) prior (equivalently, add-``alpha`` smoothing). It
    is used instead of the Dirichlet(``alpha``) MAP because that mode is undefined where
    ``alpha < 1``, which the smoothed equilibrium has on most rows at strength 4.

    ``reference`` must be strictly positive on legal actions (smooth the equilibrium first).
    Rows the data never touch stay at ``reference``.
    """
    if strength <= 0.0:
        raise ValueError("strength must be positive")
    ref = np.asarray(reference, dtype=np.float64)
    w = _check_weights(weights, hands)
    theta = ref.copy() if init is None else np.asarray(init, dtype=np.float64).copy()
    prior = strength * ref
    for _ in range(iterations):
        counts = _expected_counts(_log(theta), hands, w, ref.shape[0])
        num = counts + prior
        theta = num / num.sum(axis=1, keepdims=True)
    return theta


def discounted_posterior(
    log_prior: np.ndarray, hand_log_likelihoods: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """``[M]`` posterior with per-hand weights: ``hand_log_likelihoods`` is ``[H, M]``."""
    ll = np.asarray(hand_log_likelihoods, dtype=np.float64)
    w = np.ones(ll.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (ll.shape[0],) or np.any(w < 0.0):
        raise ValueError("weights must be non-negative, one per hand")
    lp = np.asarray(log_prior, dtype=np.float64) + (w[:, None] * ll).sum(axis=0)
    return np.exp(lp - logsumexp(lp))
