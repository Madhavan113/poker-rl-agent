"""Adaptive opponents (E5 spec §Adaptive opponents).

After every block of hands the opponent moves part of the way toward a best response to the
strategy we played in that block:

    theta_{k+1} = (1 - eta) * theta_k  (+)  eta * BR(sigma_k)

where ``(+)`` is the sequence-form (realization-equivalent) mixture, so the new profile plays
exactly like "theta_k with probability 1 - eta, BR(sigma_k) with probability eta" per hand. The
opponent sees ``sigma_k`` exactly: an omniscient learner, the fastest adapter of this family for
a given ``eta``. ``eta = 0`` is a static opponent; repeated steps against a fixed ``sigma``
converge to a best response to it.
"""

from __future__ import annotations

import numpy as np

from exsolver.games.base import N_ACTIONS, Game
from exsolver.games.tree import CompiledTree, compile_tree


def mix_sequence_form(tree: CompiledTree, a: np.ndarray, b: np.ndarray, w_b: float) -> np.ndarray:
    """Behavioural profiles ``[B, I, 3]`` realization-equivalent to ``a`` w.p. ``1 - w_b``, else ``b``.

    Infosets neither profile reaches get the plain average.
    """
    if not 0.0 <= w_b <= 1.0:
        raise ValueError("w_b must lie in [0, 1]")
    ra, rb = tree.realization_batch(a), tree.realization_batch(b)
    parent = tree.infoset_parent_seq
    wa_r = (1.0 - w_b) * ra[:, parent]  # [B, I]
    wb_r = w_b * rb[:, parent]
    den = wa_r + wb_r
    num = wa_r[:, :, None] * a + wb_r[:, :, None] * b
    plain = (1.0 - w_b) * a + w_b * b
    ok = den > 0.0
    return np.where(ok[:, :, None], num / np.where(ok, den, 1.0)[:, :, None], plain)


def best_response_profile(game: Game, ours: np.ndarray) -> np.ndarray:
    """``[B, I, 3]``: per instance, the opponent's pure best response at both seats to ``ours``."""
    tree = compile_tree(game)
    out = np.zeros_like(ours)
    for b in range(ours.shape[0]):
        for seat in (0, 1):  # the opponent's seat; ours is the other one
            sig = ours[b]
            rho = tree.realization(sig)
            cf = tree.reach_chance * tree.reach(rho, 1 - seat)
            _, a_star = tree.best_response_pass(sig, seat, cf)
            rows = tree.infoset_rows[seat]
            out[b, rows, a_star[rows]] = 1.0
    return out


def adapt(game: Game, theta: np.ndarray, ours: np.ndarray, eta: float) -> np.ndarray:
    """One adaptation step for a batch of opponents ``theta`` against our profiles ``ours``."""
    if eta == 0.0:
        return np.array(theta, copy=True)
    tree = compile_tree(game)
    if theta.shape != ours.shape or theta.shape[1:] != (tree.n_infosets, N_ACTIONS):
        raise ValueError("theta and ours must both be [B, n_infosets, 3]")
    return mix_sequence_form(tree, theta, best_response_profile(game, ours), eta)
