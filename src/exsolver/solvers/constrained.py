"""Node-locked solving: CFR+ in games where the opponent is partly fixed to a model.

Two restricted games, both for one free seat ("ours") against an opponent model ``lock`` (a
behavioural strategy for the other seat), as in ``docs/experiments/e5-population-nodelock.md``:

* **Node lock with confidence** ``c`` (one weight per opponent infoset). At every opponent
  infoset ``I`` the opponent plays ``c_I * lock(I) + (1 - c_I) * tau(I)``, where ``tau`` is
  chosen adversarially. ``c = 1`` everywhere is the best response to ``lock``; ``c = 0``
  everywhere is the unrestricted game (Nash); ``c`` in ``{0, 1}`` is a solver-style hard node
  lock; fractional ``c`` (e.g. ``n_I / (n_I + k)`` from observation counts) is the
  data-biased-response style soft lock (Johanson & Bowling 2009).
* **Restricted Nash response** ``RNR(p)`` (Johanson, Zinkevich & Bowling 2007): maximise
  ``p * EV(sigma, lock) + (1 - p) * min_tau EV(sigma, tau)``. The model and the adversary are
  mixed once per hand (in sequence form), not per infoset.

Both are zero-sum games over convex strategy sets, so CFR+ with regret matching+ on ``tau``
converges; our seat's linear-averaged strategy is returned. ``LockedCFR`` solves a batch of
``B`` instances at once and keeps its regrets between calls (warm start), which is what an
agent that re-solves every few hands wants. ``restricted_value`` and ``rnr_objective`` evaluate
the two objectives exactly for any strategy, independently of the solver.

Dense arrays follow ``CompiledTree``: ``[B, n_infosets, 3]`` strategies, ``[B, n_infosets]``
confidences. Only the rows of the seat a quantity belongs to are read.
"""

from __future__ import annotations

import numpy as np

from exsolver.games.base import N_ACTIONS, Game
from exsolver.games.tree import CompiledTree, compile_tree

_TIE_TOL = 1e-12


def _normalise_legal(tree: CompiledTree) -> np.ndarray:
    legal = tree.infoset_legal.astype(np.float64)
    return legal / legal.sum(axis=1, keepdims=True)


class _BatchedPasses:
    """Batched value / counterfactual-value passes over a compiled tree."""

    def __init__(self, tree: CompiledTree) -> None:
        self.tree = tree
        self.order: list[np.ndarray] = []
        self.starts: list[np.ndarray] = []
        self.infs: list[np.ndarray] = []
        for p in (0, 1):
            d = tree.dec_nodes[p]
            order = np.argsort(tree.infoset[d], kind="stable")
            inf_sorted = tree.infoset[d][order]
            starts = np.flatnonzero(np.r_[True, inf_sorted[1:] != inf_sorted[:-1]])
            self.order.append(d[order])
            self.starts.append(starts)
            self.infs.append(inf_sorted[starts])

    def values(self, sigma: np.ndarray) -> np.ndarray:
        """Seat-0 node values ``[B, N + 1]`` under profiles ``sigma`` ``[B, I, 3]``."""
        t = self.tree
        u = np.zeros((sigma.shape[0], t.n_nodes + 1))
        u[:, t.terminals] = t.util0[t.terminals]
        for lvl in t.levels:
            dec = lvl.dec_all
            if dec.size:
                u[:, dec] = np.einsum("bij,bij->bi", sigma[:, t.infoset[dec]], u[:, t.child[dec]])
            if lvl.chance_src.size:
                np.add.at(u, (slice(None), lvl.chance_src), lvl.chance_prob * u[:, lvl.chance_dst])
        return u

    def cf_values(self, sigma: np.ndarray, player: int) -> tuple[np.ndarray, np.ndarray]:
        """Counterfactual action values ``[B, I, 3]`` for ``player`` (own sign) and realizations.

        ``q[b, I, a] = sum_{h in I} reach_chance(h) reach_other(h) u_player(child(h, a))``; rows
        of the other player's infosets are zero.
        """
        t = self.tree
        rho = t.realization_batch(sigma)
        u = self.values(sigma)
        nodes = self.order[player]
        cf = t.reach_chance[nodes][None, :] * rho[:, t.node_seq[1 - player][nodes]]
        sign = 1.0 if player == 0 else -1.0
        g = (sign * cf)[:, :, None] * u[:, t.child[nodes]]  # [B, D, 3]
        q = np.zeros((sigma.shape[0], t.n_infosets, N_ACTIONS))
        q[:, self.infs[player]] = np.add.reduceat(g, self.starts[player], axis=1)
        return q, rho


class LockedCFR:
    """Batched CFR+ for node-locked and restricted-Nash-response games; see the module doc.

    ``seat`` is our (free) seat; ``batch`` the number of independent instances. Call ``solve``
    repeatedly with changing locks: regrets persist between calls (warm start), the average is
    restarted at every call. ``reset`` drops the regrets. A warm start helps when the lock drifts
    slightly and can be a little worse than a cold start after a large change; switching between
    node-lock and RNR games resets the adversary's regrets. After each call ``adversary`` holds the
    adversary's average strategy, which certifies the solution through ``duality_gap``.
    """

    def __init__(self, game: Game, seat: int, batch: int) -> None:
        if seat not in (0, 1):
            raise ValueError(f"seat must be 0 or 1, got {seat}")
        if batch < 1:
            raise ValueError("batch must be >= 1")
        self.game = game
        self.tree = compile_tree(game)
        self.seat = seat
        self.batch = batch
        self._passes = _BatchedPasses(self.tree)
        self._legal = self.tree.infoset_legal.astype(np.float64)
        self._uniform = _normalise_legal(self.tree)
        self._rows = self.tree.infoset_rows
        self.reset()

    def reset(self) -> None:
        shape = (self.batch, self.tree.n_infosets, N_ACTIONS)
        self.regrets = np.zeros(shape)
        self.sigma = np.broadcast_to(self._uniform, shape).copy()
        self._mode: str | None = None

    def _rm_plus(self, rows: np.ndarray) -> None:
        legal = self._legal[rows]
        pos = np.maximum(self.regrets[:, rows], 0.0) * legal
        tot = pos.sum(axis=-1, keepdims=True)
        self.sigma[:, rows] = np.where(
            tot > 0.0, pos / np.where(tot > 0.0, tot, 1.0), self._uniform[rows]
        )

    def _update(self, rows: np.ndarray, q: np.ndarray, scale: np.ndarray | float = 1.0) -> None:
        """Regret-matching+ step; ``scale`` weights the instantaneous regret (``1 - c`` for the
        adversary's share of an infoset, ``1 - p`` under RNR) so that regrets accumulated under
        different locks or confidences stay commensurate across warm starts."""
        base = (self.sigma[:, rows] * q[:, rows]).sum(axis=-1, keepdims=True)
        inst = (q[:, rows] - base) * self._legal[rows] * scale
        self.regrets[:, rows] = np.maximum(self.regrets[:, rows] + inst, 0.0)
        self._rm_plus(rows)

    def solve(
        self,
        lock: np.ndarray,
        iterations: int,
        conf: np.ndarray | None = None,
        rnr_p: float | None = None,
    ) -> np.ndarray:
        """Our linear-averaged strategy ``[B, I, 3]`` (rows of ``seat`` are meaningful).

        ``lock`` ``[B, I, 3]`` is read at the opponent's rows. Exactly one of ``conf``
        (``[B, I]`` in ``[0, 1]``, read at the opponent's rows) and ``rnr_p`` (in ``[0, 1]``)
        must be given.
        """
        t, ours, opp = self.tree, self.seat, 1 - self.seat
        lock = np.asarray(lock, dtype=np.float64)
        if lock.shape != (self.batch, t.n_infosets, N_ACTIONS):
            raise ValueError(f"lock must have shape {(self.batch, t.n_infosets, N_ACTIONS)}")
        if iterations < 1:
            raise ValueError("iterations must be >= 1")
        if (conf is None) == (rnr_p is None):
            raise ValueError("give exactly one of conf and rnr_p")
        r_opp, r_us = self._rows[opp], self._rows[ours]
        _check_rows(lock[:, r_opp], self._legal[r_opp], "lock")
        if conf is not None:
            c = np.asarray(conf, dtype=np.float64)
            if c.shape != (self.batch, t.n_infosets):
                raise ValueError(f"conf must have shape {(self.batch, t.n_infosets)}")
            c = c[:, r_opp, None]
            if np.any(~np.isfinite(c)) or np.any(c < 0.0) or np.any(c > 1.0):
                raise ValueError("conf entries must lie in [0, 1]")
        elif not 0.0 <= float(rnr_p) <= 1.0:  # type: ignore[arg-type]
            raise ValueError("rnr_p must lie in [0, 1]")
        mode = "conf" if conf is not None else "rnr"
        if self._mode is not None and mode != self._mode:
            # the adversary's regrets mean different things in the two games: start it afresh
            self.regrets[:, r_opp] = 0.0
            self.sigma[:, r_opp] = self._uniform[r_opp]
        self._mode = mode

        avg = np.zeros((self.batch, r_us.size, N_ACTIONS))
        adv = np.zeros((self.batch, r_opp.size, N_ACTIONS))
        parent = t.infoset_parent_seq[r_us]
        parent_opp = t.infoset_parent_seq[r_opp]
        passes = self._passes
        for it in range(1, iterations + 1):
            # opponent (free part) first, then us, each against the other's current strategy
            if conf is not None:
                eff = self.sigma.copy()
                eff[:, r_opp] = c * lock[:, r_opp] + (1.0 - c) * self.sigma[:, r_opp]
                q, rho_o = passes.cf_values(eff, opp)
                adv += (it * rho_o[:, parent_opp])[:, :, None] * eff[:, r_opp]
                self._update(r_opp, q, 1.0 - c)
                eff[:, r_opp] = c * lock[:, r_opp] + (1.0 - c) * self.sigma[:, r_opp]
                q, rho = passes.cf_values(eff, ours)
            else:
                p = float(rnr_p)  # type: ignore[arg-type]
                q, rho_o = passes.cf_values(self.sigma, opp)
                adv += (it * rho_o[:, parent_opp])[:, :, None] * self.sigma[:, r_opp]
                self._update(r_opp, q, 1.0 - p)
                q_free, rho = passes.cf_values(self.sigma, ours)
                model = self.sigma.copy()
                model[:, r_opp] = lock[:, r_opp]
                q_model, _ = passes.cf_values(model, ours)
                q = p * q_model + (1.0 - p) * q_free
            avg += (it * rho[:, parent])[:, :, None] * self.sigma[:, r_us]
            self._update(r_us, q)

        out = np.broadcast_to(self._uniform, (self.batch, t.n_infosets, N_ACTIONS)).copy()
        tot = avg.sum(axis=-1, keepdims=True)
        out[:, r_us] = np.where(tot > 0.0, avg / np.where(tot > 0.0, tot, 1.0), self._uniform[r_us])
        # the adversary's average: the effective (locked + free) strategy under a node lock,
        # the free adversary tau under RNR; it certifies the solution (see ``duality_gap``)
        self.adversary = np.broadcast_to(self._uniform, out.shape).copy()
        tot = adv.sum(axis=-1, keepdims=True)
        fallback = np.broadcast_to(self._uniform[r_opp], adv.shape)
        if conf is not None:
            fallback = c * lock[:, r_opp] + (1.0 - c) * self._uniform[r_opp]
        self.adversary[:, r_opp] = np.where(
            tot > 0.0, adv / np.where(tot > 0.0, tot, 1.0), fallback
        )
        return out


def _check_rows(rows: np.ndarray, legal: np.ndarray, what: str) -> None:
    if np.any(~np.isfinite(rows)) or np.any(rows < 0.0):
        raise ValueError(f"{what} has negative or non-finite entries")
    if np.any(rows * (1.0 - legal) > 0.0):
        raise ValueError(f"{what} puts mass on illegal actions")
    if np.any(np.abs(rows.sum(axis=-1) - 1.0) > 1e-9):
        raise ValueError(f"{what} rows must sum to one")


# ------------------------------------------------------------------- exact objectives
def ev_dense(game: Game, seat: int, ours: np.ndarray, opp: np.ndarray) -> float:
    """Exact EV for ``seat`` playing ``ours`` ``[I, 3]`` against ``opp`` ``[I, 3]``."""
    tree = compile_tree(game)
    sigma = np.array(opp, dtype=np.float64)
    rows = tree.infoset_rows[seat]
    sigma[rows] = ours[rows]
    v0 = float(tree.values(sigma)[tree.root])
    return v0 if seat == 0 else -v0


def restricted_value(
    game: Game, seat: int, ours: np.ndarray, lock: np.ndarray, conf: np.ndarray
) -> float:
    """Worst-case EV for ``seat`` playing ``ours`` against the node-locked opponent.

    The opponent plays ``conf[I] * lock[I] + (1 - conf[I]) * tau[I]`` at its infosets and
    picks ``tau`` to minimise our EV (exact backward induction; ``tau`` is pure w.l.o.g.).
    ``conf = 0`` gives ``-best_response_value(game, ours, 1 - seat)``; ``conf = 1`` gives the
    EV against ``lock``.
    """
    tree = compile_tree(game)
    opp = 1 - seat
    ours = np.asarray(ours, dtype=np.float64)
    lock = np.asarray(lock, dtype=np.float64)
    c = np.asarray(conf, dtype=np.float64).reshape(tree.n_infosets)
    if np.any(~np.isfinite(c)) or np.any(c < 0.0) or np.any(c > 1.0):
        raise ValueError("conf entries must lie in [0, 1]")
    sigma = np.zeros((tree.n_infosets, N_ACTIONS))
    sigma[tree.infoset_rows[seat]] = ours[tree.infoset_rows[seat]]
    rho = tree.realization(sigma)
    cf = tree.reach_chance * tree.reach(rho, seat)  # chance x our reach
    sign = 1.0 if opp == 0 else -1.0
    u = np.zeros(tree.n_nodes + 1)
    u[tree.terminals] = tree.util0[tree.terminals]
    infoset, child, legal = tree.infoset, tree.child, tree.infoset_legal
    for lvl in tree.levels:
        dec_s = lvl.dec[seat]
        if dec_s.size:
            u[dec_s] = np.einsum("ij,ij->i", sigma[infoset[dec_s]], u[child[dec_s]])
        dec_o = lvl.dec[opp]
        if dec_o.size:
            infs, inv = lvl.dec_infosets[opp], lvl.dec_inv[opp]
            cu = u[child[dec_o]] * (sign * cf[dec_o])[:, None]
            qv = np.empty((infs.size, N_ACTIONS))
            for a in range(N_ACTIONS):
                qv[:, a] = np.bincount(inv, weights=cu[:, a], minlength=infs.size)
            qv = np.where(legal[infs], qv, -np.inf)
            best = qv.max(axis=1)
            tol = _TIE_TOL * np.maximum(1.0, np.abs(best))
            a_best = np.argmax(qv >= (best - tol)[:, None], axis=1)
            free = np.zeros((infs.size, N_ACTIONS))
            free[np.arange(infs.size), a_best] = 1.0
            ci = c[infs][:, None]
            eff = ci * lock[infs] + (1.0 - ci) * free
            u[dec_o] = np.einsum("ij,ij->i", eff[inv], u[child[dec_o]])
        if lvl.chance_src.size:
            np.add.at(u, lvl.chance_src, lvl.chance_prob * u[lvl.chance_dst])
    v0 = float(u[tree.root])
    return v0 if seat == 0 else -v0


def rnr_objective(game: Game, seat: int, ours: np.ndarray, lock: np.ndarray, p: float) -> float:
    """``p * EV(ours, lock) + (1 - p) * worst-case EV(ours)`` for ``seat``."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    tree = compile_tree(game)
    worst = restricted_value(game, seat, ours, lock, np.zeros(tree.n_infosets))
    return p * ev_dense(game, seat, ours, lock) + (1.0 - p) * worst


def duality_gap(
    game: Game,
    seat: int,
    ours: np.ndarray,
    adversary: np.ndarray,
    lock: np.ndarray,
    conf: np.ndarray | None = None,
    rnr_p: float | None = None,
) -> float:
    """Upper bound on how far ``ours`` is from optimal in its restricted game (>= 0 up to rounding).

    ``adversary`` is any admissible opponent: under a node lock an effective strategy of the
    form ``c * lock + (1 - c) * tau`` (``LockedCFR.adversary`` is one), under RNR a free ``tau``.
    The game value lies between ``objective(ours)`` and our best-response value against the
    adversary (against the per-hand mixture of ``lock`` and ``tau`` under RNR).
    """
    tree = compile_tree(game)
    opp = 1 - seat
    if (conf is None) == (rnr_p is None):
        raise ValueError("give exactly one of conf and rnr_p")
    if conf is not None:
        lower = restricted_value(game, seat, ours, lock, conf)
        target = np.asarray(adversary, dtype=np.float64)
    else:
        lower = rnr_objective(game, seat, ours, lock, float(rnr_p))
        target = _mix_two(tree, np.asarray(lock, dtype=np.float64), adversary, float(rnr_p), opp)
    sigma = target.copy()
    rho = tree.realization(sigma)
    cf = tree.reach_chance * tree.reach(rho, opp)
    u, _ = tree.best_response_pass(sigma, seat, cf)
    upper = float(u[tree.root]) * (1.0 if seat == 0 else -1.0)
    return upper - lower


def _mix_two(tree: CompiledTree, a: np.ndarray, b: np.ndarray, w_a: float, seat: int) -> np.ndarray:
    """``seat``'s rows of the behavioural strategy playing ``a`` w.p. ``w_a`` per hand, else ``b``."""
    ra, rb = tree.realization(a), tree.realization(b)
    rows = tree.infoset_rows[seat]
    par = tree.infoset_parent_seq[rows]
    wa, wb = w_a * ra[par], (1.0 - w_a) * rb[par]
    den = wa + wb
    out = np.zeros_like(a)
    mixed = (wa[:, None] * a[rows] + wb[:, None] * b[rows]) / np.where(den > 0, den, 1.0)[:, None]
    out[rows] = np.where((den > 0)[:, None], mixed, w_a * a[rows] + (1.0 - w_a) * b[rows])
    return out
