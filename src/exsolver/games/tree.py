"""Compiled game trees and vectorised tree passes.

``compile_tree(game)`` walks a ``Game`` once and stores it as flat numpy arrays: nodes grouped
by depth, decision children per action, chance edges, infosets and the players' own-action
*sequences* (sequence-form). Solvers then run each pass (realization plans, expected values,
best response) as a handful of numpy operations per depth instead of a Python loop over nodes,
which is what makes exact CFR+/BR/EV cheap even for Leduc (~15k nodes).

Requirements on the game: perfect recall, and every node of an infoset at the same depth (true
for games whose infoset key contains the full public history).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from exsolver.games.base import CALL, N_ACTIONS, Game

TERMINAL, CHANCE, DECISION = 0, 1, 2
_TIE_TOL = 1e-12


@dataclass(frozen=True)
class Level:
    """All nodes at one depth; their values are computed in a single backward step."""

    depth: int
    dec_all: np.ndarray  # decision nodes at this depth
    dec: tuple[np.ndarray, np.ndarray]  # decision nodes per acting player
    dec_infosets: tuple[np.ndarray, np.ndarray]  # unique infoset ids per player at this depth
    dec_inv: tuple[np.ndarray, np.ndarray]  # position of each node's infoset in dec_infosets
    chance_src: np.ndarray  # chance edges leaving this depth
    chance_dst: np.ndarray
    chance_prob: np.ndarray


@dataclass(frozen=True)
class CompiledTree:
    """Flat representation of a two-player zero-sum game tree (see module docstring).

    Node index ``n_nodes`` is a sentinel child for illegal actions; value arrays therefore have
    length ``n_nodes + 1`` with the sentinel fixed at 0. Sequence 0 is the empty sequence.
    """

    n_nodes: int
    n_infosets: int
    n_sequences: int
    kind: np.ndarray  # [N] TERMINAL | CHANCE | DECISION
    player: np.ndarray  # [N] acting player at decision nodes, -1 elsewhere
    infoset: np.ndarray  # [N] infoset id at decision nodes, -1 elsewhere
    depth: np.ndarray  # [N]
    child: np.ndarray  # [N, 3] child node per action (sentinel n_nodes when illegal)
    util0: np.ndarray  # [N] seat-0 return at terminals, 0 elsewhere
    reach_chance: np.ndarray  # [N] product of chance probabilities on the path to the node
    node_seq: np.ndarray  # [2, N] each player's last own sequence on the path to the node
    terminals: np.ndarray
    dec_nodes: tuple[np.ndarray, np.ndarray]  # all decision nodes per player
    infoset_keys: tuple[str, ...]
    infoset_index: dict[str, int]
    infoset_player: np.ndarray  # [I]
    infoset_legal: np.ndarray  # [I, 3] bool
    infoset_depth: np.ndarray  # [I]
    infoset_parent_seq: np.ndarray  # [I] own sequence leading to the infoset
    infoset_seq: np.ndarray  # [I, 3] sequence id of (infoset, action); 0 when illegal
    infoset_rows: tuple[np.ndarray, np.ndarray]  # infoset ids per player
    seq_levels: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]  # (seq, infoset, action)
    levels: tuple[Level, ...]  # deepest first
    root: int = 0

    # -- strategies -------------------------------------------------------------------------
    def uniform_dense(self) -> np.ndarray:
        """[I, 3] uniform-over-legal strategy for both players."""
        legal = self.infoset_legal.astype(np.float64)
        return legal / legal.sum(axis=1, keepdims=True)

    def dense_from_strategy(
        self, strat: dict[str, np.ndarray], seats: tuple[int, ...] = (0, 1)
    ) -> np.ndarray:
        """[I, 3] array holding ``strat`` at the infosets of ``seats`` (zeros elsewhere).

        Raises ``KeyError`` for a missing infoset and ``ValueError`` (naming the offending key)
        for rows with negative or non-finite entries, mass on illegal actions or a sum != 1.
        """
        sigma = np.zeros((self.n_infosets, N_ACTIONS))
        keys = self.infoset_keys
        for seat in seats:
            rows = self.infoset_rows[seat]
            try:
                sigma[rows] = np.asarray([strat[keys[i]] for i in rows], dtype=np.float64).reshape(
                    rows.size, N_ACTIONS
                )
            except KeyError as e:
                raise KeyError(f"strategy is missing infoset {e.args[0]!r}") from None
        rows = np.concatenate([self.infoset_rows[s] for s in seats])
        sub = sigma[rows]
        self._check_rows(rows, ~np.isfinite(sub).all(axis=1), "has a non-finite entry")
        self._check_rows(rows, (sub < 0.0).any(axis=1), "has a negative entry")
        illegal_mass = np.where(self.infoset_legal[rows], 0.0, sub).any(axis=1)
        self._check_rows(rows, illegal_mass, "puts probability on an illegal action")
        self._check_rows(rows, np.abs(sub.sum(axis=1) - 1.0) > 1e-6, "does not sum to 1")
        return sigma

    def _check_rows(self, rows: np.ndarray, bad: np.ndarray, what: str) -> None:
        if bad.any():
            key = self.infoset_keys[int(rows[np.flatnonzero(bad)[0]])]
            raise ValueError(f"strategy row {key!r} {what}")

    def strategy_from_dense(
        self, sigma: np.ndarray, seats: tuple[int, ...] = (0, 1)
    ) -> dict[str, np.ndarray]:
        keys = self.infoset_keys
        return {keys[i]: sigma[i].copy() for seat in seats for i in self.infoset_rows[seat]}

    # -- forward passes ---------------------------------------------------------------------
    def realization(self, sigma: np.ndarray) -> np.ndarray:
        """Realization weight of every own-action sequence under ``sigma`` ([I, 3]) -> [S]."""
        rho = np.ones(self.n_sequences)
        parent = self.infoset_parent_seq
        for seq, inf, act in self.seq_levels:
            rho[seq] = rho[parent[inf]] * sigma[inf, act]
        return rho

    def realization_batch(self, sigma: np.ndarray) -> np.ndarray:
        """Batched ``realization`` for ``sigma`` of shape [M, I, 3] -> [M, S]."""
        rho = np.ones((sigma.shape[0], self.n_sequences))
        parent = self.infoset_parent_seq
        for seq, inf, act in self.seq_levels:
            rho[:, seq] = rho[:, parent[inf]] * sigma[:, inf, act]
        return rho

    def reach(self, rho: np.ndarray, player: int) -> np.ndarray:
        """[N] probability that ``player``'s own actions lead to each node."""
        return rho[self.node_seq[player]]

    # -- backward passes --------------------------------------------------------------------
    def values(self, sigma: np.ndarray) -> np.ndarray:
        """Seat-0 expected utility of every node's subtree under ``sigma`` -> [N + 1]."""
        u = np.zeros(self.n_nodes + 1)
        u[self.terminals] = self.util0[self.terminals]
        infoset, child = self.infoset, self.child
        for lvl in self.levels:
            dec = lvl.dec_all
            if dec.size:
                u[dec] = np.einsum("ij,ij->i", sigma[infoset[dec]], u[child[dec]])
            if lvl.chance_src.size:
                np.add.at(u, lvl.chance_src, lvl.chance_prob * u[lvl.chance_dst])
        return u

    def best_response_pass(
        self, sigma: np.ndarray, br_player: int, cf_reach: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact best response of ``br_player`` against ``sigma`` (other player's rows).

        ``cf_reach[n]`` must be ``reach_chance[n] * reach_opponent[n]``. Returns the seat-0 node
        values under (BR, sigma) and the BR action per infoset (-1 for the opponent's infosets).
        Ties go to the lowest-index maximising action. Infosets the opponent never lets the BR
        player reach (zero counterfactual reach, so every action value is 0) get CALL when legal
        and otherwise the lowest legal action, so the BR is passive where the choice is immaterial.
        """
        u = np.zeros(self.n_nodes + 1)
        u[self.terminals] = self.util0[self.terminals]
        sign = 1.0 if br_player == 0 else -1.0
        opp = 1 - br_player
        infoset, child, legal = self.infoset, self.child, self.infoset_legal
        a_star = np.full(self.n_infosets, -1, dtype=np.int64)
        for lvl in self.levels:
            dec_o = lvl.dec[opp]
            if dec_o.size:
                u[dec_o] = np.einsum("ij,ij->i", sigma[infoset[dec_o]], u[child[dec_o]])
            dec_b = lvl.dec[br_player]
            if dec_b.size:
                infs, inv = lvl.dec_infosets[br_player], lvl.dec_inv[br_player]
                cu = u[child[dec_b]] * (sign * cf_reach[dec_b])[:, None]
                q = np.empty((infs.size, N_ACTIONS))
                for a in range(N_ACTIONS):
                    q[:, a] = np.bincount(inv, weights=cu[:, a], minlength=infs.size)
                q = np.where(legal[infs], q, -np.inf)
                best = q.max(axis=1)
                tol = _TIE_TOL * np.maximum(1.0, np.abs(best))
                a_best = np.argmax(q >= (best - tol)[:, None], axis=1)
                reach_i = np.bincount(inv, weights=cf_reach[dec_b], minlength=infs.size)
                unreachable = reach_i <= 0.0
                if unreachable.any():
                    passive = np.where(legal[infs, CALL], CALL, np.argmax(legal[infs], axis=1))
                    a_best = np.where(unreachable, passive, a_best)
                a_star[infs] = a_best
                u[dec_b] = u[child[dec_b, a_best[inv]]]
            if lvl.chance_src.size:
                np.add.at(u, lvl.chance_src, lvl.chance_prob * u[lvl.chance_dst])
        return u, a_star


def _compile(game: Game) -> CompiledTree:
    kind: list[int] = []
    player: list[int] = []
    infoset: list[int] = []
    depth: list[int] = []
    child: list[list[int]] = []
    util0: list[float] = []
    reach_c: list[float] = []
    node_seq: tuple[list[int], list[int]] = ([], [])
    c_src: list[int] = []
    c_dst: list[int] = []
    c_prob: list[float] = []
    inf_index: dict[str, int] = {}
    inf_keys: list[str] = []
    inf_player: list[int] = []
    inf_legal: list[list[bool]] = []
    inf_depth: list[int] = []
    inf_parent_seq: list[int] = []
    inf_seq: list[list[int]] = []
    n_seq = 1  # sequence 0 is the empty sequence

    # stack entries: (state, depth, parent, action, chance prob, seq0, seq1); action -1 = chance
    stack = [(game.root(), 0, -1, -1, 1.0, 0, 0)]
    while stack:
        s, d, par, a, prob, sq0, sq1 = stack.pop()
        n = len(kind)
        depth.append(d)
        node_seq[0].append(sq0)
        node_seq[1].append(sq1)
        if par < 0:
            reach_c.append(1.0)
        elif a >= 0:
            child[par][a] = n
            reach_c.append(reach_c[par])
        else:
            c_src.append(par)
            c_dst.append(n)
            c_prob.append(prob)
            reach_c.append(reach_c[par] * prob)
        child.append([-1] * N_ACTIONS)
        if game.is_terminal(s):
            kind.append(TERMINAL)
            player.append(-1)
            infoset.append(-1)
            r = game.returns(s)
            if abs(r[0] + r[1]) > 1e-12:
                raise ValueError(f"returns are not zero-sum at {s}: {r}")
            util0.append(float(r[0]))
        elif game.is_chance(s):
            kind.append(CHANCE)
            player.append(-1)
            infoset.append(-1)
            util0.append(0.0)
            for s2, p2 in reversed(game.chance_outcomes(s)):
                stack.append((s2, d + 1, n, -1, float(p2), sq0, sq1))
        else:
            p = game.current_player(s)
            key = game.infoset_key(s, p)
            legal = list(game.legal_actions(s))
            own_seq = sq0 if p == 0 else sq1
            i = inf_index.get(key)
            if i is None:
                i = len(inf_keys)
                inf_index[key] = i
                inf_keys.append(key)
                inf_player.append(p)
                inf_depth.append(d)
                inf_parent_seq.append(own_seq)
                mask, seqs = [False] * N_ACTIONS, [0] * N_ACTIONS
                for a2 in legal:
                    mask[a2] = True
                    seqs[a2] = n_seq
                    n_seq += 1
                inf_legal.append(mask)
                inf_seq.append(seqs)
            elif inf_player[i] != p or inf_depth[i] != d or inf_parent_seq[i] != own_seq:
                raise ValueError(f"infoset {key!r} violates perfect recall or depth consistency")
            elif inf_legal[i] != [x in legal for x in range(N_ACTIONS)]:
                raise ValueError(f"infoset {key!r} has inconsistent legal actions")
            kind.append(DECISION)
            player.append(p)
            infoset.append(i)
            util0.append(0.0)
            for a2 in reversed(legal):
                nsq = inf_seq[i][a2]
                stack.append(
                    (
                        game.apply(s, a2),
                        d + 1,
                        n,
                        a2,
                        0.0,
                        nsq if p == 0 else sq0,
                        nsq if p == 1 else sq1,
                    )
                )

    n_nodes = len(kind)
    kind_a = np.asarray(kind, dtype=np.int8)
    player_a = np.asarray(player, dtype=np.int8)
    infoset_a = np.asarray(infoset, dtype=np.int64)
    depth_a = np.asarray(depth, dtype=np.int64)
    child_a = np.asarray(child, dtype=np.int64).reshape(n_nodes, N_ACTIONS)
    child_a[child_a < 0] = n_nodes
    c_src_a = np.asarray(c_src, dtype=np.int64)
    c_dst_a = np.asarray(c_dst, dtype=np.int64)
    c_prob_a = np.asarray(c_prob, dtype=np.float64)
    inf_player_a = np.asarray(inf_player, dtype=np.int64)
    inf_legal_a = np.asarray(inf_legal, dtype=bool).reshape(len(inf_keys), N_ACTIONS)
    inf_depth_a = np.asarray(inf_depth, dtype=np.int64)

    levels: list[Level] = []
    for d in range(int(depth_a.max()), -1, -1):
        at_d = depth_a == d
        dec_all = np.flatnonzero(at_d & (kind_a == DECISION))
        dec, dec_infs, dec_inv = [], [], []
        for p in (0, 1):
            nodes = dec_all[player_a[dec_all] == p]
            infs, inv = np.unique(infoset_a[nodes], return_inverse=True)
            dec.append(nodes)
            dec_infs.append(infs.astype(np.int64))
            dec_inv.append(inv.reshape(-1).astype(np.int64))
        edges = np.flatnonzero(depth_a[c_src_a] == d) if c_src_a.size else np.zeros(0, np.int64)
        levels.append(
            Level(
                depth=d,
                dec_all=dec_all,
                dec=(dec[0], dec[1]),
                dec_infosets=(dec_infs[0], dec_infs[1]),
                dec_inv=(dec_inv[0], dec_inv[1]),
                chance_src=c_src_a[edges],
                chance_dst=c_dst_a[edges],
                chance_prob=c_prob_a[edges],
            )
        )

    inf_seq_a = np.asarray(inf_seq, dtype=np.int64).reshape(len(inf_keys), N_ACTIONS)
    seq_levels = []
    for d in np.unique(inf_depth_a):
        infs = np.flatnonzero(inf_depth_a == d)
        inf_rep = np.repeat(infs, N_ACTIONS)
        act_rep = np.tile(np.arange(N_ACTIONS), infs.size)
        keep = inf_legal_a[inf_rep, act_rep]
        seq_levels.append((inf_seq_a[inf_rep, act_rep][keep], inf_rep[keep], act_rep[keep]))

    return CompiledTree(
        n_nodes=n_nodes,
        n_infosets=len(inf_keys),
        n_sequences=n_seq,
        kind=kind_a,
        player=player_a,
        infoset=infoset_a,
        depth=depth_a,
        child=child_a,
        util0=np.asarray(util0, dtype=np.float64),
        reach_chance=np.asarray(reach_c, dtype=np.float64),
        node_seq=np.asarray(node_seq, dtype=np.int64).reshape(2, n_nodes),
        terminals=np.flatnonzero(kind_a == TERMINAL),
        dec_nodes=tuple(np.flatnonzero((kind_a == DECISION) & (player_a == p)) for p in (0, 1)),
        infoset_keys=tuple(inf_keys),
        infoset_index=inf_index,
        infoset_player=inf_player_a,
        infoset_legal=inf_legal_a,
        infoset_depth=inf_depth_a,
        infoset_parent_seq=np.asarray(inf_parent_seq, dtype=np.int64),
        infoset_seq=inf_seq_a,
        infoset_rows=tuple(np.flatnonzero(inf_player_a == p) for p in (0, 1)),
        seq_levels=tuple(seq_levels),
        levels=tuple(levels),
    )


@lru_cache(maxsize=8)
def compile_tree(game: Game) -> CompiledTree:
    """Compile ``game`` once per distinct game (games hash by their spec) and cache it."""
    return _compile(game)
