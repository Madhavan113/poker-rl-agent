"""Perturbed-CFR opponent prior for Leduc hold'em (RESEARCH.md §3.3, E5 spec §Population).

An opponent is a full behavioural profile (both seats) built from the CFR+ equilibrium ``s*``:

    logit(a | I) = log s~(a | I) + beta * b_arch(a, I) + eps(I, a),    eps ~ N(0, noise^2)
    s~ = (1 - smooth) * s* + smooth * uniform-over-legal

``b_arch`` is the archetype's leak, a fixed bias per (action, situation) where the situation is
whether the player faces a bet (FOLD is legal) and its hand-strength bucket; ``beta`` scales the
leak per opponent. Smoothing keeps every legal action's probability positive, so every observed
hand has positive likelihood under every opponent (posteriors never collapse on a lie).

Hand-strength buckets: before the board J = weak, Q = mid, K = strong; after it a pair is
strong, an unpaired K is mid and anything else weak.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from exsolver.games.base import FOLD, N_ACTIONS
from exsolver.games.leduc import LeducPoker
from exsolver.games.tree import compile_tree
from exsolver.solvers.cfr import CFRPlus

WEAK, MID, STRONG = 0, 1, 2


def strength_bucket(key: str) -> int:
    """Hand-strength bucket of a Leduc infoset key (``"1:Qs|cbc/Kh|c"``)."""
    rank = key[2]
    _, _, rest = key.partition("|")
    if "/" in rest:
        board_rank = rest.partition("/")[2][0]
        if board_rank == rank:
            return STRONG
        return MID if rank == "K" else WEAK
    return {"J": WEAK, "Q": MID, "K": STRONG}[rank]


@dataclass(frozen=True)
class LeducArchetype:
    """A leak: logit biases ``[facing_bet][bucket] -> (fold, call, raise)`` and a weight."""

    name: str
    weight: float
    bias: tuple[tuple[tuple[float, float, float], ...], ...]  # [2][3][3]


def default_leduc_archetypes() -> tuple[LeducArchetype, ...]:
    """Five archetypes: near-equilibrium, maniac, calling station, rock, passive.

    Rows are (not facing a bet, facing a bet); columns weak / mid / strong; entries are logit
    biases on (FOLD, CALL, RAISE). FOLD is illegal when not facing a bet, so its entry there is
    unused.
    """
    zero = ((0.0, 0.0, 0.0),) * 3
    return (
        LeducArchetype("equilibrium", 0.2, (zero, zero)),
        LeducArchetype(
            "maniac",
            0.2,
            (
                ((0.0, 0.0, 1.5), (0.0, 0.0, 1.5), (0.0, 0.0, 1.0)),
                ((-1.0, 0.0, 1.0), (-1.0, 0.0, 1.0), (-1.0, 0.0, 1.0)),
            ),
        ),
        LeducArchetype(
            "station",
            0.2,
            (
                ((0.0, 0.5, 0.0), (0.0, 0.5, 0.0), (0.0, 0.5, 0.0)),
                ((-2.0, 1.5, -0.5), (-2.0, 1.5, -0.5), (-1.0, 1.0, -0.5)),
            ),
        ),
        LeducArchetype(
            "rock",
            0.2,
            (
                ((0.0, 0.0, -1.5), (0.0, 0.0, -0.5), (0.0, 0.0, 0.0)),
                ((1.5, 0.0, -1.5), (1.0, 0.0, -0.5), (0.0, 0.0, 0.0)),
            ),
        ),
        LeducArchetype(
            "passive",
            0.2,
            (
                ((0.0, 0.0, -2.0), (0.0, 0.0, -2.0), (0.0, 0.0, -1.5)),
                ((0.0, 0.5, -2.0), (0.0, 0.5, -2.0), (0.0, 0.5, -1.5)),
            ),
        ),
    )


@dataclass
class LeducPrior:
    """Archetype mixture of perturbed equilibria; ``sample_dense`` draws one opponent."""

    archetypes: tuple[LeducArchetype, ...] = field(default_factory=default_leduc_archetypes)
    beta_range: tuple[float, float] = (0.5, 1.5)
    noise: float = 0.15
    smooth: float = 0.05
    cfr_iterations: int = 2000

    def __post_init__(self) -> None:
        if abs(sum(a.weight for a in self.archetypes) - 1.0) > 1e-9:
            raise ValueError("archetype weights must sum to 1")
        self.game = LeducPoker()
        self.tree = compile_tree(self.game)
        legal = self.tree.infoset_legal.astype(np.float64)
        uniform = legal / legal.sum(axis=1, keepdims=True)
        self.equilibrium = CFRPlus(self.game).run(self.cfr_iterations).average_dense()
        base = (1.0 - self.smooth) * self.equilibrium + self.smooth * uniform
        with np.errstate(divide="ignore"):
            self._log_base = np.where(legal > 0, np.log(np.where(legal > 0, base, 1.0)), -np.inf)
        self._legal = legal.astype(bool)
        keys = self.tree.infoset_keys
        self._facing = self._legal[:, FOLD].astype(int)
        self._bucket = np.array([strength_bucket(k) for k in keys])
        self._bias = np.stack(
            [
                np.asarray(a.bias, dtype=np.float64)[self._facing, self._bucket]
                for a in self.archetypes
            ]
        )  # [A, I, 3]

    @property
    def names(self) -> list[str]:
        return [a.name for a in self.archetypes]

    @property
    def weights(self) -> np.ndarray:
        return np.array([a.weight for a in self.archetypes])

    def dense_for(self, archetype_id: int, beta: float, eps: np.ndarray) -> np.ndarray:
        """Profile ``[I, 3]`` for a given archetype, leak scale and logit noise ``[I, 3]``."""
        logits = self._log_base + beta * self._bias[archetype_id] + eps
        logits = np.where(self._legal, logits, -np.inf)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.where(self._legal, np.exp(logits), 0.0)
        return p / p.sum(axis=1, keepdims=True)

    def sample_dense(self, rng: np.random.Generator) -> tuple[np.ndarray, int]:
        """Draw ``(profile [I, 3], archetype_id)``."""
        k = int(rng.choice(len(self.archetypes), p=self.weights))
        beta = float(rng.uniform(*self.beta_range))
        eps = rng.normal(0.0, self.noise, size=(self.tree.n_infosets, N_ACTIONS))
        return self.dense_for(k, beta, eps), k

    def sample_population(self, m: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """``m`` opponents: ``(profiles [m, I, 3], archetype ids [m])``."""
        if m < 1:
            raise ValueError("population size must be >= 1")
        out = np.empty((m, self.tree.n_infosets, N_ACTIONS))
        ids = np.empty(m, dtype=np.int64)
        for i in range(m):
            out[i], ids[i] = self.sample_dense(rng)
        return out, ids
