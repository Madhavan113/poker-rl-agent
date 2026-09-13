"""Generative prior over Kuhn opponents: an archetype mixture with Beta noise.

An opponent is a full behavioural profile over both seats: 12 infosets, one free probability
each (``KUHN_PARAM_NAMES`` fixes the order). Each archetype has a centre ``m`` in [0, 1]^12 and a
concentration ``kappa``; a sample draws every parameter from ``Beta(m*kappa, (1-m)*kappa)`` with
centres clipped to ``[0.02, 0.98]``. The "equilibrium" archetype re-draws its centre from the
Nash family (``alpha ~ U[0, 1/3]``) for every sample; "random" is ``Beta(1, 1)`` throughout.

Only the archetype *centres* are clipped to ``[0.02, 0.98]``; the Beta draws themselves are not
(they already lie in [0, 1]). The clip alone moves the equilibrium archetype's centre off the
Nash family: the clipped centre is exploitable by about 0.012 chips/hand, so roughly 0.012 of
that archetype's ~0.045 mean oracle gain is due to the clip rather than to the Beta noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from exsolver.games.base import CALL, FOLD, N_ACTIONS, RAISE, Action
from exsolver.strategy import TabularStrategy

KUHN_PARAM_NAMES: list[str] = [
    "0:J|", "0:Q|", "0:K|",  # P(RAISE) at the seat-0 root
    "0:J|cb", "0:Q|cb", "0:K|cb",  # P(CALL) facing a bet after checking
    "1:J|c", "1:Q|c", "1:K|c",  # P(RAISE) after the opponent checks
    "1:J|b", "1:Q|b", "1:K|b",  # P(CALL) facing a bet
]  # fmt: skip
N_PARAMS = len(KUHN_PARAM_NAMES)
# action whose probability the parameter gives, and the action taking the remaining mass
PARAM_ACTIONS: tuple[Action, ...] = (RAISE,) * 3 + (CALL,) * 3 + (RAISE,) * 3 + (CALL,) * 3
_OTHER_ACTIONS: tuple[Action, ...] = (CALL,) * 3 + (FOLD,) * 3 + (CALL,) * 3 + (FOLD,) * 3

ARCHETYPE_NAMES: list[str] = ["equilibrium", "maniac", "station", "rock", "random"]


def theta_to_profile(theta: np.ndarray) -> TabularStrategy:
    """12-vector of parameters -> behavioural profile (length-3 vectors, zeros on illegal actions)."""
    th = np.asarray(theta, dtype=np.float64).reshape(N_PARAMS)
    if np.any(th < 0.0) or np.any(th > 1.0):
        raise ValueError("theta entries must lie in [0, 1]")
    profile: TabularStrategy = {}
    for name, a, other, t in zip(KUHN_PARAM_NAMES, PARAM_ACTIONS, _OTHER_ACTIONS, th, strict=True):
        v = np.zeros(N_ACTIONS)
        v[a] = t
        v[other] = 1.0 - t
        profile[name] = v
    return profile


def profile_to_theta(profile: TabularStrategy) -> np.ndarray:
    """Inverse of ``theta_to_profile`` (reads the parameterised action's probability)."""
    return np.array(
        [float(profile[name][a]) for name, a in zip(KUHN_PARAM_NAMES, PARAM_ACTIONS, strict=True)]
    )


def nash_theta(alpha: float) -> np.ndarray:
    """Kuhn Nash equilibrium with bluffing parameter ``alpha`` in [0, 1/3], as a theta vector."""
    if not 0.0 <= alpha <= 1.0 / 3.0:
        raise ValueError("alpha must lie in [0, 1/3]")
    third = 1.0 / 3.0
    return np.array(
        [alpha, 0.0, 3.0 * alpha, 0.0, alpha + third, 1.0, third, 0.0, 1.0, 0.0, third, 1.0]
    )


@dataclass(frozen=True)
class Archetype:
    """Mixture component: ``kappa=None`` means Beta(1, 1) noise; ``centre=None`` with a kappa
    means the centre is a fresh Nash strategy per sample."""

    name: str
    weight: float
    kappa: float | None = None
    centre: tuple[float, ...] | None = None


def default_archetypes() -> tuple[Archetype, ...]:
    """The E1 archetypes, in ``ARCHETYPE_NAMES`` order.

    Centres are listed in ``KUHN_PARAM_NAMES`` order: seat-0 bets J/Q/K, seat-0 calls J/Q/K,
    seat-1 bets J/Q/K, seat-1 calls J/Q/K.
    """
    return (
        Archetype("equilibrium", 0.30, 30.0, None),
        Archetype(
            "maniac", 0.15, 10.0, (0.7, 0.8, 0.95, 0.5, 0.85, 1.0, 0.7, 0.7, 0.95, 0.5, 0.9, 1.0)
        ),
        Archetype(
            "station", 0.20, 10.0, (0.05, 0.15, 0.8, 0.6, 0.95, 1.0, 0.05, 0.1, 0.9, 0.6, 0.95, 1.0)
        ),
        Archetype(
            "rock",
            0.20,
            10.0,
            (0.02, 0.05, 0.9, 0.02, 0.2, 0.95, 0.02, 0.05, 0.95, 0.02, 0.15, 1.0),
        ),
        Archetype("random", 0.15, None, None),
    )


@dataclass
class KuhnPrior:
    """Archetype mixture with Beta noise over 12-dimensional Kuhn profiles."""

    archetypes: tuple[Archetype, ...] = field(default_factory=default_archetypes)
    centre_clip: tuple[float, float] = (0.02, 0.98)
    alpha_max: float = 1.0 / 3.0

    def __post_init__(self) -> None:
        names = [a.name for a in self.archetypes]
        if len(set(names)) != len(names):
            raise ValueError("archetype names must be unique")
        if abs(float(np.sum(self.weights)) - 1.0) > 1e-9:
            raise ValueError("archetype weights must sum to 1")
        for a in self.archetypes:
            if a.centre is not None and len(a.centre) != N_PARAMS:
                raise ValueError(f"archetype {a.name!r} needs {N_PARAMS} centre values")

    @property
    def names(self) -> list[str]:
        return [a.name for a in self.archetypes]

    @property
    def weights(self) -> np.ndarray:
        return np.array([a.weight for a in self.archetypes], dtype=np.float64)

    def centre(self, archetype_id: int, rng: np.random.Generator) -> np.ndarray | None:
        """Clipped Beta centre of an archetype (draws alpha for the Nash family); None if uniform."""
        arch = self.archetypes[archetype_id]
        if arch.kappa is None:
            return None
        if arch.centre is None:
            m = nash_theta(float(rng.uniform(0.0, self.alpha_max)))
        else:
            m = np.asarray(arch.centre, dtype=np.float64)
        return np.clip(m, self.centre_clip[0], self.centre_clip[1])

    def sample_theta(self, rng: np.random.Generator) -> tuple[np.ndarray, int]:
        """Draw ``(theta[12], archetype_id)``."""
        k = int(rng.choice(len(self.archetypes), p=self.weights))
        arch = self.archetypes[k]
        m = self.centre(k, rng)
        if m is None:
            theta = rng.beta(np.ones(N_PARAMS), np.ones(N_PARAMS))
        else:
            kappa = float(arch.kappa)  # type: ignore[arg-type]
            theta = rng.beta(m * kappa, (1.0 - m) * kappa)
        return np.clip(theta, 0.0, 1.0), k

    def sample(self, rng: np.random.Generator) -> tuple[TabularStrategy, np.ndarray, int]:
        """Draw ``(profile, theta[12], archetype_id)``."""
        theta, k = self.sample_theta(rng)
        return theta_to_profile(theta), theta, k
