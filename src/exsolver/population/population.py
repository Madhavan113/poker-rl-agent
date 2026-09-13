"""A discrete opponent population sampled from a prior, with ``.npz`` persistence."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from exsolver.population.kuhn_prior import (
    ARCHETYPE_NAMES,
    KUHN_PARAM_NAMES,
    N_PARAMS,
    KuhnPrior,
    theta_to_profile,
)
from exsolver.strategy import TabularStrategy


@dataclass
class Population:
    """``M`` opponents: behavioural profiles, their parameters, archetype ids and prior weights."""

    profiles: list[TabularStrategy]
    thetas: np.ndarray  # [M, 12]
    archetypes: np.ndarray  # [M] archetype id
    weights: np.ndarray  # [M] sampling weights, sum to 1
    archetype_names: list[str] = field(default_factory=lambda: list(ARCHETYPE_NAMES))

    def __post_init__(self) -> None:
        m = len(self.profiles)
        self.thetas = np.asarray(self.thetas, dtype=np.float64).reshape(m, N_PARAMS)
        self.archetypes = np.asarray(self.archetypes, dtype=np.int64).reshape(m)
        self.weights = np.asarray(self.weights, dtype=np.float64).reshape(m)
        if m and abs(float(self.weights.sum()) - 1.0) > 1e-9:
            raise ValueError("population weights must sum to 1")

    def __len__(self) -> int:
        return len(self.profiles)

    @classmethod
    def from_thetas(
        cls,
        thetas: np.ndarray,
        archetypes: np.ndarray,
        weights: np.ndarray | None = None,
        archetype_names: list[str] | None = None,
    ) -> Population:
        thetas = np.asarray(thetas, dtype=np.float64)
        m = thetas.shape[0]
        if weights is None:
            weights = np.full(m, 1.0 / m) if m else np.zeros(0)
        return cls(
            profiles=[theta_to_profile(t) for t in thetas],
            thetas=thetas,
            archetypes=np.asarray(archetypes),
            weights=np.asarray(weights),
            archetype_names=list(archetype_names or ARCHETYPE_NAMES),
        )

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write ``thetas``, ``archetypes``, ``weights`` and the name tables to an ``.npz``."""
        np.savez(
            os.fspath(path),
            thetas=self.thetas,
            archetypes=self.archetypes,
            weights=self.weights,
            archetype_names=np.asarray(self.archetype_names, dtype=str),
            param_names=np.asarray(KUHN_PARAM_NAMES, dtype=str),
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Population:
        with np.load(os.fspath(path)) as d:
            if list(d["param_names"]) != KUHN_PARAM_NAMES:
                raise ValueError("population file uses a different parameter order")
            return cls.from_thetas(
                d["thetas"], d["archetypes"], d["weights"], [str(n) for n in d["archetype_names"]]
            )


def sample_population(prior: KuhnPrior, m: int, rng: np.random.Generator) -> Population:
    """Draw ``m`` opponents from ``prior`` with uniform weights."""
    if m < 1:
        raise ValueError("population size must be >= 1")
    thetas = np.empty((m, N_PARAMS))
    archetypes = np.empty(m, dtype=np.int64)
    for i in range(m):
        thetas[i], archetypes[i] = prior.sample_theta(rng)
    return Population.from_thetas(thetas, archetypes, np.full(m, 1.0 / m), prior.names)
