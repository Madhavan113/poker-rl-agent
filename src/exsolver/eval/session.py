"""Session runner: play ``H`` hands against one opponent and record exact per-hand metrics.

The evaluator is omniscient (it holds the opponent profile ``theta`` and, optionally, the exact
posterior) and uses that knowledge **only** to compute metrics. The agent under test sees the
``HandRecord``s of the hands it played and nothing else. Nothing here is Monte Carlo: ``ev`` and
``expl`` are game-tree traversals of the agent's hand-``t`` strategy; ``realized`` is logged only
as a sanity check.

Deviations from docs/experiments/e1-kuhn.md ("Session runner and metrics"):

* signature: ``run_session(game, agent, opp_profile, H, rng, *, posterior=None,
  equilibrium_value_seat0, probe_keys=None, opp_id=-1, archetype=-1, session=0, n_opp=None)``
  instead of the spec's ``run_session(game, agent, theta_profile, opp_id, population, H, rng)``.
  The exact posterior is passed in as a duck-typed object (``update(hand)``, ``probs``,
  ``entropy()``) rather than built from the population, so the runner does not depend on
  ``exsolver.bayes``; ``opp_id`` / ``archetype`` / ``session`` are bookkeeping that lets
  ``aggregate`` pair the sessions of different agents (same opponent, same deals); ``n_opp`` is
  the population size used to validate belief lengths when no posterior is given.
* the ``Agent`` protocol has ``observe(hand)``, called after every completed hand; the spec lists
  only ``reset`` / ``strategy_for_hand`` / ``belief``. Stateful agents update there.
* ``rng`` is split into four independent streams (deal, opponent actions, agent action sampling,
  agent-internal randomness handed to ``reset``) so that agents seeded identically face identical
  deals; the spec leaves the seeding unspecified.
* ``probe`` is recorded per named probe with one infoset key per seat (``KUHN_PROBES``: bluff =
  P(RAISE) with J, calldown = P(CALL) with Q facing a bet) and is NaN on hands where the agent's
  seat has no key.
* ``agent.belief`` is validated (1-D, finite, non-negative, sums to 1 within 1e-6, length ``M``
  taken from the posterior, else from ``n_opp``, else pinned by the first belief of the session)
  and a violation raises instead of producing NaNs; ``kl_divergence`` floors the belief at 1e-30
  wherever the exact posterior has mass, so a zero-mass miss contributes at most
  ``-log(1e-30) ~= 69.08`` nats per element (never inf). ``entropy`` and ``kl_divergence`` reject
  non-finite or negative inputs themselves, so standalone callers get errors too.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from exsolver.data.records import HandRecord, SessionContext
from exsolver.games.base import CALL, RAISE, Game
from exsolver.play import play_hand
from exsolver.solvers.best_response import expected_value, seat_exploitability
from exsolver.strategy import TabularStrategy, seat_of_key

Probe = tuple[str, int]  # (infoset key, action)
ProbeSpec = Mapping[str, Probe | Sequence[Probe]]

# Kuhn probes (one key per seat): bluff = P(RAISE) with J at the first opportunity,
# calldown = P(CALL) with Q facing a bet.
KUHN_PROBES: dict[str, tuple[Probe, ...]] = {
    "bluff": (("0:J|", RAISE), ("1:J|c", RAISE)),
    "calldown": (("0:Q|cb", CALL), ("1:Q|b", CALL)),
}
BELIEF_FLOOR = 1e-30  # KL guard: a belief of exactly 0 where the truth has mass is capped here
BELIEF_FLOOR_CAP_NATS = -math.log(BELIEF_FLOOR)  # ~69.08 nats per element
BELIEF_SUM_TOL = 1e-6


class Agent(Protocol):
    """Protocol the evaluator drives (mirrors ``exsolver.agents.base.Agent``)."""

    name: str

    def reset(self, rng: np.random.Generator) -> None: ...

    def observe(self, hand: HandRecord) -> None: ...

    def strategy_for_hand(self, ctx: SessionContext, seat: int) -> TabularStrategy: ...

    def belief(self, ctx: SessionContext) -> np.ndarray | None: ...


class Posterior(Protocol):
    """Duck-typed exact posterior over a discrete population (``exsolver.bayes``)."""

    probs: np.ndarray

    def update(self, hand: HandRecord) -> None: ...

    def entropy(self) -> float: ...


class FixedStrategyAgent:
    """Evaluation utility / test double: plays a fixed profile, learns nothing, has no belief."""

    def __init__(self, strategy: TabularStrategy, name: str = "Fixed") -> None:
        self.strategy = strategy
        self.name = name
        self.observed: list[HandRecord] = []

    def reset(self, rng: np.random.Generator) -> None:
        self.observed = []

    def observe(self, hand: HandRecord) -> None:
        self.observed.append(hand)

    def strategy_for_hand(self, ctx: SessionContext, seat: int) -> TabularStrategy:
        return self.strategy

    def belief(self, ctx: SessionContext) -> np.ndarray | None:
        return None


@dataclass
class SessionMetrics:
    """Per-hand metrics of one session (arrays of length ``H``; NaN where undefined)."""

    agent: str
    opp_id: int
    archetype: int
    session: int
    seats: np.ndarray  # int, agent's seat per hand
    ev: np.ndarray  # exact EV of the hand-t strategy vs theta, agent's perspective
    realized: np.ndarray  # chips actually won (sanity only)
    expl: np.ndarray  # seat_exploitability of the hand-t strategy
    post_entropy: np.ndarray  # exact posterior entropy before hand t (nats)
    agent_entropy: np.ndarray  # entropy of agent.belief before hand t (nats)
    kl: np.ndarray  # KL(exact || agent.belief) before hand t (nats)
    probes: dict[str, np.ndarray] = field(default_factory=dict)
    n_showdowns: int = 0

    @property
    def H(self) -> int:
        return int(self.ev.shape[0])

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "agent": self.agent,
            "opp_id": self.opp_id,
            "archetype": self.archetype,
            "session": self.session,
            "n_showdowns": self.n_showdowns,
        }
        for k in ("seats", "ev", "realized", "expl", "post_entropy", "agent_entropy", "kl"):
            out[k] = getattr(self, k).tolist()
        out["probes"] = {k: v.tolist() for k, v in self.probes.items()}
        return out


# ---------------------------------------------------------------------- information measures
def _probability_vector(p: Any, who: str) -> np.ndarray:
    """Flatten to float64 and reject non-finite or negative entries."""
    q = np.asarray(p, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(q)):
        raise ValueError(f"{who}: probabilities must be finite, got {q}")
    if np.any(q < 0.0):
        raise ValueError(f"{who}: probabilities must be non-negative, got {q}")
    return q


def entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats; zero entries contribute nothing.

    Raises ``ValueError`` on non-finite or negative entries.
    """
    q = _probability_vector(p, "entropy")
    nz = q[q > 0.0]
    return float(-(nz * np.log(nz)).sum())


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """``KL(p || q)`` in nats over the support of ``p``.

    ``q`` is floored at ``BELIEF_FLOOR`` (1e-30) where ``p > 0``: an agent that puts zero mass on
    the true opponent is charged at most ``-log(1e-30) ~= 69.08`` nats for that element instead
    of infinity. Every KL reported by the evaluator carries this cap. Raises ``ValueError`` on
    shape mismatch or non-finite / negative entries in either argument.
    """
    p = _probability_vector(p, "kl_divergence: p")
    q = _probability_vector(q, "kl_divergence: q")
    if p.shape != q.shape:
        raise ValueError(f"belief shapes differ: {p.shape} vs {q.shape}")
    sup = p > 0.0
    return float((p[sup] * (np.log(p[sup]) - np.log(np.maximum(q[sup], BELIEF_FLOOR)))).sum())


def check_belief(belief: Any, m: int | None, who: str) -> np.ndarray:
    """Validate an agent's belief: 1-D, finite, non-negative, sums to one, length ``m`` if known."""
    b = np.asarray(belief, dtype=np.float64)
    if b.ndim != 1:
        raise ValueError(f"{who}: belief must be a 1-D probability vector, got shape {b.shape}")
    if m is not None and b.shape[0] != m:
        raise ValueError(f"{who}: belief has length {b.shape[0]}, expected M={m}")
    if not np.all(np.isfinite(b)):
        raise ValueError(f"{who}: belief must be finite, got {b}")
    if np.any(b < 0.0):
        raise ValueError(f"{who}: belief has negative entries: {b}")
    total = float(b.sum())
    if abs(total - 1.0) > BELIEF_SUM_TOL:
        raise ValueError(f"{who}: belief sums to {total!r}, not 1 (tolerance {BELIEF_SUM_TOL})")
    return b


def _split_rng(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    """``n`` independent child generators (deterministic given ``rng``)."""
    try:
        return list(rng.spawn(n))
    except (AttributeError, TypeError, ValueError):  # bit generator without a seed sequence
        seeds = rng.integers(0, 2**63 - 1, size=n, dtype=np.int64)
        return [np.random.default_rng(int(s)) for s in seeds]


def _normalise_probes(game: Game, probe_keys: ProbeSpec | None) -> dict[str, tuple[Probe, ...]]:
    if probe_keys is None:
        return dict(KUHN_PROBES) if game.spec.name == "kuhn" else {}
    out: dict[str, tuple[Probe, ...]] = {}
    for name, spec in probe_keys.items():
        if len(spec) == 2 and isinstance(spec[0], str):
            pairs: tuple[Probe, ...] = ((str(spec[0]), int(spec[1])),)
        else:
            pairs = tuple((str(k), int(a)) for k, a in spec)  # type: ignore[misc]
        seats = [seat_of_key(k) for k, _ in pairs]
        if len(set(seats)) != len(seats):
            raise ValueError(f"probe {name!r} lists two keys for the same seat")
        out[name] = pairs
    return out


def seat_ev(game: Game, sigma: TabularStrategy, opp_profile: TabularStrategy, seat: int) -> float:
    """Exact EV for the agent playing ``sigma`` at ``seat`` against ``opp_profile``."""
    if seat == 0:
        return expected_value(game, sigma, opp_profile)
    return -expected_value(game, opp_profile, sigma)


def run_session(
    game: Game,
    agent: Agent,
    opp_profile: TabularStrategy,
    H: int,
    rng: np.random.Generator,
    *,
    posterior: Posterior | None = None,
    equilibrium_value_seat0: float,
    probe_keys: ProbeSpec | None = None,
    opp_id: int = -1,
    archetype: int = -1,
    session: int = 0,
    n_opp: int | None = None,
) -> SessionMetrics:
    """Play ``H`` hands (agent at seat ``t % 2``) and record exact per-hand metrics.

    ``rng`` is split into four independent streams (deal, opponent actions, agent action
    sampling, agent-internal randomness handed to ``agent.reset``), so two agents evaluated
    with generators seeded identically face the same deals. ``posterior`` (duck-typed:
    ``update(hand)``, ``probs``, ``entropy()``) is read before each hand and updated after it
    by the evaluator; the agent never sees it. ``probe_keys`` maps a probe name to a
    ``(infoset_key, action)`` pair or to one such pair per seat; a probe is NaN on hands where
    the agent's seat has no key. Defaults to ``KUHN_PROBES`` for Kuhn.

    Beliefs are validated against the population size ``M``: the posterior's length when one is
    given, else ``n_opp``, else the length of the agent's first belief (later hands must agree).
    """
    if H < 1:
        raise ValueError("H must be >= 1")
    deal_rng, opp_rng, act_rng, agent_rng = _split_rng(rng, 4)
    probes = _normalise_probes(game, probe_keys)
    probe_by_seat: dict[str, dict[int, Probe]] = {
        name: {seat_of_key(k): (k, a) for k, a in pairs} for name, pairs in probes.items()
    }
    expected_m: int | None = None if n_opp is None else int(n_opp)
    if posterior is not None:
        m_post = int(np.asarray(posterior.probs).reshape(-1).shape[0])
        if expected_m is not None and expected_m != m_post:
            raise ValueError(f"posterior has {m_post} opponents but n_opp={expected_m}")
        expected_m = m_post

    seats = np.fromiter((t % 2 for t in range(H)), dtype=np.int64, count=H)
    ev = np.full(H, np.nan)
    realized = np.full(H, np.nan)
    expl = np.full(H, np.nan)
    post_entropy = np.full(H, np.nan)
    agent_entropy = np.full(H, np.nan)
    kl = np.full(H, np.nan)
    probe_vals = {name: np.full(H, np.nan) for name in probes}
    n_showdowns = 0

    agent.reset(agent_rng)
    ctx = SessionContext(hands=[])
    for t in range(H):
        seat = int(seats[t])
        sigma = agent.strategy_for_hand(ctx, seat)
        # exact quantities of the hand-t strategy (before anything is sampled); the dense
        # conversion inside validates coverage of the seat's infosets, legality and row sums
        try:
            ev[t] = seat_ev(game, sigma, opp_profile, seat)
        except (KeyError, ValueError) as e:
            raise type(e)(f"agent {agent.name!r}, hand {t}, seat {seat}: {e}") from None
        expl[t] = seat_exploitability(game, sigma, seat, equilibrium_value_seat0)
        for name, by_seat in probe_by_seat.items():
            probe = by_seat.get(seat)
            if probe is not None:
                probe_vals[name][t] = float(sigma[probe[0]][probe[1]])

        # beliefs before the hand
        exact = None
        if posterior is not None:
            post_entropy[t] = float(posterior.entropy())
            exact = _probability_vector(posterior.probs, f"exact posterior, hand {t}")
        belief = agent.belief(ctx)
        if belief is not None:
            belief = check_belief(belief, expected_m, f"agent {agent.name!r}, hand {t}")
            if expected_m is None:
                expected_m = int(belief.shape[0])  # pinned by the first belief of the session
            agent_entropy[t] = entropy(belief)
            if exact is not None:
                kl[t] = kl_divergence(exact, belief)

        hand, _decisions = play_hand(
            game, seat, sigma, opp_profile, deal_rng, agent_rng=act_rng, opp_rng=opp_rng
        )
        realized[t] = hand.result
        n_showdowns += hand.opp_cards is not None
        if posterior is not None:
            posterior.update(hand)
        agent.observe(hand)
        ctx.hands.append(hand)

    return SessionMetrics(
        agent=agent.name,
        opp_id=int(opp_id),
        archetype=int(archetype),
        session=int(session),
        seats=seats,
        ev=ev,
        realized=realized,
        expl=expl,
        post_entropy=post_entropy,
        agent_entropy=agent_entropy,
        kl=kl,
        probes=probe_vals,
        n_showdowns=int(n_showdowns),
    )


def uniform_entropy(m: int) -> float:
    """Entropy of the uniform prior over ``m`` opponents (the value every curve starts from)."""
    return math.log(m) if m > 0 else 0.0
