"""Aggregate ``SessionMetrics`` over opponents x sessions.

``aggregate`` returns a plain (JSON-serialisable) dict with, per agent, the per-hand mean and
standard error of every metric, cumulative EV, cumulative regret versus a reference agent
(paired by ``(opp_id, session)`` so both faced the same deals), per-archetype breakdowns and
the E1 success criteria as booleans next to the numbers behind them.

Standard errors are across sessions at a fixed hand index: ``se = std(ddof=1) / sqrt(n)`` over
the non-NaN entries (NaN when fewer than two).

Deviation from docs/experiments/e1-kuhn.md: the success criteria are evaluated **pointwise on
per-hand means for every** ``t >= threshold`` (e.g. mean KL below 0.1 nats at every hand from 32
on), which is stricter than the spec's "by t = 32 on average"; hand thresholds are clipped to
``H - 1`` for short runs and flagged ``scaled_to_H``. Paired quantities (regret, H2 gaps) require
every agent to have run exactly the same ``(opp_id, session)`` set; a mismatch raises instead of
silently dropping sessions.

E2 (docs/experiments/e2-decision-relevant-inference.md): ``kl_policy`` / ``kl_policy_unweighted``
are aggregated like every other metric (NaN-aware: pure-strategy hands are NaN and drop out of
the mean, ``n`` says how many sessions contributed). ``check_e2_criteria`` evaluates the E2
criteria, either for one condition with the plain agent names (E2-1) or across conditions whose
transformer agents are tagged ``Transformer(sample)[B]`` (``tag_condition``), all pointwise on
per-hand means for ``t >= threshold`` with paired one-sided shortfalls, like the E1 criteria.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from exsolver.eval.session import SessionMetrics

METRICS: tuple[str, ...] = (
    "ev",
    "realized",
    "expl",
    "post_entropy",
    "agent_entropy",
    "kl",
    "kl_policy",
    "kl_policy_unweighted",
)
FIRST_HANDS = 8
LAST_HANDS = 16
KL_POLICY_HANDS: tuple[int, ...] = (16, 32, 63)  # hands the E2 tables report the policy KL at
_DOMINANCE_TOL = 1e-9


@dataclass(frozen=True)
class CriteriaNames:
    """Agent names the success criteria refer to (resolved leniently, see ``resolve_agent``)."""

    transformer: str = "Transformer(sample)"
    bayes: str = "BayesBR"
    equilibrium: str = "Equilibrium"
    oracle: str = "OracleBR"
    transformer_argmax: str = "Transformer(argmax)"
    thompson: str = "Thompson"
    plurality: str = "PluralityBR"


@dataclass(frozen=True)
class Thresholds:
    """Numbers from docs/experiments/e1-kuhn.md "Success criteria"."""

    h1_kl: float = 0.1  # KL(exact || model) below this ...
    h1_t: int = 32  # ... from this hand on
    h1_entropy_gap: float = 0.2  # |agent entropy - exact entropy| within this at every hand
    h2_gap: float = 0.02  # transformer EV within this of BayesBR ...
    h2_t_from: int = 16  # ... from this hand on
    h2_eq_t_from: int = 4  # transformer EV above Equilibrium from this hand on
    eq_expl: float = 1e-3  # exploitability of the Equilibrium agent is ~0


@dataclass(frozen=True)
class E2Thresholds:
    """Numbers from docs/experiments/e2-decision-relevant-inference.md "Success criteria"."""

    kl_policy: float = 0.05  # E2-1: policy KL of Transformer(sample) below this ...
    kl_policy_t_from: int = 16  # ... at every hand from this one on
    match_gap: float = 0.01  # E2-1: Transformer(sample) vs Thompson, (argmax) vs PluralityBR ...
    match_t_from: int = 16  # ... paired one-sided shortfall at most match_gap for t >= this
    bayes_gap: float = 0.02  # E2-2: condition E within this of BayesBR for t >= bayes_t_from
    bayes_t_from: int = 16
    ev_change: float = 0.01  # E2-3: |EV change| of C / D versus A below this for t >= ...
    ev_change_t_from: int = 16


@dataclass(frozen=True)
class E2Roles:
    """Which condition id plays which role in the E2 criteria (spec table "Conditions")."""

    baseline: str = "A"
    convergence: str = "B"
    no_identity: str = "C"
    forced_identity: str = "D"
    bayes_labels: str = "E"


def tag_condition(name: str, condition: str) -> str:
    """``"Transformer(sample)"`` + ``"B"`` -> ``"Transformer(sample)[B]"``."""
    return f"{name}[{condition}]"


def split_condition(name: str) -> tuple[str, str | None]:
    """Inverse of ``tag_condition``: ``("Transformer(sample)", "B")``; ``(name, None)`` untagged."""
    if name.endswith("]") and "[" in name:
        base, _, cond = name[:-1].rpartition("[")
        if base and cond:
            return base, cond
    return name, None


# ---------------------------------------------------------------------- statistics
def mean_se(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """nan-aware column mean, standard error (ddof=1) and count of a ``[n, H]`` array."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    valid = ~np.isnan(x)
    n = valid.sum(axis=0)
    total = np.where(valid, x, 0.0).sum(axis=0)
    mean = np.where(n > 0, total / np.maximum(n, 1), np.nan)
    dev = np.where(valid, x - mean, 0.0)
    var = np.where(n > 1, (dev * dev).sum(axis=0) / np.maximum(n - 1, 1), np.nan)
    se = np.sqrt(var / np.maximum(n, 1))
    return mean, se, n


def curve(x: np.ndarray) -> dict[str, list[float] | list[int]]:
    """``{"mean": [H], "se": [H], "n": [H]}`` of a ``[n, H]`` array."""
    m, s, n = mean_se(x)
    return {"mean": m.tolist(), "se": s.tolist(), "n": n.astype(int).tolist()}


def scalar(x: np.ndarray) -> dict[str, float | int]:
    """Mean and s.e. across sessions of a per-session scalar (1-D array)."""
    m, s, n = mean_se(np.asarray(x, dtype=np.float64).reshape(-1, 1))
    return {"mean": float(m[0]), "se": float(s[0]), "n": int(n[0])}


def _session_means(x: np.ndarray, t_slice: slice) -> np.ndarray:
    """Per-session mean over a slice of hands (nan-aware)."""
    sub = x[:, t_slice]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows (e.g. pure-strategy KL)
        return np.nanmean(sub, axis=1) if sub.shape[1] else np.full(x.shape[0], np.nan)


def pair_average(x: np.ndarray) -> np.ndarray:
    """Per-session average of consecutive hand pairs ``(2k, 2k+1)`` -> ``[n, H // 2]``.

    Seats alternate every hand, so a pair holds one hand at each seat: this removes the seat
    zig-zag from per-hand curves without pairing sessions.
    """
    n, H = x.shape
    k = H // 2
    if k == 0:
        return np.zeros((n, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x[:, : 2 * k].reshape(n, k, 2), axis=2)


def pair_x(H: int) -> list[float]:
    """x positions (hand index) of the pair averages: ``2k + 0.5``."""
    return [2 * k + 0.5 for k in range(H // 2)]


# ---------------------------------------------------------------------- grouping helpers
def _stack(group: Sequence[SessionMetrics], field: str) -> np.ndarray:
    return np.stack([getattr(m, field) for m in group]).astype(np.float64)


def _stack_probe(group: Sequence[SessionMetrics], name: str) -> np.ndarray:
    return np.stack([m.probes[name] for m in group]).astype(np.float64)


def _ordered_agents(metrics: Iterable[SessionMetrics]) -> list[str]:
    seen: dict[str, None] = {}
    for m in metrics:
        seen.setdefault(m.agent, None)
    return list(seen)


def group_by_agent(metrics: Iterable[SessionMetrics]) -> dict[str, list[SessionMetrics]]:
    """Sessions per agent name, agents in order of first appearance."""
    groups: dict[str, list[SessionMetrics]] = {}
    for m in metrics:
        groups.setdefault(m.agent, []).append(m)
    return groups


def _index(group: Sequence[SessionMetrics]) -> dict[tuple[int, int], SessionMetrics]:
    idx: dict[tuple[int, int], SessionMetrics] = {}
    for m in group:
        key = (m.opp_id, m.session)
        if key in idx:
            raise ValueError(f"agent {m.agent!r} has two sessions keyed {key}")
        idx[key] = m
    return idx


def paired_difference(
    a: Sequence[SessionMetrics], b: Sequence[SessionMetrics], field: str = "ev"
) -> np.ndarray:
    """``[n, H]`` array of ``a - b`` over sessions paired by ``(opp_id, session)``.

    Both groups must cover exactly the same ``(opp_id, session)`` keys; otherwise ``ValueError``
    (no session is ever dropped silently).
    """
    ia, ib = _index(a), _index(b)
    if set(ia) != set(ib):
        only_a, only_b = sorted(set(ia) - set(ib)), sorted(set(ib) - set(ia))
        raise ValueError(
            "sessions cannot be paired: (opp_id, session) keys differ "
            f"(only in first group: {only_a[:5]}{'...' if len(only_a) > 5 else ''}; "
            f"only in second group: {only_b[:5]}{'...' if len(only_b) > 5 else ''})"
        )
    if not ia:
        return np.zeros((0, a[0].H if a else 0))
    keys = sorted(ia)
    return np.stack([getattr(ia[k], field) - getattr(ib[k], field) for k in keys]).astype(
        np.float64
    )


def resolve_agent(names: Sequence[str], wanted: str) -> str | None:
    """Exact match, else case-insensitive, else every alphanumeric token of ``wanted`` in a name."""
    if wanted in names:
        return wanted
    low = {n.lower(): n for n in names}
    if wanted.lower() in low:
        return low[wanted.lower()]
    squashed = {"".join(_tokens(n)): n for n in names}  # "oracle_br" == "OracleBR"
    key = "".join(_tokens(wanted))
    if key in squashed:
        return squashed[key]
    tokens = [tok for tok in _tokens(wanted) if tok]
    hits = [n for n in names if all(tok in n.lower() for tok in tokens)]
    return hits[0] if len(hits) == 1 else None


def _tokens(s: str) -> list[str]:
    out, cur = [], []
    for ch in s.lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


# ---------------------------------------------------------------------- per-agent block
def _agent_block(
    group: Sequence[SessionMetrics],
    reference: Sequence[SessionMetrics] | None,
    archetype_names: Sequence[str] | None,
) -> dict[str, Any]:
    H = group[0].H
    ev = _stack(group, "ev")
    block: dict[str, Any] = {
        "n_sessions": len(group),
        "n_opponents": len({m.opp_id for m in group}),
        "metrics": {f: curve(_stack(group, f)) for f in METRICS},
        "pairs": {f: curve(pair_average(_stack(group, f))) for f in ("ev", "realized", "expl")},
        "probes": {p: curve(_stack_probe(group, p)) for p in group[0].probes},
        "ev_cumulative": curve(np.cumsum(ev, axis=1)),
        "ev_by_seat": {
            str(seat): scalar(_session_means(ev, slice(seat, None, 2))) for seat in (0, 1)
        },
        "showdown_rate": float(np.mean([m.n_showdowns / m.H for m in group])),
    }
    regret = None
    if reference is not None:
        diff = paired_difference(reference, group, "ev")  # ref - agent, per paired session
        if diff.shape[0]:
            regret = np.cumsum(diff, axis=1)
            block["regret_cumulative"] = {**curve(regret), "n_paired": int(diff.shape[0])}
    if regret is None:
        block["regret_cumulative"] = None
    block["summary"] = _summary(group, ev, regret, H)
    block["by_archetype"] = _by_archetype(group, reference, archetype_names)
    return block


def _summary(
    group: Sequence[SessionMetrics], ev: np.ndarray, regret: np.ndarray | None, H: int
) -> dict[str, Any]:
    expl = _stack(group, "expl")
    kl = _stack(group, "kl")
    pe = _stack(group, "post_entropy")
    ae = _stack(group, "agent_entropy")
    realized = _stack(group, "realized")
    klp = _stack(group, "kl_policy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN hands (pure strategies)
        klp_mean, _, _ = mean_se(klp)
    return {
        "ev_first8": scalar(_session_means(ev, slice(0, min(FIRST_HANDS, H)))),
        "ev_last16": scalar(_session_means(ev, slice(max(H - LAST_HANDS, 0), H))),
        "ev_all": scalar(_session_means(ev, slice(0, H))),
        "realized_all": scalar(_session_means(realized, slice(0, H))),
        "regret_final": None if regret is None else scalar(regret[:, -1]),
        "expl_mean": scalar(_session_means(expl, slice(0, H))),
        "kl_final": scalar(kl[:, -1]),
        "post_entropy_final": scalar(pe[:, -1]),
        "agent_entropy_final": scalar(ae[:, -1]),
        # E2: policy KL (NaN-aware; pure-strategy agents have no entries at all)
        "kl_policy_all": scalar(_session_means(klp, slice(0, H))),
        "kl_policy_final": scalar(klp[:, -1]),
        "kl_policy_at": {
            str(min(t, H - 1)): scalar(klp[:, min(t, H - 1)]) for t in KL_POLICY_HANDS
        },
        "kl_at": {str(min(t, H - 1)): scalar(kl[:, min(t, H - 1)]) for t in KL_POLICY_HANDS},
        "pure_hands_share": float(np.mean([m.n_pure_hands / m.H for m in group])),
    }


def _by_archetype(
    group: Sequence[SessionMetrics],
    reference: Sequence[SessionMetrics] | None,
    archetype_names: Sequence[str] | None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arch in sorted({m.archetype for m in group}):
        sub = [m for m in group if m.archetype == arch]
        name = _arch_name(arch, archetype_names)
        ev = _stack(sub, "ev")
        H = sub[0].H
        regret = None
        if reference is not None:
            ref_sub = [m for m in reference if m.archetype == arch]
            diff = paired_difference(ref_sub, sub, "ev")
            regret = np.cumsum(diff, axis=1) if diff.shape[0] else None
        out[name] = {
            "archetype_id": int(arch),
            "n_sessions": len(sub),
            "ev": curve(ev),
            "summary": _summary(sub, ev, regret, H),
        }
    return out


def _arch_name(arch: int, names: Sequence[str] | None) -> str:
    if names is not None and 0 <= arch < len(names):
        return str(names[arch])
    return str(arch)


# ---------------------------------------------------------------------- criterion primitives
def _clip_t(t: int, H: int) -> tuple[int, bool]:
    return (min(t, H - 1), t > H - 1)


def pointwise_below(
    group: Sequence[SessionMetrics], field: str, threshold: float, t_from: int, H: int
) -> dict[str, Any]:
    """Per-hand mean of ``field`` below ``threshold`` at every hand ``t >= t_from`` (clipped)."""
    t0, scaled = _clip_t(t_from, H)
    mean, se, n = mean_se(_stack(group, field))
    out: dict[str, Any] = {"pass": None, "threshold": threshold, "t": t0, "scaled_to_H": scaled}
    if not np.all(n[t0:] > 0):
        out["note"] = f"no {field} recorded at some hand >= {t0}"
        return out
    tail = mean[t0:]
    out.update(
        {
            "value_at_t": float(mean[t0]),
            "value_at_t_se": float(se[t0]),
            "max_from_t": float(np.max(tail)),
            "t_of_max": int(t0 + np.argmax(tail)),
            "n_sessions_at_t": int(n[t0]),
            "pass": bool(np.all(tail < threshold)),
        }
    )
    return out


def paired_shortfall(
    reference: Sequence[SessionMetrics],
    group: Sequence[SessionMetrics],
    field: str,
    threshold: float,
    t_from: int,
    H: int,
) -> dict[str, Any]:
    """One-sided: ``mean_sessions(reference - group)`` at most ``threshold`` for every ``t >= t_from``.

    Sessions are paired by ``(opp_id, session)``; being *better* than the reference never fails.
    """
    t0, scaled = _clip_t(t_from, H)
    out: dict[str, Any] = {
        "pass": None,
        "threshold": threshold,
        "t_from": t0,
        "scaled_to_H": scaled,
    }
    short = paired_difference(reference, group, field)
    if not short.shape[0]:
        out["note"] = "no sessions paired by (opp_id, session)"
        return out
    m, s, _ = mean_se(short)
    tail = m[t0:]
    out.update(
        {
            "n_paired": int(short.shape[0]),
            "max_shortfall": float(tail.max()),
            "t_of_max": int(t0 + tail.argmax()),
            "mean_shortfall": float(tail.mean()),
            "mean_shortfall_se": float(np.sqrt(np.mean(s[t0:] ** 2) / tail.size)),
            "pass": bool(np.all(tail <= threshold)),
        }
    )
    return out


def paired_abs_change(
    a: Sequence[SessionMetrics],
    b: Sequence[SessionMetrics],
    field: str,
    threshold: float,
    t_from: int,
    H: int,
) -> dict[str, Any]:
    """Two-sided: ``|mean_sessions(a - b)|`` below ``threshold`` at every ``t >= t_from``."""
    t0, scaled = _clip_t(t_from, H)
    out: dict[str, Any] = {
        "pass": None,
        "threshold": threshold,
        "t_from": t0,
        "scaled_to_H": scaled,
    }
    diff = paired_difference(a, b, field)
    if not diff.shape[0]:
        out["note"] = "no sessions paired by (opp_id, session)"
        return out
    m, s, _ = mean_se(diff)
    tail = m[t0:]
    out.update(
        {
            "n_paired": int(diff.shape[0]),
            "max_abs_change": float(np.abs(tail).max()),
            "t_of_max": int(t0 + np.abs(tail).argmax()),
            "mean_change": float(tail.mean()),
            "mean_change_se": float(np.sqrt(np.mean(s[t0:] ** 2) / tail.size)),
            "pass": bool(np.all(np.abs(tail) < threshold)),
        }
    )
    return out


def paired_lower_at_every_hand(
    lower: Sequence[SessionMetrics], higher: Sequence[SessionMetrics], field: str
) -> dict[str, Any]:
    """``mean_sessions(higher - lower) > 0`` at every hand (paired; NaN hands are skipped)."""
    out: dict[str, Any] = {"pass": None}
    diff = paired_difference(higher, lower, field)
    if not diff.shape[0]:
        out["note"] = "no sessions paired by (opp_id, session)"
        return out
    m, s, n = mean_se(diff)
    ok = n > 0
    if not ok.any():
        out["note"] = f"no {field} recorded"
        return out
    margin = m[ok]
    out.update(
        {
            "n_paired": int(diff.shape[0]),
            "n_hands_compared": int(ok.sum()),
            "min_margin": float(margin.min()),
            "t_of_min": int(np.flatnonzero(ok)[margin.argmin()]),
            "mean_margin": float(margin.mean()),
            "n_hands_failing": int(np.sum(margin <= 0.0)),
            "pass": bool(np.all(margin > 0.0)),
        }
    )
    return out


# ---------------------------------------------------------------------- success criteria
def check_criteria(
    groups: dict[str, list[SessionMetrics]],
    H: int,
    names: CriteriaNames,
    thresholds: Thresholds,
) -> dict[str, Any]:
    """The spec's H1 / H2 / sanity checks as ``{"pass": bool | None, ...numbers}`` entries.

    ``pass`` is ``None`` when an agent the check needs is absent. Hand thresholds are clipped
    to ``H - 1`` (flagged with ``"scaled_to_H"``) so short smoke runs still evaluate.
    """
    present = list(groups)
    tr = resolve_agent(present, names.transformer)
    bayes = resolve_agent(present, names.bayes)
    eq = resolve_agent(present, names.equilibrium)
    oracle = resolve_agent(present, names.oracle)
    resolved = {"transformer": tr, "bayes": bayes, "equilibrium": eq, "oracle": oracle}
    out: dict[str, Any] = {"resolved_agents": resolved, "thresholds": asdict(thresholds)}

    def clip_t(t: int) -> tuple[int, bool]:
        return _clip_t(t, H)

    # H1: KL(exact || model) < 0.1 nats from hand 32 on; entropy tracks within 0.2 nats.
    h1_kl: dict[str, Any] = {"pass": None, "agent": tr, "threshold": thresholds.h1_kl}
    h1_ent: dict[str, Any] = {"pass": None, "agent": tr, "threshold": thresholds.h1_entropy_gap}
    if tr is not None:
        below = pointwise_below(groups[tr], "kl", thresholds.h1_kl, thresholds.h1_t, H)
        if below.get("note"):
            h1_kl.update({"t": below["t"], "scaled_to_H": below["scaled_to_H"]})
            h1_kl["note"] = "no KL recorded (agent has no belief or no exact posterior given)"
        else:
            h1_kl.update(
                {
                    "t": below["t"],
                    "scaled_to_H": below["scaled_to_H"],
                    "kl_at_t": below["value_at_t"],
                    "kl_at_t_se": below["value_at_t_se"],
                    "kl_max_from_t": below["max_from_t"],
                    "pass": below["pass"],
                }
            )
        pe_mean, _, pe_n = mean_se(_stack(groups[tr], "post_entropy"))
        ae_mean, _, ae_n = mean_se(_stack(groups[tr], "agent_entropy"))
        ok = (pe_n > 0) & (ae_n > 0)
        if ok.all():
            gap = np.abs(ae_mean - pe_mean)
            h1_ent.update(
                {
                    "max_abs_gap": float(gap.max()),
                    "t_of_max_gap": int(gap.argmax()),
                    "mean_abs_gap": float(gap.mean()),
                    "pass": bool(gap.max() <= thresholds.h1_entropy_gap),
                }
            )
        else:
            h1_ent["note"] = "entropies not recorded for every hand"
    out["H1_kl"] = h1_kl
    out["H1_entropy_tracking"] = h1_ent

    # H2: transformer within 0.02 chips/hand of BayesBR for t >= 16, above Equilibrium for t >= 4.
    h2_b: dict[str, Any] = {
        "pass": None,
        "agent": tr,
        "versus": bayes,
        "threshold": thresholds.h2_gap,
    }
    if tr is not None and bayes is not None:
        short = paired_shortfall(
            groups[bayes], groups[tr], "ev", thresholds.h2_gap, thresholds.h2_t_from, H
        )  # bayes - transformer
        short.pop("t_of_max", None)
        h2_b.update(short)
    out["H2_within_bayes"] = h2_b

    h2_e: dict[str, Any] = {"pass": None, "agent": tr, "versus": eq}
    if tr is not None and eq is not None:
        t0, scaled = clip_t(thresholds.h2_eq_t_from)
        adv = paired_difference(groups[tr], groups[eq], "ev")  # transformer - equilibrium
        if adv.shape[0]:
            m, s, _ = mean_se(adv)
            tail = m[t0:]
            h2_e.update(
                {
                    "t_from": t0,
                    "scaled_to_H": scaled,
                    "n_paired": int(adv.shape[0]),
                    "min_advantage": float(tail.min()),
                    "t_of_min": int(t0 + tail.argmin()),
                    "mean_advantage": float(tail.mean()),
                    "n_hands_failing": int(np.sum(tail <= 0.0)),
                    "pass": bool(np.all(tail > 0.0)),
                }
            )
        else:
            h2_e["note"] = "no sessions paired by (opp_id, session)"
    out["H2_beats_equilibrium"] = h2_e

    # Sanity: the oracle dominates every agent at every hand (per paired session, exact).
    dom: dict[str, Any] = {"pass": None, "reference": oracle, "tolerance": _DOMINANCE_TOL}
    if oracle is not None:
        worst = 0.0
        worst_agent = None
        per_agent: dict[str, float] = {}
        for name, group in groups.items():
            if name == oracle:
                continue
            diff = paired_difference(groups[oracle], group, "ev")
            if not diff.shape[0]:
                continue
            violation = float(max(0.0, -diff.min()))
            per_agent[name] = violation
            if violation > worst:
                worst, worst_agent = violation, name
        dom.update(
            {
                "max_violation": worst,
                "worst_agent": worst_agent,
                "violation_by_agent": per_agent,
                "pass": bool(worst <= _DOMINANCE_TOL) if per_agent else None,
            }
        )
    out["sanity_oracle_dominates"] = dom

    # Sanity: the Equilibrium agent is (essentially) unexploitable at every hand.
    eq_x: dict[str, Any] = {"pass": None, "agent": eq, "threshold": thresholds.eq_expl}
    if eq is not None:
        expl = _stack(groups[eq], "expl")
        eq_x.update({"expl_mean": float(expl.mean()), "expl_max": float(expl.max())})
        eq_x["pass"] = bool(expl.max() < thresholds.eq_expl)
    out["sanity_equilibrium_exploitability"] = eq_x
    return out


# ---------------------------------------------------------------------- E2 criteria
def _e2_entry(description: str, **fields: Any) -> dict[str, Any]:
    return {"pass": None, "description": description, **fields}


def _e2_absent(
    entry: dict[str, Any], missing: list[str], conditions: dict[str, str] | None
) -> None:
    pending = [
        n
        for n in missing
        if conditions is not None and conditions.get(split_condition(n)[1] or "", "") == "pending"
    ]
    if pending:
        entry["note"] = f"condition pending (checkpoint not trained yet): {', '.join(pending)}"
    else:
        entry["note"] = f"agent(s) absent: {', '.join(missing)}"


def check_e2_criteria(
    groups: dict[str, list[SessionMetrics]],
    H: int,
    *,
    names: CriteriaNames | None = None,
    thresholds: E2Thresholds | None = None,
    conditions: dict[str, str] | None = None,
    roles: E2Roles | None = None,
) -> dict[str, Any]:
    """E2-1 / E2-2 / E2-3 as ``{"pass": bool | None, "description": str, ...numbers}`` entries.

    ``conditions`` is ``None`` for a single-condition aggregate with plain agent names (only the
    E2-1 checks apply), or ``{condition_id: "evaluated" | "pending"}`` for the combined
    aggregate whose transformer agents are tagged ``tag_condition(name, id)``; ``roles`` says
    which id is the baseline (E2-1), the convergence run (E2-3a), the ``lambda_opp`` variants
    (E2-3b) and the Bayes-BR-label run (E2-2). Checks whose agents are absent get ``pass: None``
    and a note saying whether the condition is pending. Hand thresholds are clipped to ``H - 1``
    and flagged ``scaled_to_H`` like the E1 criteria.
    """
    names = names or CriteriaNames()
    th = thresholds or E2Thresholds()
    roles = roles or E2Roles()
    present = list(groups)

    def res(base: str, cond: str | None) -> str | None:
        want = base if cond is None else tag_condition(base, cond)
        if want in present:
            return want
        return resolve_agent(present, want) if cond is None else None

    def tr(mode: str, cond: str | None) -> str | None:
        return res(names.transformer if mode == "sample" else names.transformer_argmax, cond)

    def want(mode: str, cond: str | None) -> str:
        base = names.transformer if mode == "sample" else names.transformer_argmax
        return base if cond is None else tag_condition(base, cond)

    thompson, plural, bayes = (res(n, None) for n in (names.thompson, names.plurality, names.bayes))
    base_cond = None if conditions is None else roles.baseline
    suffix = "" if base_cond is None else f"[{base_cond}]"
    out: dict[str, Any] = {
        "thresholds": asdict(th),
        "roles": None if conditions is None else asdict(roles),
        "conditions": conditions,
        "resolved_agents": {"thompson": thompson, "plurality": plural, "bayes": bayes},
    }

    # E2-1a: policy KL of Transformer(sample) below 0.05 nats from t = 16 on
    e = _e2_entry(
        f"policy KL of {want('sample', base_cond)} < {th.kl_policy} nats at every t >= {th.kl_policy_t_from}",
        agent=tr("sample", base_cond),
    )
    if e["agent"] is not None:
        e.update(
            pointwise_below(groups[e["agent"]], "kl_policy", th.kl_policy, th.kl_policy_t_from, H)
        )
    else:
        _e2_absent(e, [want("sample", base_cond)], conditions)
    out[f"E2-1a_kl_policy{suffix}"] = e

    # E2-1b / E2-1c: sample within 0.01 of Thompson, argmax within 0.01 of PluralityBR (paired)
    for key, mode, ref_name, ref in (
        ("E2-1b_sample_within_thompson", "sample", names.thompson, thompson),
        ("E2-1c_argmax_within_plurality", "argmax", names.plurality, plural),
    ):
        agent = tr(mode, base_cond)
        e = _e2_entry(
            f"{want(mode, base_cond)} EV within {th.match_gap} chips/hand of {ref_name} for t >= {th.match_t_from} (paired, one-sided)",
            agent=agent,
            versus=ref,
        )
        if agent is not None and ref is not None:
            e.update(
                paired_shortfall(groups[ref], groups[agent], "ev", th.match_gap, th.match_t_from, H)
            )
        else:
            _e2_absent(
                e,
                [n for n, r in ((want(mode, base_cond), agent), (ref_name, ref)) if r is None],
                conditions,
            )
        out[f"{key}{suffix}"] = e

    if conditions is None:
        return out

    # E2-2: condition E within 0.02 of BayesBR for t >= 16 (both modes reported)
    for mode in ("sample", "argmax"):
        agent = tr(mode, roles.bayes_labels)
        e = _e2_entry(
            f"{want(mode, roles.bayes_labels)} EV within {th.bayes_gap} chips/hand of {names.bayes} for t >= {th.bayes_t_from} (paired, one-sided)",
            agent=agent,
            versus=bayes,
        )
        if agent is not None and bayes is not None:
            e.update(
                paired_shortfall(
                    groups[bayes], groups[agent], "ev", th.bayes_gap, th.bayes_t_from, H
                )
            )
        else:
            _e2_absent(
                e,
                [
                    n
                    for n, r in ((want(mode, roles.bayes_labels), agent), (names.bayes, bayes))
                    if r is None
                ],
                conditions,
            )
        out[f"E2-2_{mode}_within_bayes[{roles.bayes_labels}]"] = e

    # E2-3a: condition B's identity KL below condition A's at every t (Transformer(sample))
    a, b = tr("sample", roles.baseline), tr("sample", roles.convergence)
    e = _e2_entry(
        f"identity KL of {want('sample', roles.convergence)} below {want('sample', roles.baseline)} at every hand (paired)",
        agent=b,
        versus=a,
    )
    if a is not None and b is not None:
        e.update(paired_lower_at_every_hand(groups[b], groups[a], "kl"))
    else:
        _e2_absent(
            e,
            [
                n
                for n, r in (
                    (want("sample", roles.convergence), b),
                    (want("sample", roles.baseline), a),
                )
                if r is None
            ],
            conditions,
        )
    out[f"E2-3a_identity_kl[{roles.convergence}]<[{roles.baseline}]"] = e

    # E2-3b: conditions C / D change the transformer's EV by less than 0.01 (both modes)
    for cond in (roles.no_identity, roles.forced_identity):
        e = _e2_entry(
            f"|EV change| of Transformer[{cond}] versus Transformer[{roles.baseline}] < {th.ev_change} chips/hand for t >= {th.ev_change_t_from}, both modes (paired)",
        )
        missing = []
        per_mode: dict[str, Any] = {}
        for mode in ("sample", "argmax"):
            x, y = tr(mode, cond), tr(mode, roles.baseline)
            if x is None:
                missing.append(want(mode, cond))
            if y is None:
                missing.append(want(mode, roles.baseline))
            if x is not None and y is not None:
                per_mode[mode] = paired_abs_change(
                    groups[x], groups[y], "ev", th.ev_change, th.ev_change_t_from, H
                )
        if missing:
            _e2_absent(e, sorted(set(missing)), conditions)
        else:
            e.update(
                {
                    "threshold": th.ev_change,
                    "t_from": per_mode["sample"]["t_from"],
                    "scaled_to_H": per_mode["sample"]["scaled_to_H"],
                    "max_abs_change_sample": per_mode["sample"].get("max_abs_change"),
                    "max_abs_change_argmax": per_mode["argmax"].get("max_abs_change"),
                    "mean_change_sample": per_mode["sample"].get("mean_change"),
                    "mean_change_argmax": per_mode["argmax"].get("mean_change"),
                    "pass": None
                    if any(v["pass"] is None for v in per_mode.values())
                    else bool(all(v["pass"] for v in per_mode.values())),
                }
            )
            if any(v.get("note") for v in per_mode.values()):
                e["note"] = "; ".join(v["note"] for v in per_mode.values() if v.get("note"))
        out[f"E2-3b_ev_change[{cond}]vs[{roles.baseline}]"] = e
    return out


# ---------------------------------------------------------------------- entry point
def aggregate(
    metrics: Sequence[SessionMetrics],
    *,
    reference: str = "OracleBR",
    archetype_names: Sequence[str] | None = None,
    names: CriteriaNames | None = None,
    thresholds: Thresholds | None = None,
) -> dict[str, Any]:
    """Aggregate over sessions; see the module docstring for the layout of the result."""
    if not metrics:
        raise ValueError("no sessions to aggregate")
    H = metrics[0].H
    if any(m.H != H for m in metrics):
        raise ValueError("all sessions must have the same number of hands")
    names = names or CriteriaNames()
    thresholds = thresholds or Thresholds()
    agents = _ordered_agents(metrics)
    groups = {name: [m for m in metrics if m.agent == name] for name in agents}
    ref_name = resolve_agent(agents, reference) if reference else None
    ref_group = groups[ref_name] if ref_name is not None else None

    per_agent = {
        name: _agent_block(group, ref_group, archetype_names) for name, group in groups.items()
    }
    opp_ids = sorted({m.opp_id for m in metrics})
    archetypes = sorted({m.archetype for m in metrics})
    return {
        "H": H,
        "pair_x": pair_x(H),
        "n_sessions": len(metrics),
        "n_sessions_per_agent": {name: len(group) for name, group in groups.items()},
        "n_opponents": len(opp_ids),
        "n_sessions_per_opponent": max(
            (
                len({m.session for m in metrics if m.opp_id == o and m.agent == agents[0]})
                for o in opp_ids
            ),
            default=0,
        ),
        "agents": agents,
        "reference": ref_name,
        "archetypes": {_arch_name(a, archetype_names): int(a) for a in archetypes},
        "probes": list(metrics[0].probes),
        "per_agent": per_agent,
        "criteria": check_criteria(groups, H, names, thresholds),
    }
