"""Figures and the markdown summary for E1, drawn from an ``aggregate`` dict (or ``summary.json``).

Every figure is regenerable from ``summary.json`` alone, so re-plotting never needs a re-run.
Conventions (see the dataviz method): one y-axis per panel, thin 2px lines, ±1 s.e. bands as a
10% wash of the series hue, hairline solid grid, text in ink tokens (never the series colour),
a legend whenever two or more series are drawn. The learning agents take the four leading
categorical slots in a fixed order; the oracle, the equilibrium and the random agent are
reference / context series in ink grays so the comparison of interest stays in colour.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from exsolver.eval.session import BELIEF_FLOOR, BELIEF_FLOOR_CAP_NATS  # noqa: E402

# ---------------------------------------------------------------------- palette (reference instance)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
CATEGORICAL = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
BAND_ALPHA = 0.12
LINE_W = 1.6  # ~2px at the default 100 dpi
FIG_DPI = 150


@dataclass(frozen=True)
class Style:
    color: str
    linestyle: str = "-"
    linewidth: float = LINE_W
    band: bool = True
    zorder: int = 3


def _tokens(name: str) -> set[str]:
    out, cur = [], []
    for ch in name.lower():
        if ch.isalnum():
            cur.append(ch)
        elif cur:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return set(out)


def agent_styles(agents: Sequence[str]) -> dict[str, Style]:
    """Fixed colour per entity: known agents always get the same slot, whatever else is present."""
    styles: dict[str, Style] = {}
    used: set[int] = set()
    unknown: list[str] = []
    for name in agents:
        toks = _tokens(name)
        if "transformer" in toks and "argmax" in toks:
            slot = 1
        elif "transformer" in toks:
            slot = 0
        elif "bayes" in toks or "bayesbr" in toks:
            slot = 2
        elif "thompson" in toks:
            slot = 3
        elif "oracle" in toks or "oraclebr" in toks:
            styles[name] = Style(INK, "--", LINE_W, band=False, zorder=2)
            continue
        elif "equilibrium" in toks or "cfr" in toks or "nash" in toks:
            styles[name] = Style(INK_2, ":", LINE_W, band=True, zorder=2)
            continue
        elif "random" in toks or "uniform" in toks:
            styles[name] = Style(MUTED, "-", 1.0, band=False, zorder=1)
            continue
        else:
            unknown.append(name)
            continue
        styles[name] = Style(CATEGORICAL[slot])
        used.add(slot)
    free = [i for i in range(len(CATEGORICAL)) if i not in used]
    for name in unknown:
        if not free:
            raise ValueError("more than eight categorical series: fold agents or facet the figure")
        styles[name] = Style(CATEGORICAL[free.pop(0)])
    return styles


# ---------------------------------------------------------------------- drawing helpers
def _arr(values: Sequence[float | None] | np.ndarray) -> np.ndarray:
    """float64 array with ``None`` (strict-JSON null) read as NaN."""
    return np.array([np.nan if v is None else v for v in values], dtype=np.float64)


def _all_nan(values: Sequence[float | None] | np.ndarray) -> bool:
    return bool(np.isnan(_arr(values)).all())


def _setup_axes(ax: Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


def _plot_curve(
    ax: Axes,
    x: Sequence[float],
    mean: Sequence[float],
    se: Sequence[float] | None,
    style: Style,
    label: str,
) -> bool:
    """Line + s.e. band, skipping NaN points; returns whether anything was drawn."""
    x_arr = _arr(x)
    m = _arr(mean)
    ok = ~np.isnan(m)
    if not ok.any():
        return False
    ax.plot(
        x_arr[ok],
        m[ok],
        color=style.color,
        linestyle=style.linestyle,
        linewidth=style.linewidth,
        solid_joinstyle="round",
        solid_capstyle="round",
        label=label,
        zorder=style.zorder,
        marker="o" if ok.sum() == 1 else None,
        markersize=4,
    )
    if style.band and se is not None:
        s = _arr(se)
        s_ok = ok & ~np.isnan(s)
        if s_ok.any():
            ax.fill_between(
                x_arr[s_ok],
                (m - s)[s_ok],
                (m + s)[s_ok],
                color=style.color,
                alpha=BAND_ALPHA,
                linewidth=0,
                zorder=style.zorder - 1,
            )
    return True


def _hline(ax: Axes, y: float, text: str) -> None:
    ax.axhline(y, color=BASELINE, linewidth=0.8, linestyle="-", zorder=1)
    ax.annotate(
        text,
        xy=(1.0, y),
        xycoords=("axes fraction", "data"),
        xytext=(-2, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color=MUTED,
    )


def _figure(n_panels: int, width: float = 5.2, height: float = 3.6) -> tuple[Figure, list[Axes]]:
    fig, axes = plt.subplots(
        1, n_panels, figsize=(width * n_panels, height), squeeze=False, layout="constrained"
    )
    fig.patch.set_facecolor(SURFACE)
    return fig, list(axes[0])


def _finish(fig: Figure, axes: Iterable[Axes], suptitle: str, path: Path) -> Path:
    """Shared legend below the panels, two-line header, save and close (constrained layout)."""
    handles: dict[str, Any] = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            handles.setdefault(lab, h)
    if len(handles) >= 2:
        fig.legend(
            list(handles.values()),
            list(handles.keys()),
            loc="outside lower center",
            ncol=min(len(handles), 4),
            frameon=False,
            fontsize=8,
            labelcolor=INK_2,
        )
    fig.suptitle(suptitle, color=INK, fontsize=10.5, x=0.01, ha="left")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, facecolor=SURFACE)
    plt.close(fig)
    return path


def _header(agg: dict[str, Any], what: str) -> str:
    """Two-line figure title: what is plotted, then sessions / opponents / hands."""
    per_agent = agg.get("n_sessions_per_agent", {})
    n = max(per_agent.values()) if per_agent else agg.get("n_sessions", 0)
    return (
        f"E1 Kuhn \u2014 {what}\n{n} sessions per agent "
        f"({agg['n_opponents']} opponents \u00d7 {agg.get('n_sessions_per_opponent', '?')} sessions), "
        f"H = {agg['H']}"
    )


def _x(agg: dict[str, Any]) -> np.ndarray:
    return np.arange(agg["H"], dtype=np.float64)


# ---------------------------------------------------------------------- figures
def plot_ev_vs_hand(agg: dict[str, Any], path: str | Path) -> Path:
    """Exact EV per hand (left) and per seat-pair of hands (right), mean ± 1 s.e. across sessions."""
    styles = agent_styles(agg["agents"])
    fig, (ax_h, ax_p) = _figure(2)
    x = _x(agg)
    px = np.asarray(agg["pair_x"], dtype=np.float64)
    for name in agg["agents"]:
        block = agg["per_agent"][name]
        c = block["metrics"]["ev"]
        _plot_curve(ax_h, x, c["mean"], c["se"], styles[name], name)
        p = block["pairs"]["ev"]
        _plot_curve(ax_p, px, p["mean"], p["se"], styles[name], name)
    _setup_axes(ax_h, "EV per hand (exact; even t = seat 0)", "hand index t", "chips / hand")
    _setup_axes(ax_p, "EV per pair of hands (both seats)", "hand index t", "chips / hand")
    _hline(ax_p, 0.0, "break-even")
    return _finish(fig, (ax_h, ax_p), _header(agg, "expected value vs the true opponent"), path)


def plot_entropy_vs_hand(
    agg: dict[str, Any], path: str | Path, n_opponents_prior: int | None = None
) -> Path:
    """Exact posterior entropy under each agent's play, the belief read-out entropy, and the KL."""
    styles = agent_styles(agg["agents"])
    fig, (ax_e, ax_k) = _figure(2)
    x = _x(agg)
    drew_any_kl = False
    for name in agg["agents"]:
        block = agg["per_agent"][name]
        pe = block["metrics"]["post_entropy"]
        _plot_curve(ax_e, x, pe["mean"], pe["se"], styles[name], f"{name} — exact posterior")
        ae = block["metrics"]["agent_entropy"]
        if not _all_nan(ae["mean"]):
            st = styles[name]
            _plot_curve(
                ax_e,
                x,
                ae["mean"],
                ae["se"],
                Style(st.color, "--", st.linewidth, st.band, st.zorder),
                f"{name} — belief read-out",
            )
        kl = block["metrics"]["kl"]
        if not _all_nan(kl["mean"]):
            m = np.maximum(_arr(kl["mean"]), 1e-3)
            drew_any_kl |= _plot_curve(ax_k, x, m, kl["se"], styles[name], name)
    m_prior = n_opponents_prior or agg.get("n_population")
    if m_prior:
        _hline(ax_e, math.log(m_prior), f"prior: log M = {math.log(m_prior):.2f}")
    _setup_axes(ax_e, "Posterior entropy before hand t", "hand index t", "nats")
    ax_e.set_ylim(bottom=0)
    if drew_any_kl:
        ax_k.set_yscale("log")
        _hline(ax_k, 0.1, "H1 threshold 0.1 nats")
        _setup_axes(
            ax_k, "KL(exact ‖ belief) before hand t", "hand index t", "nats (log scale, floor 1e-3)"
        )
    else:
        _setup_axes(ax_k, "KL(exact ‖ belief): no agent reported a belief", "hand index t", "nats")
    return _finish(fig, (ax_e, ax_k), _header(agg, "opponent inference"), path)


def plot_exploitability_vs_hand(agg: dict[str, Any], path: str | Path) -> Path:
    """Seat exploitability of the hand-t strategy: per hand (left) and per seat-pair (right)."""
    styles = agent_styles(agg["agents"])
    fig, (ax_h, ax_p) = _figure(2)
    x = _x(agg)
    px = np.asarray(agg["pair_x"], dtype=np.float64)
    for name in agg["agents"]:
        block = agg["per_agent"][name]
        c = block["metrics"]["expl"]
        _plot_curve(ax_h, x, c["mean"], c["se"], styles[name], name)
        p = block["pairs"]["expl"]
        _plot_curve(ax_p, px, p["mean"], p["se"], styles[name], name)
    _setup_axes(
        ax_h,
        "Exploitability per hand (even t = seat 0)",
        "hand index t",
        "chips / hand (omniscient opponent's gain)",
    )
    _setup_axes(ax_p, "Exploitability per pair of hands", "hand index t", "chips / hand")
    for ax in (ax_h, ax_p):
        ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))
    return _finish(fig, (ax_h, ax_p), _header(agg, "price of commitment"), path)


def plot_regret_cumulative(agg: dict[str, Any], path: str | Path) -> Path:
    """Cumulative regret versus the reference (oracle) agent, paired by (opponent, session)."""
    styles = agent_styles(agg["agents"])
    fig, (ax,) = _figure(1, width=6.4)
    x = _x(agg)
    ref = agg.get("reference")
    for name in agg["agents"]:
        block = agg["per_agent"][name]
        reg = block.get("regret_cumulative")
        if reg is None or name == ref:
            continue
        _plot_curve(ax, x, reg["mean"], reg["se"], styles[name], name)
    _hline(ax, 0.0, f"{ref or 'reference'}")
    _setup_axes(
        ax, f"Cumulative regret vs {ref or 'reference'} (paired sessions)", "hand index t", "chips"
    )
    return _finish(fig, (ax,), _header(agg, "cumulative regret"), path)


def plot_probes(agg: dict[str, Any], path: str | Path) -> Path:
    """One panel per (probe, seat): the probed action probability over the hands at that seat."""
    styles = agent_styles(agg["agents"])
    probes: list[str] = list(agg.get("probes", []))
    if not probes:
        fig, (ax,) = _figure(1)
        _setup_axes(ax, "no probes recorded", "hand index t", "probability")
        return _finish(fig, (ax,), _header(agg, "probes"), path)
    fig, axes = plt.subplots(
        len(probes), 2, figsize=(10.4, 3.2 * len(probes)), squeeze=False, layout="constrained"
    )
    fig.patch.set_facecolor(SURFACE)
    x = _x(agg)
    flat: list[Axes] = []
    for r, probe in enumerate(probes):
        for seat in (0, 1):
            ax = axes[r][seat]
            flat.append(ax)
            sel = np.arange(seat, agg["H"], 2)
            for name in agg["agents"]:
                c = agg["per_agent"][name]["probes"].get(probe)
                if c is None:
                    continue
                m = _arr(c["mean"])[sel]
                s = _arr(c["se"])[sel]
                _plot_curve(ax, x[sel], m, s, styles[name], name)
            _setup_axes(ax, f"{probe} — agent at seat {seat}", "hand index t", "probability")
            ax.set_ylim(-0.02, 1.02)
    return _finish(
        fig,
        flat,
        _header(
            agg,
            "information-seeking probes (bluff = P(RAISE) with J, calldown = P(CALL) with Q facing a bet)",
        ),
        path,
    )


FIGURES = {
    "ev_vs_hand.png": plot_ev_vs_hand,
    "entropy_vs_hand.png": plot_entropy_vs_hand,
    "exploitability_vs_hand.png": plot_exploitability_vs_hand,
    "regret_cumulative.png": plot_regret_cumulative,
    "probes.png": plot_probes,
}


def write_all_plots(agg: dict[str, Any], out_dir: str | Path) -> list[Path]:
    """Write every figure of ``FIGURES`` into ``out_dir``."""
    out_dir = Path(out_dir)
    return [fn(agg, out_dir / name) for name, fn in FIGURES.items()]


# ---------------------------------------------------------------------- markdown summary
def _fmt(entry: dict[str, Any] | None, digits: int = 4) -> str:
    if (
        entry is None
        or entry.get("mean") is None
        or (isinstance(entry.get("mean"), float) and math.isnan(entry["mean"]))
    ):
        return "–"
    se = entry.get("se")
    se_txt = (
        "" if se is None or (isinstance(se, float) and math.isnan(se)) else f" ± {se:.{digits}f}"
    )
    return f"{entry['mean']:+.{digits}f}{se_txt}"


def _verdict(entry: dict[str, Any]) -> str:
    p = entry.get("pass")
    return "n/a" if p is None else ("PASS" if p else "FAIL")


def _num(v: Any, digits: int = 4) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.{digits}f}"
    return str(v)


def summary_table(agg: dict[str, Any]) -> str:
    header = [
        "agent",
        "sessions",
        f"EV first {min(8, agg['H'])} hands",
        f"EV last {min(16, agg['H'])} hands",
        "EV all hands",
        f"cum. regret vs {agg.get('reference') or 'ref'} (t=H)",
        "mean exploitability",
        "final KL (nats)",
    ]
    rows = []
    for name in agg["agents"]:
        s = agg["per_agent"][name]["summary"]
        rows.append(
            [
                name,
                str(agg["per_agent"][name]["n_sessions"]),
                _fmt(s["ev_first8"]),
                _fmt(s["ev_last16"]),
                _fmt(s["ev_all"]),
                _fmt(s["regret_final"], 3),
                _fmt(s["expl_mean"]),
                _fmt(s["kl_final"], 3),
            ]
        )
    return _md_table(header, rows)


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def criteria_table(agg: dict[str, Any]) -> str:
    crit = agg["criteria"]
    rows = []
    descriptions = {
        "H1_kl": "KL(exact ‖ transformer belief) < {threshold} nats from hand {t} on",
        "H1_entropy_tracking": "belief entropy within {threshold} nats of the exact posterior entropy at every hand",
        "H2_within_bayes": "transformer EV within {threshold} chips/hand of BayesBR for t ≥ {t_from} (paired)",
        "H2_beats_equilibrium": "transformer EV above Equilibrium for every t ≥ {t_from} (paired)",
        "sanity_oracle_dominates": "OracleBR ≥ every agent at every hand of every paired session",
        "sanity_equilibrium_exploitability": "Equilibrium agent exploitability < {threshold} at every hand",
        "sanity_equilibrium_vs_nash": "Equilibrium agent EV vs an exact Nash opponent is ∓1/18 by seat",
    }
    for key, entry in crit.items():
        if key in ("resolved_agents", "thresholds") or not isinstance(entry, dict):
            continue
        desc = descriptions.get(key, key)
        try:
            desc = desc.format(**entry)
        except (KeyError, IndexError):
            pass
        numbers = {
            k: v
            for k, v in entry.items()
            if k
            not in (
                "pass",
                "agent",
                "versus",
                "reference",
                "threshold",
                "t",
                "t_from",
                "note",
                "violation_by_agent",
            )
        }
        num_txt = ", ".join(f"{k}={_num(v)}" for k, v in numbers.items())
        if entry.get("note"):
            num_txt = (num_txt + "; " if num_txt else "") + str(entry["note"])
        rows.append([key, desc, _verdict(entry), num_txt])
    return _md_table(["check", "criterion", "verdict", "numbers"], rows)


def archetype_table(agg: dict[str, Any]) -> str:
    """Mean EV over all hands per agent × archetype."""
    arch_names = list(agg.get("archetypes", {}))
    if not arch_names:
        return ""
    header = ["agent"] + [f"EV vs {a}" for a in arch_names]
    rows = []
    for name in agg["agents"]:
        by = agg["per_agent"][name]["by_archetype"]
        rows.append(
            [name] + [_fmt(by[a]["summary"]["ev_all"]) if a in by else "–" for a in arch_names]
        )
    return _md_table(header, rows)


def training_table(agg: dict[str, Any]) -> str:
    """The training configuration read back from the checkpoint (``agg["training"]``)."""
    tr = agg.get("training") or {}
    if not tr:
        return ""
    rows = []
    for key, value in tr.items():
        if key == "spec_deviation":
            continue
        text = json.dumps(value, default=str) if isinstance(value, dict | list) else _num(value, 6)
        rows.append([key, text])
    return _md_table(["field", "value (from the checkpoint)"], rows)


def notes(agg: dict[str, Any]) -> list[str]:
    """Caveats every reader of the numbers needs."""
    out = [
        f'"final KL" is KL(exact ‖ belief) *before* the last hand, i.e. after H − 1 = {agg["H"] - 1} '
        "observed hands (the belief after the final hand is not recorded).",
        f"KL floors the agent's belief at {BELIEF_FLOOR:g} wherever the exact posterior has mass, so a "
        f"zero-mass miss contributes at most −log({BELIEF_FLOOR:g}) ≈ {BELIEF_FLOOR_CAP_NATS:.2f} nats per "
        "element and is never infinite.",
        "Criteria are evaluated pointwise on per-hand means for every t ≥ threshold (stricter than the "
        'spec\'s "by t = 32 on average"); hand thresholds are clipped to H − 1 for short runs '
        "(`scaled_to_H=True`).",
        "Regret pairs sessions by (opponent, session index): every agent faced the same deals.",
    ]
    tr = agg.get("training") or {}
    batch, steps = tr.get("batch_size"), tr.get("steps")
    spec_b, spec_s = tr.get("spec_batch_size", 64), tr.get("spec_steps", 20_000)
    if batch is not None and steps is not None and (batch, steps) != (spec_b, spec_s):
        reason = tr.get("deviation_reason") or tr.get("spec_deviation") or ""
        out.append(
            f"This run used batch {batch} \u00d7 {steps} steps (spec: {spec_b} \u00d7 {spec_s})"
            + (f"; {reason}" if reason else "")
            + "."
        )
    return out


def render_summary_md(agg: dict[str, Any], *, extra: dict[str, Any] | None = None) -> str:
    n_per = agg.get("n_sessions_per_agent", {})
    n = max(n_per.values()) if n_per else agg.get("n_sessions", 0)
    parts = [
        "# E1 Kuhn — summary",
        "",
        f"{n} sessions per agent = {agg['n_opponents']} opponents × "
        f"{agg.get('n_sessions_per_opponent', '?')} sessions, H = {agg['H']} hands per session, "
        f"agents: {', '.join(agg['agents'])}. All EV / exploitability numbers are exact "
        "(game-tree traversal of the agent's hand-t strategy); ± is one standard error across sessions. "
        f"Regret is cumulative EV shortfall versus `{agg.get('reference')}` on paired sessions (same opponent, same deals).",
        "",
        "## Agents",
        "",
        summary_table(agg),
        "",
        "## Success criteria (docs/experiments/e1-kuhn.md)",
        "",
        criteria_table(agg),
        "",
        f"Resolved agent names: {agg['criteria'].get('resolved_agents')}. Hand thresholds are clipped to H − 1 "
        "when H is shorter than the spec's 64 hands (`scaled_to_H=True`).",
        "",
        "## EV over all hands by opponent archetype",
        "",
        archetype_table(agg),
        "",
    ]
    if agg.get("training"):
        parts += ["## Training configuration (from the checkpoint)", "", training_table(agg), ""]
    parts += ["## Notes", ""] + [f"- {n}" for n in notes(agg)] + [""]
    parts += [
        "## Figures",
        "",
        "`ev_vs_hand.png`, `entropy_vs_hand.png`, `exploitability_vs_hand.png`, `regret_cumulative.png`, `probes.png` "
        "(all regenerable from `summary.json`).",
        "",
    ]
    if extra:
        parts += ["## Run", ""]
        parts += [f"- **{k}**: {v}" for k, v in extra.items()]
        parts.append("")
    return "\n".join(parts)


def write_summary_md(
    agg: dict[str, Any], path: str | Path, *, extra: dict[str, Any] | None = None
) -> Path:
    """Write ``summary.md`` (agent table, criteria verdicts, archetype breakdown)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_summary_md(agg, extra=extra))
    return path
