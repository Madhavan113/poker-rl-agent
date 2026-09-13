"""Summarise a sampled Kuhn population: equilibrium EV, oracle best-response EV, gap, exploitability.

    uv run python -m exsolver.population.summary --m 256 --seed 0 --out runs/e1/population_summary.md

All numbers are exact (game-tree traversal), in chips per hand for the agent averaged over both
seats. Note that the oracle best-response EV averaged over seats *is* the opponent's
exploitability (NashConv/2), so those two columns coincide by definition.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from exsolver.games.kuhn import KuhnPoker
from exsolver.population.kuhn_prior import KUHN_PARAM_NAMES, KuhnPrior
from exsolver.population.population import Population, sample_population
from exsolver.solvers.best_response import best_response_value, expected_value, exploitability
from exsolver.solvers.cfr import cfr_plus
from exsolver.strategy import TabularStrategy


@dataclass
class PopulationStats:
    """Per-opponent numbers, agent's point of view, averaged over the two seats."""

    eq_ev: np.ndarray  # EV of the CFR+ equilibrium versus the opponent
    br_ev: np.ndarray  # EV of the oracle best response versus the opponent
    gap: np.ndarray  # br_ev - eq_ev: value left on the table by the equilibrium
    expl: np.ndarray  # opponent's exploitability, NashConv/2


def evaluate_population(game: KuhnPoker, pop: Population, eq: TabularStrategy) -> PopulationStats:
    m = len(pop)
    eq_ev, br_ev, expl = np.empty(m), np.empty(m), np.empty(m)
    for j, opp in enumerate(pop.profiles):
        eq_ev[j] = 0.5 * (expected_value(game, eq, opp) - expected_value(game, opp, eq))
        br_ev[j] = 0.5 * (best_response_value(game, opp, 0) + best_response_value(game, opp, 1))
        expl[j] = exploitability(game, opp)
    return PopulationStats(eq_ev=eq_ev, br_ev=br_ev, gap=br_ev - eq_ev, expl=expl)


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _mean_sd(x: np.ndarray) -> str:
    if x.size == 0:
        return "-"
    return f"{x.mean():+.4f} ± {x.std():.4f}"


def archetype_table(pop: Population, stats: PopulationStats) -> str:
    """Mean ± sd of the per-opponent numbers for each archetype and overall."""
    rows = []
    groups = [(name, pop.archetypes == k) for k, name in enumerate(pop.archetype_names)]
    groups.append(("all", np.ones(len(pop), dtype=bool)))
    for name, mask in groups:
        rows.append(
            [
                name,
                str(int(mask.sum())),
                _mean_sd(stats.eq_ev[mask]),
                _mean_sd(stats.br_ev[mask]),
                _mean_sd(stats.gap[mask]),
                _mean_sd(stats.expl[mask]),
            ]
        )
    return _md_table(
        ["archetype", "n", "EV equilibrium", "EV oracle BR", "gap (BR - eq)", "exploitability"],
        rows,
    )


def theta_table(pop: Population) -> str:
    """Mean theta per archetype (columns in ``KUHN_PARAM_NAMES`` order)."""
    rows = []
    for k, name in enumerate(pop.archetype_names):
        mask = pop.archetypes == k
        if mask.any():
            rows.append([name] + [f"{v:.2f}" for v in pop.thetas[mask].mean(axis=0)])
        else:
            rows.append([name] + ["-"] * len(KUHN_PARAM_NAMES))
    return _md_table(["archetype"] + [f"`{n}`" for n in KUHN_PARAM_NAMES], rows)


def extremes_table(pop: Population, stats: PopulationStats, idx: np.ndarray) -> str:
    rows = []
    for rank, j in enumerate(idx, start=1):
        theta = " ".join(f"{v:.2f}" for v in pop.thetas[j])
        rows.append(
            [
                str(rank),
                str(int(j)),
                pop.archetype_names[int(pop.archetypes[j])],
                f"{stats.expl[j]:.4f}",
                f"{stats.eq_ev[j]:+.4f}",
                f"{stats.br_ev[j]:+.4f}",
                f"`{theta}`",
            ]
        )
    return _md_table(
        ["rank", "id", "archetype", "exploitability", "EV equilibrium", "EV oracle BR", "theta"],
        rows,
    )


def render_markdown(
    pop: Population,
    stats: PopulationStats,
    *,
    seed: int,
    iterations: int,
    eq_value: float,
    eq_expl: float,
    top: int,
) -> str:
    order = np.argsort(-stats.expl, kind="stable")
    parts = [
        "# Kuhn opponent population summary",
        "",
        f"M = {len(pop)} opponents sampled from `KuhnPrior` with seed {seed}; uniform weights. "
        f"Equilibrium: CFR+ with {iterations} iterations, seat-0 value {eq_value:+.6f} "
        f"(target -1/18 = {-1 / 18:+.6f}), exploitability {eq_expl:.2e}.",
        "",
        "All numbers are exact expected chips per hand for the agent, averaged over both seats. "
        "`gap` = EV of the oracle best response minus EV of the equilibrium; `exploitability` is "
        "NashConv/2 of the opponent, which coincides with the seat-averaged oracle BR EV.",
        "",
        "## Per-archetype statistics (mean ± sd)",
        "",
        archetype_table(pop, stats),
        "",
        "## Mean theta per archetype",
        "",
        "Columns follow `KUHN_PARAM_NAMES`: P(RAISE) at the seat-0 root, P(CALL) facing a bet after "
        "checking, P(RAISE) after the opponent checks, P(CALL) facing a bet; each for J, Q, K.",
        "",
        theta_table(pop),
        "",
        f"## {top} most exploitable opponents",
        "",
        extremes_table(pop, stats, order[:top]),
        "",
        f"## {top} least exploitable opponents",
        "",
        extremes_table(pop, stats, order[::-1][:top]),
        "",
    ]
    return "\n".join(parts)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--m", type=int, default=256, help="population size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs/e1/population_summary.md"))
    ap.add_argument("--iterations", type=int, default=2000, help="CFR+ iterations")
    ap.add_argument("--top", type=int, default=10, help="opponents listed at each extreme")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    game = KuhnPoker()
    rng = np.random.default_rng(args.seed)
    pop = sample_population(KuhnPrior(), args.m, rng)
    eq = cfr_plus(game, args.iterations)
    eq_value = expected_value(game, eq, eq)
    eq_expl = exploitability(game, eq)
    stats = evaluate_population(game, pop, eq)
    text = render_markdown(
        pop,
        stats,
        seed=args.seed,
        iterations=args.iterations,
        eq_value=eq_value,
        eq_expl=eq_expl,
        top=args.top,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(archetype_table(pop, stats))
    print()
    print(theta_table(pop))
    print(f"\nwrote {args.out} ({time.perf_counter() - t0:.2f}s)")


if __name__ == "__main__":
    main()
