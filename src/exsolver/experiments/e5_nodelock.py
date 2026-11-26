"""E5: population node-locking against static and adaptive opponents (Kuhn or Leduc, exact).

Spec: ``docs/experiments/e5-population-nodelock.md``. One command runs everything and writes
``<out>/summary.json``, ``<out>/summary.md``, ``<out>/ev_by_block.png``, ``<out>/frontier.png``,
``<out>/blocks.npz`` and appends its command line and resolved config to ``<out>/README.md``:

    uv run python -m exsolver.experiments.e5_nodelock --game leduc --out runs/e5
    uv run python -m exsolver.experiments.e5_nodelock --game leduc --out runs/e5-smoke --smoke

Protocol per (opponent set, eta, inference, gamma, method), batched over the opponents: the
session is ``n_blocks`` blocks of ``block`` hands with alternating seats. Block 0 is played with
the equilibrium; before every later block the agent re-estimates the opponent from all hands so
far (hand weights ``gamma ** age_in_blocks``), builds its strategy with the method, and plays
the block; after it the opponent adapts by ``eta`` (``exsolver.population.adaptive``). The
reported EV of a block is exact (our strategy against the opponent's current profile, averaged
over the two seats); only the observations are sampled. Deals and action draws come from
per-(opponent, block, hand) seeds, so every config faces the same cards.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from exsolver.bayes.online import (
    discounted_posterior,
    em_fit,
    expected_visits,
    pad_replays,
)
from exsolver.bayes.posterior import log_likelihoods_from_replay, replay_hand, safe_log
from exsolver.games.base import Game
from exsolver.games.kuhn import KuhnPoker
from exsolver.games.leduc import LeducPoker
from exsolver.games.tree import compile_tree
from exsolver.play import play_hand
from exsolver.population.adaptive import adapt
from exsolver.population.kuhn_prior import KuhnPrior
from exsolver.population.leduc_prior import LeducPrior
from exsolver.population.population import sample_population
from exsolver.solvers.cfr import CFRPlus
from exsolver.solvers.constrained import LockedCFR, duality_gap, ev_dense, restricted_value
from exsolver.solvers.mixture import StrategyMixer

METHODS = ("nash", "br", "hard", "soft", "rnr_lo", "rnr_hi")


@dataclass
class E5Config:
    game: str = "leduc"
    out: Path = Path("runs/e5")
    m: int = 256  # population size (the Bayes prior's support)
    n_opp: int = 64  # opponents per set
    sets: tuple[str, ...] = ("in-pop", "fresh")
    block: int = 32
    n_blocks: int = 16
    etas: tuple[float, ...] = (0.0, 0.1, 0.3)
    inferences: tuple[str, ...] = ("bayes", "em")
    gammas: tuple[float, ...] = (1.0, 0.8)
    methods: tuple[str, ...] = METHODS
    hard_m: float = 10.0
    soft_k: float = 10.0
    rnr_lo: float = 0.5
    rnr_hi: float = 0.8
    em_strength: float = 4.0
    em_iterations: int = 30
    ref_smooth: float = 0.05
    cfr_iterations: int = 300  # CFR+ iterations per (cold) re-solve
    eq_iterations: int = 2000
    seed: int = 0
    workers: int = 8
    smoke: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def apply_smoke(cfg: E5Config) -> E5Config:
    cfg.m, cfg.n_opp, cfg.block, cfg.n_blocks = 16, 4, 8, 3
    cfg.etas, cfg.gammas = (0.0, 0.3), (1.0,)
    cfg.cfr_iterations, cfg.eq_iterations = 50, 300
    cfg.workers = 1
    return cfg


# --------------------------------------------------------------------- world construction
def make_game(name: str) -> Game:
    if name == "kuhn":
        return KuhnPoker()
    if name == "leduc":
        return LeducPoker()
    raise ValueError(f"unknown game {name!r}")


@dataclass
class World:
    """Everything a worker needs, built deterministically from the config."""

    game: Game
    equilibrium: np.ndarray  # [I, 3]
    reference: np.ndarray  # smoothed equilibrium, the EM prior centre
    population: np.ndarray  # [M, I, 3]
    pop_archetypes: np.ndarray
    archetype_names: list[str]
    opponents: dict[str, np.ndarray]  # set -> [B, I, 3]
    opp_archetypes: dict[str, np.ndarray]


def _sample(cfg: E5Config, game: Game, m: int, rng: np.random.Generator):
    tree = compile_tree(game)
    if cfg.game == "kuhn":
        pop = sample_population(KuhnPrior(), m, rng)
        dense = np.stack([tree.dense_from_strategy(p, (0, 1)) for p in pop.profiles])
        return dense, pop.archetypes, pop.archetype_names
    prior = _leduc_prior(cfg.eq_iterations)
    dense, ids = prior.sample_population(m, rng)
    return dense, ids, prior.names


@lru_cache(maxsize=2)
def _leduc_prior(eq_iterations: int) -> LeducPrior:
    return LeducPrior(cfr_iterations=eq_iterations)


def build_world(cfg: E5Config) -> World:
    game = make_game(cfg.game)
    tree = compile_tree(game)
    eq = CFRPlus(game).run(cfg.eq_iterations).average_dense()
    legal = tree.infoset_legal.astype(np.float64)
    ref = (1.0 - cfg.ref_smooth) * eq + cfg.ref_smooth * legal / legal.sum(1, keepdims=True)
    pop, pop_ids, names = _sample(cfg, game, cfg.m, np.random.default_rng([cfg.seed, 1]))
    # in-pop: an evenly spaced subset of the population; fresh: new draws from the same prior
    idx = np.linspace(0, cfg.m - 1, cfg.n_opp).round().astype(int)
    fresh, fresh_ids, _ = _sample(cfg, game, cfg.n_opp, np.random.default_rng([cfg.seed, 2]))
    return World(
        game,
        eq,
        ref,
        pop,
        pop_ids,
        list(names),
        {"in-pop": pop[idx], "fresh": fresh},
        {"in-pop": pop_ids[idx], "fresh": fresh_ids},
    )


# --------------------------------------------------------------------- one session batch
@dataclass(frozen=True)
class Job:
    set_name: str
    eta: float
    inference: str
    gamma: float
    method: str

    @property
    def key(self) -> str:
        return f"{self.set_name}|eta={self.eta}|{self.inference}|gamma={self.gamma}|{self.method}"


def jobs_for(cfg: E5Config) -> list[Job]:
    out = []
    for s in cfg.sets:
        for eta in cfg.etas:
            for method in cfg.methods:
                if method == "nash":  # never looks at the data
                    out.append(Job(s, eta, "-", 1.0, method))
                    continue
                for inf in cfg.inferences:
                    for g in cfg.gammas:
                        out.append(Job(s, eta, inf, g, method))
    return out


class _Estimator:
    """Per-opponent data store and estimate for one inference method."""

    def __init__(self, world: World, cfg: E5Config, kind: str, batch: int) -> None:
        self.world, self.cfg, self.kind = world, cfg, kind
        self.tree = compile_tree(world.game)
        self.replays: list[list] = [[] for _ in range(batch)]
        self.blocks: list[list[int]] = [[] for _ in range(batch)]
        self.prev: list[np.ndarray | None] = [None] * batch
        if kind == "bayes":
            self.log_pop = safe_log(world.population)
            self.log_prior = np.full(
                world.population.shape[0], -math.log(world.population.shape[0])
            )
            self.ll: list[list[np.ndarray]] = [[] for _ in range(batch)]
            profiles = [self.tree.strategy_from_dense(p) for p in world.population]
            self.mixers = [StrategyMixer(world.game, profiles, s) for s in (0, 1)]

    def add(self, b: int, record: Any, block: int) -> None:
        rep = replay_hand(self.world.game, record)
        self.replays[b].append(rep)
        self.blocks[b].append(block)
        if self.kind == "bayes":
            self.ll[b].append(log_likelihoods_from_replay(self.log_pop, rep))

    def estimate(self, b: int, now: int, gamma: float) -> tuple[np.ndarray, np.ndarray]:
        """(profile [I, 3], expected visits [I]) from opponent ``b``'s hands before block ``now``."""
        age = now - 1 - np.asarray(self.blocks[b])
        w = gamma**age
        hands = pad_replays(self.replays[b])
        if self.kind == "bayes":
            post = discounted_posterior(self.log_prior, np.stack(self.ll[b]), w)
            est = np.zeros((self.tree.n_infosets, 3))
            for s in (0, 1):
                est[self.tree.infoset_rows[s]] = self.mixers[s].mix_dense(post)
        else:
            est = em_fit(
                hands,
                self.world.reference,
                self.cfg.em_strength,
                w,
                init=self.prev[b],
                iterations=self.cfg.em_iterations,
            )
            self.prev[b] = est
        return est, expected_visits(est, hands, w)


def _our_strategy(
    job: Job,
    cfg: E5Config,
    solvers: dict[int, LockedCFR],
    lock: np.ndarray,
    visits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Our profile ``[B, I, 3]`` (both seats) and each solve's certified gap, max over seats."""
    game = solvers[0].game
    tree = compile_tree(game)
    out = np.zeros_like(lock)
    gaps = np.zeros(lock.shape[0])
    for seat in (0, 1):
        rows = tree.infoset_rows[seat]
        if job.method == "br":
            out[:, rows] = _exact_br(game, seat, lock)[:, rows]
            continue  # exact: gap 0
        kw: dict[str, Any]
        if job.method == "hard":
            kw = {"conf": (visits >= cfg.hard_m).astype(float)}
        elif job.method == "soft":
            kw = {"conf": visits / (visits + cfg.soft_k)}
        elif job.method in ("rnr_lo", "rnr_hi"):
            kw = {"rnr_p": cfg.rnr_lo if job.method == "rnr_lo" else cfg.rnr_hi}
        else:
            raise ValueError(job.method)
        solver = solvers[seat]
        solver.reset()  # cold re-solve (see LockedCFR on warm starts)
        sol = solver.solve(lock, cfg.cfr_iterations, **kw)
        out[:, rows] = sol[:, rows]
        for b in range(lock.shape[0]):
            one = {k: (v[b] if isinstance(v, np.ndarray) else v) for k, v in kw.items()}
            g = duality_gap(game, seat, sol[b], solver.adversary[b], lock[b], **one)
            gaps[b] = max(gaps[b], g)
    return out, gaps


def _exact_br(game: Game, seat: int, opp: np.ndarray) -> np.ndarray:
    tree = compile_tree(game)
    out = np.zeros_like(opp)
    for b in range(opp.shape[0]):
        rho = tree.realization(opp[b])
        cf = tree.reach_chance * tree.reach(rho, 1 - seat)
        _, a_star = tree.best_response_pass(opp[b], seat, cf)
        rows = tree.infoset_rows[seat]
        out[b, rows, a_star[rows]] = 1.0
    return out


def run_job(job: Job, cfg: E5Config, world: World) -> dict[str, Any]:
    game = world.game
    tree = compile_tree(game)
    theta = world.opponents[job.set_name].copy()
    bsz = theta.shape[0]
    eq = np.broadcast_to(world.equilibrium, theta.shape).copy()
    gv0 = float(tree.values(world.equilibrium)[tree.root])
    est = None if job.method == "nash" else _Estimator(world, cfg, job.inference, bsz)
    solvers = {s: LockedCFR(game, s, bsz) for s in (0, 1)}
    ev = np.zeros((bsz, cfg.n_blocks))
    expl = np.zeros((bsz, cfg.n_blocks))
    gap = np.zeros((bsz, cfg.n_blocks))
    for k in range(cfg.n_blocks):
        if k == 0 or est is None:
            ours = eq
        else:
            pairs = [est.estimate(b, k, job.gamma) for b in range(bsz)]
            lock = np.stack([p[0] for p in pairs])
            visits = np.stack([p[1] for p in pairs])
            ours, gap[:, k] = _our_strategy(job, cfg, solvers, lock, visits)
        for b in range(bsz):
            ev[b, k] = 0.5 * sum(ev_dense(game, s, ours[b], theta[b]) for s in (0, 1))
            worst = [
                restricted_value(game, s, ours[b], theta[b], np.zeros(tree.n_infosets))
                for s in (0, 1)
            ]
            expl[b, k] = 0.5 * ((gv0 - worst[0]) + (-gv0 - worst[1]))
        if est is not None:
            for b in range(bsz):
                ours_tab = tree.strategy_from_dense(ours[b])
                opp_tab = tree.strategy_from_dense(theta[b])
                for h in range(cfg.block):
                    rng = np.random.default_rng([cfg.seed, b, k, h])
                    rec, _ = play_hand(
                        game,
                        h % 2,
                        ours_tab,
                        opp_tab,
                        rng,
                        agent_rng=np.random.default_rng([cfg.seed, b, k, h, 1]),
                        opp_rng=np.random.default_rng([cfg.seed, b, k, h, 2]),
                    )
                    est.add(b, rec, k)
        theta = adapt(game, theta, ours, job.eta)
    return {"key": job.key, "job": asdict(job), "ev": ev, "expl": expl, "gap": gap}


# --------------------------------------------------------------------- parallel driver
_WORLD: World | None = None
_CFG: E5Config | None = None


def _init_worker(cfg: E5Config) -> None:
    global _WORLD, _CFG
    _CFG = cfg
    _WORLD = build_world(cfg)


def _run_in_worker(job: Job) -> dict[str, Any]:
    assert _WORLD is not None and _CFG is not None
    return run_job(job, _CFG, _WORLD)


def run_all(cfg: E5Config, world: World, log=print) -> list[dict[str, Any]]:
    jobs = jobs_for(cfg)
    results: list[dict[str, Any]] = []
    t0 = time.time()
    if cfg.workers <= 1:
        for j in jobs:
            results.append(run_job(j, cfg, world))
            log(f"[{time.time() - t0:7.0f}s] {len(results)}/{len(jobs)} {j.key}")
        return results
    with ProcessPoolExecutor(cfg.workers, initializer=_init_worker, initargs=(cfg,)) as ex:
        for r in ex.map(_run_in_worker, jobs):
            results.append(r)
            log(f"[{time.time() - t0:7.0f}s] {len(results)}/{len(jobs)} {r['key']}")
    return results


# --------------------------------------------------------------------- aggregation
def _mse(x: np.ndarray) -> tuple[float, float]:
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else 0.0


def aggregate(cfg: E5Config, world: World, results: list[dict[str, Any]]) -> dict[str, Any]:
    tree = compile_tree(world.game)
    by_key = {r["key"]: r for r in results}
    rows = []
    for r in results:
        j = r["job"]
        nash = by_key[Job(j["set_name"], j["eta"], "-", 1.0, "nash").key]["ev"]
        ev, ex = r["ev"], r["expl"]
        late = slice(cfg.n_blocks // 2, None)
        row = {
            **j,
            "ev": _mse(ev.mean(1)),
            "ev_late": _mse(ev[:, late].mean(1)),
            "gain_vs_nash": _mse((ev - nash).mean(1)),
            "gain_vs_nash_late": _mse((ev - nash)[:, late].mean(1)),
            "exploitability": _mse(ex.mean(1)),
            "solver_gap_mean": float(r["gap"][:, 1:].mean()) if cfg.n_blocks > 1 else 0.0,
            "solver_gap_max": float(r["gap"].max()),
            "ev_by_block": ev.mean(0).tolist(),
            "ev_by_block_se": (ev.std(0, ddof=1) / math.sqrt(ev.shape[0])).tolist()
            if ev.shape[0] > 1
            else [0.0] * cfg.n_blocks,
            "by_archetype": {
                name: float((ev - nash)[world.opp_archetypes[j["set_name"]] == a].mean())
                for a, name in enumerate(world.archetype_names)
                if np.any(world.opp_archetypes[j["set_name"]] == a)
            },
        }
        rows.append(row)
    oracle = {}
    for s, opp in world.opponents.items():
        vals = []
        for b in range(opp.shape[0]):
            br = [
                _exact_br(world.game, seat, opp[b : b + 1])[0] for seat in (0, 1)
            ]  # static oracle: BR to the base profile
            vals.append(0.5 * sum(ev_dense(world.game, seat, br[seat], opp[b]) for seat in (0, 1)))
        oracle[s] = _mse(np.array(vals))
    return {
        "config": _jsonable(asdict(cfg)),
        "game": cfg.game,
        "n_infosets": tree.n_infosets,
        "archetype_names": world.archetype_names,
        "static_oracle_ev": oracle,
        "rows": rows,
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _fmt(ms: Sequence[float], digits: int = 3) -> str:
    return f"{ms[0]:+.{digits}f} ± {ms[1]:.{digits}f}"


def summary_markdown(agg: dict[str, Any]) -> str:
    cfg = agg["config"]
    lines = [
        f"# E5 summary — {agg['game']} ({agg['n_infosets']} infosets)",
        "",
        f"{cfg['n_opp']} opponents per set, {cfg['n_blocks']} blocks × {cfg['block']} hands, "
        f"seed {cfg['seed']}. EV in chips/hand, exact; ± is one s.e. over opponents. "
        "Gain = paired EV minus the equilibrium agent's EV against the same opponents.",
        "",
    ]
    for s, o in agg["static_oracle_ev"].items():
        lines.append(f"- static oracle BR EV ({s}, η = 0): {_fmt(o)}")
    lines.append("")
    for s in cfg["sets"]:
        for eta in cfg["etas"]:
            lines += [
                f"## {s}, η = {eta}",
                "",
                "| inference | γ | method | EV | gain vs Nash | gain, 2nd half | exploitability "
                "| solver gap (max) |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for r in agg["rows"]:
                if r["set_name"] != s or r["eta"] != eta:
                    continue
                lines.append(
                    f"| {r['inference']} | {r['gamma']} | {r['method']} | {_fmt(r['ev'])} | "
                    f"{_fmt(r['gain_vs_nash'])} | {_fmt(r['gain_vs_nash_late'])} | "
                    f"{r['exploitability'][0]:.3f} | {r['solver_gap_max']:.1e} |"
                )
            lines.append("")
    return "\n".join(lines)


def plot(agg: dict[str, Any], out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from exsolver.eval.plots import BAND_ALPHA, CATEGORICAL, FIG_DPI, GRID, INK, INK_2, LINE_W

    cfg = agg["config"]
    colours = {m: CATEGORICAL[i] for i, m in enumerate([m for m in METHODS if m != "nash"])}
    paths = []
    for s in cfg["sets"]:
        for inf in cfg["inferences"]:
            fig, axes = plt.subplots(
                len(cfg["gammas"]),
                len(cfg["etas"]),
                figsize=(4.2 * len(cfg["etas"]), 3.2 * len(cfg["gammas"])),
                sharey=True,
                squeeze=False,
            )
            for gi, g in enumerate(cfg["gammas"]):
                for ei, eta in enumerate(cfg["etas"]):
                    ax = axes[gi][ei]
                    for r in agg["rows"]:
                        if r["set_name"] != s or r["eta"] != eta:
                            continue
                        is_nash = r["method"] == "nash"
                        if not is_nash and (r["inference"] != inf or r["gamma"] != g):
                            continue
                        y, se = np.array(r["ev_by_block"]), np.array(r["ev_by_block_se"])
                        x = np.arange(y.size)
                        c = INK if is_nash else colours[r["method"]]
                        ax.plot(
                            x, y, color=c, lw=LINE_W, ls="--" if is_nash else "-", label=r["method"]
                        )
                        if not is_nash:
                            ax.fill_between(x, y - se, y + se, color=c, alpha=BAND_ALPHA, lw=0)
                    ax.set_title(f"η = {eta}, γ = {g}", color=INK, fontsize=10)
                    ax.grid(color=GRID, lw=0.6)
                    ax.tick_params(colors=INK_2, labelsize=8)
                    for sp in ax.spines.values():
                        sp.set_color(GRID)
                    if ei == 0:
                        ax.set_ylabel("EV per hand (chips)", color=INK_2, fontsize=9)
                    if gi == len(cfg["gammas"]) - 1:
                        ax.set_xlabel(f"block ({cfg['block']} hands)", color=INK_2, fontsize=9)
            axes[0][-1].legend(fontsize=8, frameon=False, labelcolor=INK_2)
            fig.suptitle(
                f"E5 {agg['game']}: EV by block, {s} opponents, {inf} inference "
                f"(±1 s.e., {cfg['n_opp']} opponents)",
                color=INK,
                fontsize=11,
            )
            fig.tight_layout()
            p = out / f"ev_by_block_{s}_{inf}.png"
            fig.savefig(p, dpi=FIG_DPI)
            plt.close(fig)
            paths.append(p)
    # frontier: exploitability vs gain, one panel per eta, in-pop, gamma = 1
    s, g = cfg["sets"][0], cfg["gammas"][0]
    fig, axes = plt.subplots(
        1, len(cfg["etas"]), figsize=(4.2 * len(cfg["etas"]), 3.4), squeeze=False
    )
    markers = {"bayes": "o", "em": "s"}
    for ei, eta in enumerate(cfg["etas"]):
        ax = axes[0][ei]
        for r in agg["rows"]:
            if r["set_name"] != s or r["eta"] != eta or r["method"] == "nash" or r["gamma"] != g:
                continue
            ax.scatter(
                r["exploitability"][0],
                r["gain_vs_nash"][0],
                s=40,
                color=colours[r["method"]],
                marker=markers.get(r["inference"], "o"),
                edgecolor="white",
                lw=1.0,
                label=f"{r['method']} ({r['inference']})",
                zorder=3,
            )
        ax.axhline(0.0, color=INK, lw=0.8, ls="--")
        ax.set_title(f"η = {eta}", color=INK, fontsize=10)
        ax.set_xlabel("exploitability (chips/hand)", color=INK_2, fontsize=9)
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(colors=INK_2, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRID)
    axes[0][0].set_ylabel("EV gain over Nash (chips/hand)", color=INK_2, fontsize=9)
    axes[0][-1].legend(fontsize=7, frameon=False, labelcolor=INK_2, loc="best")
    fig.suptitle(f"E5 {agg['game']}: gain vs exploitability ({s}, γ = {g})", color=INK, fontsize=11)
    fig.tight_layout()
    p = out / "frontier.png"
    fig.savefig(p, dpi=FIG_DPI)
    plt.close(fig)
    paths.append(p)
    return paths


# --------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--game", choices=("kuhn", "leduc"), default="leduc")
    p.add_argument("--out", type=Path, default=Path("runs/e5"))
    p.add_argument("--m", type=int)
    p.add_argument("--n-opp", type=int)
    p.add_argument("--block", type=int)
    p.add_argument("--n-blocks", type=int)
    p.add_argument("--etas", type=float, nargs="+")
    p.add_argument("--gammas", type=float, nargs="+")
    p.add_argument("--inferences", nargs="+", choices=("bayes", "em"))
    p.add_argument("--sets", nargs="+", choices=("in-pop", "fresh"))
    p.add_argument("--methods", nargs="+", choices=METHODS)
    p.add_argument("--seed", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--smoke", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> E5Config:
    cfg = E5Config(game=args.game, out=args.out)
    if args.smoke:
        cfg = apply_smoke(cfg)
        cfg.smoke = True
    for name in ("m", "n_opp", "block", "n_blocks", "seed", "workers"):
        v = getattr(args, name)
        if v is not None:
            setattr(cfg, name, v)
    for name in ("etas", "gammas", "inferences", "sets", "methods"):
        v = getattr(args, name)
        if v is not None:
            setattr(cfg, name, tuple(v))
    if "nash" not in cfg.methods:
        cfg.methods = ("nash", *cfg.methods)  # every gain is paired against it
    return cfg


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    cfg.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    world = build_world(cfg)
    results = run_all(cfg, world, log=lambda m: print(m, file=sys.stderr, flush=True))
    agg = aggregate(cfg, world, results)
    agg["wall_seconds"] = time.time() - t0
    (cfg.out / "summary.json").write_text(json.dumps(agg, indent=1))
    (cfg.out / "summary.md").write_text(summary_markdown(agg))
    np.savez_compressed(
        cfg.out / "blocks.npz",
        keys=np.array([r["key"] for r in results]),
        ev=np.stack([r["ev"] for r in results]),
        expl=np.stack([r["expl"] for r in results]),
        gap=np.stack([r["gap"] for r in results]),
    )
    figures = plot(agg, cfg.out)
    cmd = "uv run python -m exsolver.experiments.e5_nodelock " + " ".join(
        shlex.quote(a) for a in (sys.argv[1:] if argv is None else list(argv))
    )
    with open(cfg.out / "README.md", "a") as f:
        f.write(
            f"\n## run — {datetime.now(UTC).isoformat(timespec='seconds')}\n\n```\n{cmd}\n```\n\n"
            f"```json\n{json.dumps(_jsonable(asdict(cfg)), indent=1)}\n```\n\n"
            f"wall time {agg['wall_seconds']:.0f} s, git {os.popen('git rev-parse HEAD').read().strip()}, "
            f"figures: {', '.join(p.name for p in figures)}\n"
        )
    return agg


if __name__ == "__main__":
    main()
