"""E1 — in-context Bayesian exploitation in Kuhn poker (spec: docs/experiments/e1-kuhn.md).

    uv run python -m exsolver.experiments.e1_kuhn {population,gen,train,eval,plots,all} [--out runs/e1]
        [--smoke] [--data data/e1a] [--hands 64] [--m 256] [--sessions 100000] [--steps 12000]
        [--batch 32] [--eval-sessions-per-opp 4] [--seed 0] [--device auto]
        [--max-train-minutes X] [--workers 8] [--retrain]

Stages write into ``--out``: ``population.npz`` plus the provenance sidecar ``population.json``
(seed, M, prior, content hash, command), the dataset (``--data``, default ``data/e1a``),
``model/`` (checkpoint + log), ``sessions.npz`` (raw per-session metrics), ``summary.json`` (the
aggregate, strict JSON with ``null`` for undefined values), ``summary.md``, the figures, and
``README.md``, to which **every** stage appends its exact command line and resolved config --
also when it skips because its outputs already exist (it says so). ``plots`` re-draws the figures
from ``summary.json`` alone.

Provenance is cross-checked with the population's content hash (sha256 of thetas + weights,
``exsolver.data.generate.population_fingerprint``): an existing ``population.npz`` is reused only
with a consistent sidecar (or, lacking one, when it matches a recomputation for the requested
seed / M); the dataset's ``meta.json`` and the checkpoint's ``data_meta`` must carry that hash;
the checkpoint's opponent head must have ``M`` outputs. Any mismatch fails loudly. ``eval`` copies
the training configuration out of the checkpoint into ``summary.json`` / ``summary.md`` so every
reported number carries the config that produced it.

Deviations from docs/experiments/e1-kuhn.md:

* training defaults are batch 32 x 12 000 steps instead of the spec's 64 x 20 000: measured MPS
  throughput for the 4 x 128 model at L = 640 is ~100-135 session-passes/s (attention backward
  is slow on MPS), so the spec budget would take several hours; ``--batch`` / ``--steps``
  override the defaults and the values actually used are recorded next to the results;
* ``--smoke`` (M = 16, 200 sessions, H = 8, 100 steps, 1 eval session per opponent) writes to
  ``runs/e1_smoke`` (dataset in ``<out>/data``) unless ``--out`` is given, and refuses
  ``--out runs/e1``;
* success criteria are evaluated pointwise on per-hand means for every ``t >= threshold``
  (stricter than the spec's "by t = 32 on average"); thresholds are clipped to ``H - 1`` for
  short runs and flagged ``scaled_to_H`` (``exsolver.eval.aggregate``);
* ``run_session`` takes the exact posterior as an object and the ``Agent`` protocol has
  ``observe`` (``exsolver.eval.session``).

The evaluator is omniscient (it holds every opponent's ``theta`` and the exact posterior) and
uses that knowledge only for metrics; agents see ``HandRecord``s and nothing else. Agents, the
exact posterior and the data generator (``exsolver.agents`` / ``exsolver.bayes`` /
``exsolver.data.generate``) are imported lazily inside the stages that need them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from exsolver.data.shards import read_meta, shard_paths
from exsolver.eval.aggregate import aggregate
from exsolver.eval.plots import write_all_plots, write_summary_md
from exsolver.eval.session import Agent, Posterior, SessionMetrics, run_session
from exsolver.games.kuhn import KuhnPoker
from exsolver.population.kuhn_prior import KuhnPrior, nash_theta, theta_to_profile
from exsolver.population.population import Population, sample_population
from exsolver.strategy import TabularStrategy

KUHN_GAME_VALUE_SEAT0 = -1.0 / 18.0
PROBE_STEPS = 20
STAGES = ("population", "gen", "train", "eval", "plots", "all")
DEFAULT_OUT = Path("runs/e1")
SMOKE_OUT = Path("runs/e1_smoke")
SMOKE = {"m": 16, "sessions": 200, "hands": 8, "steps": 100, "eval_sessions_per_opp": 1}
SPEC_BATCH, SPEC_STEPS = 64, 20_000
DEFAULT_BATCH, DEFAULT_STEPS = 32, 12_000
TRAINING_DEVIATION = (
    f"Training defaults are batch {DEFAULT_BATCH} x {DEFAULT_STEPS} steps instead of the spec's "
    f"{SPEC_BATCH} x {SPEC_STEPS}: measured MPS throughput for the 4 x 128 model at L = 640 is "
    "~100-135 session-passes/s, so the spec budget would take several hours; --batch/--steps "
    "override the defaults."
)
TRAINING_DEVIATION_REASON = (
    "measured MPS throughput for the 4 x 128 model at L = 640 is ~100-135 session-passes/s, so the "
    f"spec budget ({SPEC_BATCH} x {SPEC_STEPS}) would take several hours; the runner defaults to "
    f"{DEFAULT_BATCH} x {DEFAULT_STEPS} and --batch/--steps override it"
)
# agent names as in the spec's baseline table (aggregate / plots resolve them by these names)
AGENT_NAMES = {
    "equilibrium": "Equilibrium",
    "oracle": "OracleBR",
    "thompson": "Thompson",
    "bayes": "BayesBR",
    "transformer": "Transformer({mode})",
    "random": "Random",
}
TRANSFORMER_MODES = ("sample", "argmax")


def _same_path(a: Path, b: Path) -> bool:
    return Path(a).resolve() == Path(b).resolve()


def _inside(child: Path, parent: Path) -> bool:
    """``child`` is ``parent`` or lies below it (both resolved against the cwd)."""
    return Path(child).resolve().is_relative_to(Path(parent).resolve())


@dataclass
class E1Config:
    """Everything the runner needs; ``--smoke`` overrides the ``SMOKE`` entries."""

    out: Path = DEFAULT_OUT
    data: Path | None = None  # default data/e1a (or <out>/data under --smoke)
    m: int = 256
    hands: int = 64
    sessions: int = 100_000
    steps: int = DEFAULT_STEPS
    batch: int = DEFAULT_BATCH
    eval_sessions_per_opp: int = 4
    seed: int = 0  # population seed (spec: 0); also the data-generation seed
    eval_seed: int = 1  # spec: eval seed 1
    device: str = "auto"
    smoke: bool = False
    max_train_minutes: float | None = None
    workers: int = 8  # data-generation processes
    retrain: bool = False  # train even if <out>/model/model.pt exists
    cfr_iterations: int = 2000
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    progress: bool = True
    extra_train: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.out = Path(self.out)
        if self.data is not None:
            self.data = Path(self.data)
        if self.smoke and self.data is not None and not _inside(self.data, self.out):
            raise ValueError(
                f"--smoke must not generate into {self.data}: under --smoke the dataset directory "
                f"has to lie inside --out ({self.out}); omit --data (defaults to <out>/data)"
            )
        if self.smoke and _same_path(self.out, DEFAULT_OUT):
            raise ValueError(
                f"--smoke must not write into {DEFAULT_OUT}; use --out {SMOKE_OUT} (the default "
                "under --smoke) or another directory"
            )

    # -- derived paths ------------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        if self.data is not None:
            return self.data
        return self.out / "data" if self.smoke else Path("data/e1a")

    @property
    def population_path(self) -> Path:
        return self.out / "population.npz"

    @property
    def population_sidecar(self) -> Path:
        return self.out / "population.json"

    @property
    def model_dir(self) -> Path:
        return self.out / "model"

    @property
    def checkpoint(self) -> Path:
        return self.model_dir / "model.pt"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["out"] = str(self.out)
        d["data"] = None if self.data is None else str(self.data)
        d["data_dir"] = str(self.data_dir)
        return d


def apply_smoke(cfg: E1Config) -> E1Config:
    """The ``--smoke`` configuration: everything shrunk to run in under two minutes on CPU."""
    return replace(cfg, smoke=True, **SMOKE)


# ---------------------------------------------------------------------- bookkeeping
def _log(msg: str) -> None:
    print(f"[e1] {msg}", flush=True)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def command_line() -> str:
    """The invoking command line, re-expressed as the documented ``uv run`` form."""
    args = " ".join(shlex.quote(a) for a in sys.argv[1:])
    return f"uv run python -m exsolver.experiments.e1_kuhn {args}".rstrip()


def strict_json(obj: Any) -> Any:
    """Recursively convert to strict-JSON builtins: NaN / inf -> ``None``, numpy -> Python."""
    if isinstance(obj, dict):
        return {str(k): strict_json(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [strict_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return strict_json(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def dump_json(obj: Any, path: Path, indent: int = 1) -> None:
    path.write_text(json.dumps(strict_json(obj), indent=indent, allow_nan=False) + "\n")


def record_command(cfg: E1Config, stage: str, info: dict[str, Any]) -> None:
    """Append the exact command line, the resolved config and ``info`` to ``<out>/README.md``.

    Called by every stage, including when it skips work (``info["skipped"]`` says so).
    """
    cfg.out.mkdir(parents=True, exist_ok=True)
    readme = cfg.out / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# {cfg.out}\n\nEvery stage appends its exact command line and resolved config "
            "here so each number in `summary.md` can be regenerated; skipped stages are recorded "
            "too.\n\nDeviations from docs/experiments/e1-kuhn.md: see the docstring of "
            f"`exsolver.experiments.e1_kuhn`. {TRAINING_DEVIATION}\n"
        )
    lines = [
        "",
        f"## {stage} — {_now()}" + (" (skipped)" if info.get("skipped") else ""),
        "",
        "```",
        command_line(),
        "```",
        "",
        "```json",
        json.dumps(strict_json({"config": cfg.to_dict(), **info}), indent=2, allow_nan=False),
        "```",
    ]
    with open(readme, "a") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------- population
def population_fingerprint(pop: Population) -> str:
    """sha256 of the population's thetas and weights (``exsolver.data.generate`` helper)."""
    from exsolver.data.generate import population_fingerprint as fingerprint

    return fingerprint(pop)


def write_population_sidecar(cfg: E1Config, pop: Population) -> dict[str, Any]:
    info = {
        "seed": int(cfg.seed),
        "m": int(cfg.m),
        "prior": type(KuhnPrior()).__name__,
        "content_sha256": population_fingerprint(pop),
        "created_at": _now(),
        "command": command_line(),
    }
    dump_json(info, cfg.population_sidecar, indent=2)
    return info


def load_or_sample_population(cfg: E1Config) -> tuple[Population, dict[str, Any]]:
    """Population for ``cfg`` with provenance checks; returns ``(population, record info)``.

    * no ``population.npz``: sample ``M`` opponents from ``KuhnPrior`` with ``seed``, save it and
      write the sidecar;
    * ``population.npz`` + ``population.json``: reuse only if the sidecar's seed, M and content
      hash match the request and the file;
    * ``population.npz`` without sidecar: recompute the population for the requested seed / M and
      reuse only if the content hashes agree (then write the sidecar); otherwise refuse.
    """
    path, sidecar = cfg.population_path, cfg.population_sidecar
    if not path.exists():
        pop = sample_population(KuhnPrior(), cfg.m, np.random.default_rng(cfg.seed))
        cfg.out.mkdir(parents=True, exist_ok=True)
        pop.save(path)
        side = write_population_sidecar(cfg, pop)
        counts = {n: int((pop.archetypes == k).sum()) for k, n in enumerate(pop.archetype_names)}
        return pop, {
            "skipped": False,
            "path": str(path),
            "sidecar": side,
            "archetype_counts": counts,
        }
    pop = Population.load(path)
    actual = population_fingerprint(pop)
    if sidecar.exists():
        side = json.loads(sidecar.read_text())
        problems = []
        for key, want in (("seed", cfg.seed), ("m", cfg.m), ("content_sha256", actual)):
            if side.get(key) != want:
                problems.append(f"{key}: sidecar has {side.get(key)!r}, this run needs {want!r}")
        if len(pop) != cfg.m:
            problems.append(f"size: file holds {len(pop)} opponents, --m is {cfg.m}")
        if problems:
            raise RuntimeError(
                f"{path} does not match this run ({'; '.join(problems)}); delete it or change --out"
            )
        reason = "population.npz and population.json present and consistent"
    else:
        expected = sample_population(KuhnPrior(), cfg.m, np.random.default_rng(cfg.seed))
        if len(pop) != cfg.m or population_fingerprint(expected) != actual:
            raise RuntimeError(
                f"{path} has no provenance sidecar and its content does not match KuhnPrior with "
                f"seed={cfg.seed}, M={cfg.m}; delete it or change --out"
            )
        side = write_population_sidecar(cfg, pop)
        reason = "population.npz matched a recomputation for seed / M; sidecar written"
    return pop, {"skipped": True, "reason": reason, "path": str(path), "sidecar": side}


def stage_population(cfg: E1Config) -> Population:
    """Sample ``M`` opponents from ``KuhnPrior`` with ``seed`` into ``<out>/population.npz``."""
    t0 = time.perf_counter()
    pop, info = load_or_sample_population(cfg)
    if info["skipped"]:
        _log(
            f"population: {cfg.population_path} exists (M={len(pop)}), skipping -- {info['reason']}"
        )
    else:
        _log(
            f"population: wrote {cfg.population_path} (M={cfg.m}, seed={cfg.seed}, "
            f"archetypes={info['archetype_counts']})"
        )
    record_command(cfg, "population", {**info, "elapsed_s": time.perf_counter() - t0})
    return pop


# ---------------------------------------------------------------------- dataset
def dataset_status(cfg: E1Config, pop: Population) -> str:
    """``missing`` | ``match`` | ``mismatch`` (H / session count) | ``population_mismatch``."""
    if not shard_paths(cfg.data_dir):
        return "missing"
    meta = read_meta(cfg.data_dir)
    if int(meta.get("H", -1)) != cfg.hands or int(meta.get("n_sessions", -1)) != cfg.sessions:
        return "mismatch"
    have = (meta.get("population") or {}).get("content_sha256")
    if have != population_fingerprint(pop):
        return "population_mismatch"
    return "match"


def require_dataset(cfg: E1Config, pop: Population) -> dict[str, Any]:
    """``meta.json`` of a dataset that matches ``cfg`` and ``pop``; raises otherwise."""
    status = dataset_status(cfg, pop)
    if status == "match":
        return read_meta(cfg.data_dir)
    if status == "missing":
        raise FileNotFoundError(f"no dataset in {cfg.data_dir}; run the gen stage first")
    if status == "mismatch":
        raise RuntimeError(
            f"{cfg.data_dir} holds a dataset with a different H / session count than "
            f"H={cfg.hands}, n={cfg.sessions}; delete it or pass --data"
        )
    raise RuntimeError(
        f"{cfg.data_dir}/meta.json population.content_sha256 does not match the population in "
        f"{cfg.population_path}: the dataset was generated from a different population; "
        "regenerate it or pass --data"
    )


def stage_gen(cfg: E1Config) -> Path:
    """Generate the DPT dataset with ``exsolver.data.generate.generate_dataset``."""
    t0 = time.perf_counter()
    pop, _ = load_or_sample_population(cfg)
    status = dataset_status(cfg, pop)
    if status == "match":
        _log(
            f"gen: {cfg.data_dir} already holds a matching dataset (H={cfg.hands}, n={cfg.sessions}), skipping"
        )
        record_command(
            cfg,
            "gen",
            {
                "skipped": True,
                "reason": "dataset present with matching H, session count and population hash",
                "data_dir": str(cfg.data_dir),
                "elapsed_s": time.perf_counter() - t0,
            },
        )
        return cfg.data_dir
    if status != "missing":
        require_dataset(cfg, pop)  # raises with the specific reason
    from exsolver.data.generate import generate_dataset  # imported lazily (multiprocessing)
    from exsolver.data.tokenizer import Tokenizer

    game = KuhnPoker()
    meta = generate_dataset(
        game,
        pop,
        Tokenizer(game.spec),
        cfg.data_dir,
        n_sessions=cfg.sessions,
        hands_per_session=cfg.hands,
        seed=cfg.seed,
        n_workers=max(1, cfg.workers),
        population_path=cfg.population_path,
        progress=cfg.progress,
    )
    keep = (
        "n_sessions",
        "H",
        "L",
        "n_shards",
        "n_decisions",
        "collector_counts",
        "elapsed_s",
        "population",
    )
    info = {k: meta.get(k) for k in keep}
    _log(
        f"gen: wrote {cfg.data_dir} ({meta['n_sessions']} sessions x {meta['H']} hands, "
        f"L={meta['L']}) in {time.perf_counter() - t0:.1f}s"
    )
    record_command(
        cfg,
        "gen",
        {
            "skipped": False,
            "data_dir": str(cfg.data_dir),
            "meta": info,
            "elapsed_s": time.perf_counter() - t0,
        },
    )
    return cfg.data_dir


# ---------------------------------------------------------------------- training
def cap_steps(
    requested: int, probe_steps: int, probe_seconds: float, budget_minutes: float
) -> tuple[int, float]:
    """Steps that fit the wall-clock budget at the probe's rate; returns ``(steps, steps_per_sec)``.

    The probe time already spent is subtracted from the budget; never fewer than ``probe_steps``.
    """
    rate = probe_steps / max(probe_seconds, 1e-9)
    remaining = max(budget_minutes * 60.0 - probe_seconds, 0.0)
    fit = int(rate * remaining)
    return max(probe_steps, min(requested, fit)), rate


def stage_train(cfg: E1Config) -> dict[str, Any]:
    """Train the transformer with ``exsolver.train.train``; ``--max-train-minutes`` caps the steps.

    Skips (and says so) when ``<out>/model/model.pt`` exists unless ``--retrain`` is given.
    """
    from exsolver.train import TrainConfig, train

    t0 = time.perf_counter()
    pop, _ = load_or_sample_population(cfg)
    data_meta = require_dataset(cfg, pop)
    if cfg.checkpoint.exists() and not cfg.retrain:
        summary_path = cfg.model_dir / "train_summary.json"
        previous = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        _log(f"train: {cfg.checkpoint} exists, skipping (pass --retrain to train again)")
        info = {
            "skipped": True,
            "reason": "checkpoint present; --retrain not given",
            "checkpoint": str(cfg.checkpoint),
            "previous_summary": {k: v for k, v in previous.items() if k != "model_config"},
            "elapsed_s": time.perf_counter() - t0,
        }
        record_command(cfg, "train", info)
        return {"skipped": True, "checkpoint": str(cfg.checkpoint), **previous}
    base = TrainConfig(
        data_dir=str(cfg.data_dir),
        out_dir=str(cfg.model_dir),
        steps=cfg.steps,
        batch_size=cfg.batch,
        warmup=min(500, max(1, cfg.steps // 10)),
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        max_len=max(1024, int(data_meta.get("L", 0))),
        seed=cfg.seed,
        device=cfg.device,
        eval_every=min(500, cfg.steps),
        progress=cfg.progress,
        **cfg.extra_train,
    )
    budget: dict[str, Any] = {
        "max_train_minutes": cfg.max_train_minutes,
        "requested_steps": cfg.steps,
    }
    if cfg.max_train_minutes is not None:
        probe_dir = cfg.out / "model_probe"
        probe_cfg = replace(
            base,
            steps=PROBE_STEPS,
            out_dir=str(probe_dir),
            eval_every=0,
            eval_size=0,
            progress=False,
        )
        _log(f"train: measuring speed over {PROBE_STEPS} steps ...")
        probe = train(probe_cfg)
        shutil.rmtree(probe_dir, ignore_errors=True)
        steps, rate = cap_steps(
            cfg.steps, PROBE_STEPS, float(probe["elapsed"]), cfg.max_train_minutes
        )
        budget.update(
            {"probe_elapsed_s": probe["elapsed"], "steps_per_sec": rate, "capped_steps": steps}
        )
        if steps < cfg.steps:
            _log(
                f"train: {rate:.2f} steps/s measured; {cfg.steps} steps would take "
                f"{cfg.steps / rate / 60:.1f} min > budget {cfg.max_train_minutes} min -> reducing to {steps} steps"
            )
        else:
            _log(
                f"train: {rate:.2f} steps/s measured; {cfg.steps} steps fit the {cfg.max_train_minutes} min budget"
            )
        base = replace(
            base,
            steps=steps,
            warmup=min(base.warmup, max(1, steps // 10)),
            eval_every=min(500, steps),
        )
    summary = train(base)
    summary["budget"] = budget
    summary["population_content_sha256"] = population_fingerprint(pop)
    if (base.batch_size, base.steps) != (SPEC_BATCH, SPEC_STEPS):
        summary["spec_deviation"] = TRAINING_DEVIATION
    dump_json(summary, cfg.model_dir / "train_summary.json", indent=2)
    _log(
        f"train: {summary['steps']} steps in {summary['elapsed']:.0f}s "
        f"({summary['steps_per_sec']:.2f} steps/s); eval {summary.get('eval')}"
    )
    record_command(
        cfg,
        "train",
        {
            "skipped": False,
            "train_config": asdict(base),
            "summary": {k: v for k, v in summary.items() if k != "model_config"},
            "elapsed_s": time.perf_counter() - t0,
        },
    )
    return summary


# ---------------------------------------------------------------------- agents
def build_agents(
    game: KuhnPoker, pop: Population, policy: Any, tokenizer: Any, *, cfr_iterations: int
) -> tuple[dict[str, Agent], Callable[[TabularStrategy], Agent]]:
    """The spec's baseline agents (``exsolver.agents``) plus a per-opponent ``OracleBR`` factory.

    Only the oracle factory receives an opponent profile; the transformer sees the policy and the
    tokenizer, the exact-Bayes agents the population (their prior), nothing else.
    """
    from exsolver.agents import (
        BayesBRAgent,
        EquilibriumAgent,
        OracleBRAgent,
        RandomAgent,
        ThompsonAgent,
        TransformerAgent,
    )

    agents: list[Agent] = [
        EquilibriumAgent(game, iterations=cfr_iterations, name=AGENT_NAMES["equilibrium"]),
        ThompsonAgent(game, pop, name=AGENT_NAMES["thompson"]),
        BayesBRAgent(game, pop, name=AGENT_NAMES["bayes"]),
    ]
    agents += [
        TransformerAgent(
            game, policy, tokenizer, mode=mode, name=AGENT_NAMES["transformer"].format(mode=mode)
        )
        for mode in TRANSFORMER_MODES
    ]
    agents.append(RandomAgent(game, name=AGENT_NAMES["random"]))

    def oracle_factory(opp_profile: TabularStrategy) -> Agent:
        return OracleBRAgent(game, opp_profile, name=AGENT_NAMES["oracle"])

    return {a.name: a for a in agents}, oracle_factory


def build_posterior_factory(game: KuhnPoker, pop: Population) -> Callable[[], Posterior]:
    """Fresh exact posterior per session (evaluator-side only)."""
    from exsolver.bayes.posterior import ExactPosterior

    return lambda: ExactPosterior(game, pop)


# ---------------------------------------------------------------------- evaluation
def evaluate_population(
    game: KuhnPoker,
    pop: Population,
    agents: dict[str, Agent],
    *,
    oracle_factory: Callable[[TabularStrategy], Agent] | None,
    posterior_factory: Callable[[], Posterior] | None,
    H: int,
    sessions_per_opp: int,
    seed: int,
    equilibrium_value_seat0: float = KUHN_GAME_VALUE_SEAT0,
    progress: bool = False,
) -> list[SessionMetrics]:
    """Every opponent x ``sessions_per_opp`` sessions x every agent.

    The generator for ``(opp_id, session)`` is seeded identically for every agent, and
    ``run_session`` splits it into deal / opponent / agent streams, so all agents face the same
    cards. ``oracle_factory(theta)`` builds the per-opponent oracle; ``posterior_factory()`` a
    fresh exact posterior per session (evaluator-side only).
    """
    metrics: list[SessionMetrics] = []
    it = tqdm(
        range(len(pop)), disable=not progress, desc="eval", file=sys.stderr, dynamic_ncols=True
    )
    for opp_id in it:
        profile = pop.profiles[opp_id]
        roster = dict(agents)
        if oracle_factory is not None:
            oracle = oracle_factory(profile)
            roster[oracle.name] = oracle
        for s in range(sessions_per_opp):
            for agent in roster.values():
                rng = np.random.default_rng([seed, opp_id, s])
                posterior = posterior_factory() if posterior_factory is not None else None
                metrics.append(
                    run_session(
                        game,
                        agent,
                        profile,
                        H,
                        rng,
                        posterior=posterior,
                        equilibrium_value_seat0=equilibrium_value_seat0,
                        opp_id=opp_id,
                        archetype=int(pop.archetypes[opp_id]),
                        session=s,
                        n_opp=len(pop),
                    )
                )
    return metrics


def equilibrium_sanity(
    game: KuhnPoker, agent: Agent, H: int = 8, alpha: float = 1.0 / 6.0
) -> dict[str, Any]:
    """EV of ``agent`` against an exact Nash opponent must be -1/18 at seat 0 and +1/18 at seat 1."""
    nash = theta_to_profile(nash_theta(alpha))
    m = run_session(
        game,
        agent,
        nash,
        H,
        np.random.default_rng(0),
        equilibrium_value_seat0=KUHN_GAME_VALUE_SEAT0,
    )
    target = np.where(m.seats == 0, KUHN_GAME_VALUE_SEAT0, -KUHN_GAME_VALUE_SEAT0)
    err = float(np.max(np.abs(m.ev - target)))
    return {
        "agent": agent.name,
        "ev_seat0": float(m.ev[m.seats == 0].mean()),
        "ev_seat1": float(m.ev[m.seats == 1].mean()) if H > 1 else float("nan"),
        "target": [KUHN_GAME_VALUE_SEAT0, -KUHN_GAME_VALUE_SEAT0],
        "max_abs_error": err,
        "threshold": 1e-3,
        "pass": bool(err < 1e-3),
    }


def save_sessions(metrics: Sequence[SessionMetrics], path: Path) -> None:
    """Raw per-session arrays (``[N, H]`` per metric) so any aggregate can be recomputed."""
    fields = ("seats", "ev", "realized", "expl", "post_entropy", "agent_entropy", "kl")
    arrays: dict[str, np.ndarray] = {f: np.stack([getattr(m, f) for m in metrics]) for f in fields}
    for probe in metrics[0].probes:
        arrays[f"probe_{probe}"] = np.stack([m.probes[probe] for m in metrics])
    arrays["agent"] = np.asarray([m.agent for m in metrics], dtype=str)
    arrays["opp_id"] = np.asarray([m.opp_id for m in metrics], dtype=np.int64)
    arrays["archetype"] = np.asarray([m.archetype for m in metrics], dtype=np.int64)
    arrays["session"] = np.asarray([m.session for m in metrics], dtype=np.int64)
    arrays["n_showdowns"] = np.asarray([m.n_showdowns for m in metrics], dtype=np.int64)
    np.savez_compressed(path, **arrays)


def check_checkpoint(meta: dict[str, Any], model_cfg: Any, pop: Population) -> str:
    """The checkpoint must come from data of this population and have an ``M``-way opponent head."""
    fp = population_fingerprint(pop)
    have = ((meta.get("data_meta") or {}).get("population") or {}).get("content_sha256")
    if have != fp:
        raise RuntimeError(
            "checkpoint was trained on data from a different population: data_meta "
            f"population.content_sha256 {have!r} != {fp!r} (population in --out)"
        )
    if model_cfg.n_opp != len(pop):
        raise RuntimeError(
            f"checkpoint opponent head has {model_cfg.n_opp} outputs but the population has {len(pop)}"
        )
    return fp


def training_provenance(meta: dict[str, Any]) -> dict[str, Any]:
    """Training configuration read back from a checkpoint (``load_checkpoint`` meta).

    Model dimensions come from the checkpoint's ``model_config`` (what was built), the rest from
    its ``train_config`` and ``data_meta``. ``spec_deviation`` states the batch / steps *this
    checkpoint* used against the spec's ``64 x 20 000`` (``None`` when they coincide).
    """
    tc = meta.get("train_config") or {}
    dm = meta.get("data_meta") or {}
    mc = meta.get("model_config") or {}  # the architecture actually built (authoritative)
    batch, steps = tc.get("batch_size"), tc.get("steps")
    deviation = None
    if (batch, steps) != (SPEC_BATCH, SPEC_STEPS):
        deviation = (
            f"This run used batch {batch} \u00d7 {steps} steps (spec: {SPEC_BATCH} \u00d7 {SPEC_STEPS}); "
            f"{TRAINING_DEVIATION_REASON}."
        )
    return {
        "checkpoint_step": meta.get("step"),
        "steps": steps,
        "batch_size": batch,
        "spec_steps": SPEC_STEPS,
        "spec_batch_size": SPEC_BATCH,
        "lr": tc.get("lr"),
        "warmup": tc.get("warmup"),
        "weight_decay": tc.get("weight_decay"),
        "lambda_opp": tc.get("lambda_opp"),
        "device": tc.get("device"),
        "seed": tc.get("seed"),
        "data_dir": tc.get("data_dir"),
        "model": {
            k: mc.get(k, tc.get(k)) for k in ("d_model", "n_layers", "n_heads", "max_len", "n_opp")
        },
        "data": {k: dm.get(k) for k in ("n_sessions", "H", "L", "seed", "collection_mix")},
        "population_content_sha256": (dm.get("population") or {}).get("content_sha256"),
        "deviation_reason": TRAINING_DEVIATION_REASON,
        "spec_deviation": deviation,
    }


def write_eval_outputs(
    agg: dict[str, Any], out: Path, *, run_info: dict[str, Any] | None = None
) -> list[Path]:
    """``summary.json`` (strict JSON), ``summary.md`` and every figure into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    dump_json(agg, out / "summary.json")
    paths = write_all_plots(agg, out)
    paths.append(write_summary_md(agg, out / "summary.md", extra=run_info))
    paths.append(out / "summary.json")
    return paths


def stage_eval(cfg: E1Config) -> dict[str, Any]:
    """Evaluate every agent against every opponent; write plots, summary.md/json and README."""
    from exsolver.model.inference import Policy
    from exsolver.train import load_checkpoint, resolve_device, tokenizer_from_checkpoint

    t0 = time.perf_counter()
    game = KuhnPoker()
    pop, _ = load_or_sample_population(cfg)
    if not cfg.checkpoint.exists():
        raise FileNotFoundError(f"no checkpoint at {cfg.checkpoint}; run the train stage first")
    device = resolve_device(cfg.device)
    model, model_cfg, meta = load_checkpoint(cfg.checkpoint, device)
    fp = check_checkpoint(meta, model_cfg, pop)
    training = training_provenance(meta)
    tokenizer = tokenizer_from_checkpoint(meta)
    policy = Policy(model, tokenizer, device)
    agents, oracle_factory = build_agents(
        game, pop, policy, tokenizer, cfr_iterations=cfg.cfr_iterations
    )
    posterior_factory = build_posterior_factory(game, pop)
    _log(
        f"eval: agents {list(agents)} + {AGENT_NAMES['oracle']} per opponent; {len(pop)} opponents x "
        f"{cfg.eval_sessions_per_opp} sessions x H={cfg.hands}; device {device}; checkpoint step "
        f"{training['checkpoint_step']} (batch {training['batch_size']} x {training['steps']} steps)"
    )

    t_eval = time.perf_counter()
    metrics = evaluate_population(
        game,
        pop,
        agents,
        oracle_factory=oracle_factory,
        posterior_factory=posterior_factory,
        H=cfg.hands,
        sessions_per_opp=cfg.eval_sessions_per_opp,
        seed=cfg.eval_seed,
        progress=cfg.progress,
    )
    eval_seconds = time.perf_counter() - t_eval
    save_sessions(metrics, cfg.out / "sessions.npz")

    agg = aggregate(metrics, reference=AGENT_NAMES["oracle"], archetype_names=pop.archetype_names)
    agg["n_population"] = len(pop)
    eq_name = agg["criteria"]["resolved_agents"].get("equilibrium")
    if eq_name is not None:
        agg["criteria"]["sanity_equilibrium_vs_nash"] = equilibrium_sanity(
            game, agents[eq_name], H=min(cfg.hands, 8)
        )
    run_info = {
        "command": command_line(),
        "config": json.dumps(cfg.to_dict(), default=str),
        "checkpoint": str(cfg.checkpoint),
        "checkpoint_step": meta.get("step"),
        "population_content_sha256": fp,
        "device": str(device),
        "eval_seconds": f"{eval_seconds:.1f}",
        "sessions_file": str(cfg.out / "sessions.npz"),
    }
    agg["run"] = run_info
    agg["training"] = training
    paths = write_eval_outputs(agg, cfg.out, run_info=run_info)
    verdicts = {
        k: v.get("pass") for k, v in agg["criteria"].items() if isinstance(v, dict) and "pass" in v
    }
    _log(
        f"eval: {len(metrics)} sessions in {eval_seconds:.0f}s; criteria {verdicts}; wrote {[p.name for p in paths]}"
    )
    record_command(
        cfg,
        "eval",
        {
            "skipped": False,
            "n_sessions": len(metrics),
            "eval_seconds": eval_seconds,
            "criteria": verdicts,
            "training": training,
            "outputs": [str(p) for p in paths],
            "elapsed_s": time.perf_counter() - t0,
        },
    )
    return agg


def stage_plots(cfg: E1Config) -> list[Path]:
    """Re-draw every figure (and summary.md) from ``<out>/summary.json``."""
    t0 = time.perf_counter()
    path = cfg.out / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"no {path}; run the eval stage first")
    agg = json.loads(path.read_text())
    paths = write_all_plots(agg, cfg.out)
    paths.append(write_summary_md(agg, cfg.out / "summary.md", extra=agg.get("run")))
    _log(f"plots: wrote {[p.name for p in paths]}")
    record_command(
        cfg,
        "plots",
        {
            "skipped": False,
            "source": str(path),
            "outputs": [str(p) for p in paths],
            "elapsed_s": time.perf_counter() - t0,
        },
    )
    return paths


# ---------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("stage", choices=STAGES)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"run directory (default {DEFAULT_OUT}; {SMOKE_OUT} under --smoke)",
    )
    p.add_argument(
        "--smoke", action="store_true", help=f"shrink to {SMOKE} (under two minutes on CPU)"
    )
    p.add_argument(
        "--data",
        type=Path,
        default=None,
        help="dataset directory (default data/e1a; <out>/data under --smoke)",
    )
    p.add_argument("--hands", type=int, default=E1Config.hands, help="hands per session H")
    p.add_argument("--m", type=int, default=E1Config.m, help="population size M")
    p.add_argument("--sessions", type=int, default=E1Config.sessions, help="training sessions")
    p.add_argument(
        "--steps", type=int, default=E1Config.steps, help=f"training steps (spec: {SPEC_STEPS})"
    )
    p.add_argument(
        "--batch",
        type=int,
        default=E1Config.batch,
        help=f"training batch size (spec: {SPEC_BATCH})",
    )
    p.add_argument("--eval-sessions-per-opp", type=int, default=E1Config.eval_sessions_per_opp)
    p.add_argument("--seed", type=int, default=E1Config.seed, help="population / data seed")
    p.add_argument("--eval-seed", type=int, default=E1Config.eval_seed)
    p.add_argument("--device", default=E1Config.device, help="auto | cpu | mps")
    p.add_argument(
        "--max-train-minutes",
        type=float,
        default=None,
        help="cap training wall-clock; steps are reduced after a 20-step speed probe",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=min(E1Config.workers, os.cpu_count() or 1),
        help="data-generation processes",
    )
    p.add_argument(
        "--retrain", action="store_true", help="train even if <out>/model/model.pt exists"
    )
    p.add_argument("--cfr-iterations", type=int, default=E1Config.cfr_iterations)
    p.add_argument("--no-progress", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> E1Config:
    """Resolve the CLI namespace; ``--smoke`` defaults to ``runs/e1_smoke`` and refuses ``runs/e1``."""
    if args.smoke and args.out is not None and _same_path(args.out, DEFAULT_OUT):
        raise ValueError(
            f"--smoke must not write into {DEFAULT_OUT}; omit --out (defaults to {SMOKE_OUT}) or "
            "pass another directory"
        )
    out = args.out if args.out is not None else (SMOKE_OUT if args.smoke else DEFAULT_OUT)
    cfg = E1Config(
        out=out,
        data=args.data,
        m=args.m,
        hands=args.hands,
        sessions=args.sessions,
        steps=args.steps,
        batch=args.batch,
        eval_sessions_per_opp=args.eval_sessions_per_opp,
        seed=args.seed,
        eval_seed=args.eval_seed,
        device=args.device,
        max_train_minutes=args.max_train_minutes,
        workers=args.workers,
        retrain=args.retrain,
        cfr_iterations=args.cfr_iterations,
        progress=not args.no_progress,
    )
    return apply_smoke(cfg) if args.smoke else cfg


def run_stage(stage: str, cfg: E1Config) -> Any:
    if stage == "population":
        return stage_population(cfg)
    if stage == "gen":
        return stage_gen(cfg)
    if stage == "train":
        return stage_train(cfg)
    if stage == "eval":
        return stage_eval(cfg)
    if stage == "plots":
        return stage_plots(cfg)
    if stage == "all":
        stage_population(cfg)
        stage_gen(cfg)
        stage_train(cfg)
        return stage_eval(cfg)
    raise ValueError(f"unknown stage {stage!r}")


def main(argv: Sequence[str] | None = None) -> Any:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = config_from_args(args)
    except ValueError as e:
        parser.error(str(e))
    t0 = time.perf_counter()
    result = run_stage(args.stage, cfg)
    _log(f"{args.stage}: done in {time.perf_counter() - t0:.1f}s (out={cfg.out})")
    return result


if __name__ == "__main__":
    main()
