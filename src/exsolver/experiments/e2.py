"""E2 -- decision-relevant inference, label semantics, training convergence (evaluator half).

Spec: docs/experiments/e2-decision-relevant-inference.md. ::

    uv run python -m exsolver.experiments.e2 eval --out runs/e2 --data data/e1a \\
        --conditions A=runs/e1 B=runs/e2/B C=runs/e2/C D=runs/e2/D [E=runs/e2/E] \\
        [--eval-sessions-per-opp 4] [--hands 64] [--eval-seed 1] [--device auto] [--smoke]
    uv run python -m exsolver.experiments.e2 plots --out runs/e2

A *condition* is a directory holding ``model/model.pt`` (a checkpoint written by
``exsolver.train``) and ``population.npz``. Every condition must be trained on the **same**
population: the content hashes (``population_fingerprint``) of all present ``population.npz``
files, of the dataset ``--data`` (its ``meta.json``) and of every checkpoint's ``data_meta`` have
to agree, otherwise the run is refused. A condition whose checkpoint does not exist yet is
reported as **pending** (the training queue is still running) and skipped; it never fails the
run. Condition directories are read only; everything is written under ``--out``.

The checkpoint-independent exact references -- Equilibrium, Thompson, BayesBR, PluralityBR,
Random and the per-opponent OracleBR -- are evaluated **once** and reused for every condition;
each condition adds Transformer(sample) and Transformer(argmax). All agents face identical deals
(``e1_kuhn.evaluate_population`` seeds ``(eval_seed, opp_id, session)``), so every comparison is
paired. The evaluator holds the exact posterior and the ``BRActionReference`` (``pi*_t`` and the
reach weights) for the policy-level KL; agents see nothing but their hands.

Outputs: ``<out>/<cond>/summary.{json,md}`` + the E1 figure set + ``kl_policy_vs_hand.png`` per
condition (its own E2-1 checks with plain agent names), ``<out>/summary.{json,md}`` with the E2
criteria table (E2-1 for the baseline, E2-2 for the Bayes-BR-label condition, E2-3 for the
convergence / ``lambda_opp`` conditions) and the per-condition table, ``conditions_ev.png``,
``kl_policy_vs_hand.png``, ``entropy_vs_hand.png`` across conditions, ``sessions.npz`` per
condition and ``references_sessions.npz``, and ``README.md`` to which every stage appends its
exact command line and resolved config. ``summary.json`` is strict JSON (``null`` for NaN).

``--smoke`` writes under ``runs/e2_smoke`` (or ``--out`` outside ``runs/e1``, ``runs/e2`` and
``data/e1a``), shrinks the evaluation to H = 8 and one session per opponent and, when no
``--conditions`` are given, trains two tiny conditions (1 x 32 model; ``A``: 40 steps, ``B``: 20
steps; one shared population and dataset) with the E1 stages inside ``--out`` first.

Deviations from the spec: the criteria are evaluated pointwise on per-hand means for every
``t >= threshold`` (as in E1); E2-2 is reported for both transformer modes (the spec does not
say which one "condition E" means); E2-3's identity-KL comparison uses Transformer(sample);
E2-3's "EV change" is the two-sided paired change of both modes for ``t >= 16``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from exsolver.data.shards import read_meta
from exsolver.eval.aggregate import (
    E2Roles,
    E2Thresholds,
    aggregate,
    check_e2_criteria,
    group_by_agent,
    tag_condition,
)
from exsolver.eval.plots import (
    criteria_table,
    md_table,
    notes,
    plot_conditions_ev,
    plot_entropy_vs_hand,
    plot_kl_policy_vs_hand,
    training_table,
)
from exsolver.eval.session import Agent, SessionMetrics
from exsolver.experiments import e1_kuhn as e1
from exsolver.games.kuhn import KuhnPoker
from exsolver.population.population import Population

STAGES = ("eval", "plots")
DEFAULT_OUT = Path("runs/e2")
SMOKE_OUT = Path("runs/e2_smoke")
PROTECTED = (Path("runs/e1"), Path("runs/e2"), Path("data/e1a"))
SMOKE = {"hands": 8, "eval_sessions_per_opp": 1}
SMOKE_CONDITION_STEPS = {"A": 40, "B": 20}
SMOKE_MODEL = {"d_model": 32, "n_layers": 1, "n_heads": 2}
AGENT_NAMES = {**e1.AGENT_NAMES, "plurality": "PluralityBR"}
REFERENCE_AGENTS = ("equilibrium", "thompson", "bayes", "plurality", "random")
COMBINED_FIGURES = {
    "conditions_ev.png": plot_conditions_ev,
    "kl_policy_vs_hand.png": plot_kl_policy_vs_hand,
    "entropy_vs_hand.png": plot_entropy_vs_hand,
}
PENDING, EVALUATED = "pending", "evaluated"


# ---------------------------------------------------------------------- configuration
@dataclass
class E2Config:
    out: Path = DEFAULT_OUT
    data: Path | None = None  # dataset whose meta.json population hash is cross-checked
    conditions: dict[str, Path] = field(default_factory=dict)  # id -> directory, in order
    hands: int = 64
    eval_sessions_per_opp: int = 4
    eval_seed: int = 1  # as E1: identical deals to runs/e1
    device: str = "auto"
    smoke: bool = False
    cfr_iterations: int = 2000
    progress: bool = True
    roles: E2Roles = field(default_factory=E2Roles)
    thresholds: E2Thresholds = field(default_factory=E2Thresholds)

    def __post_init__(self) -> None:
        self.out = Path(self.out)
        if self.data is not None:
            self.data = Path(self.data)
        self.conditions = {str(k): Path(v) for k, v in self.conditions.items()}
        for cid in self.conditions:
            if not cid or any(ch in cid for ch in "[]/ "):
                raise ValueError(
                    f"condition id {cid!r} must be non-empty without '[', ']', '/' or spaces"
                )
        if self.smoke:
            for prot in PROTECTED:
                if e1._inside(self.out, prot):
                    raise ValueError(
                        f"--smoke must not write into {prot}; use --out {SMOKE_OUT} (the default "
                        "under --smoke) or another directory"
                    )
            if self.data is not None and not e1._inside(self.data, self.out):
                raise ValueError(
                    f"--smoke must not read a dataset outside --out ({self.out}); omit --data"
                )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["out"] = str(self.out)
        d["data"] = None if self.data is None else str(self.data)
        d["conditions"] = {k: str(v) for k, v in self.conditions.items()}
        return d

    def condition_out(self, cid: str) -> Path:
        return self.out / cid


def apply_smoke(cfg: E2Config) -> E2Config:
    return replace(cfg, smoke=True, **SMOKE)


def parse_conditions(items: Sequence[str]) -> dict[str, Path]:
    """``["A=runs/e1", "B=runs/e2/B"]`` -> ordered ``{"A": Path("runs/e1"), ...}``."""
    out: dict[str, Path] = {}
    for item in items:
        cid, sep, path = item.partition("=")
        if not sep or not cid or not path:
            raise ValueError(f"condition {item!r} must have the form ID=PATH")
        if cid in out:
            raise ValueError(f"condition id {cid!r} given twice")
        out[cid] = Path(path)
    return out


# ---------------------------------------------------------------------- bookkeeping
def _log(msg: str) -> None:
    print(f"[e2] {msg}", flush=True)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def command_line() -> str:
    args = " ".join(shlex.quote(a) for a in sys.argv[1:])
    return f"uv run python -m exsolver.experiments.e2 {args}".rstrip()


def record_command(cfg: E2Config, stage: str, info: dict[str, Any]) -> None:
    """Append the exact command line, the resolved config and ``info`` to ``<out>/README.md``."""
    cfg.out.mkdir(parents=True, exist_ok=True)
    readme = cfg.out / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# {cfg.out}\n\nE2 evaluation (spec docs/experiments/e2-decision-relevant-inference.md). "
            "Every stage appends its exact command line and resolved config here so each number "
            "in `summary.md` can be regenerated. Condition directories are inputs and are never "
            "written to; deviations from the spec are listed in the docstring of "
            "`exsolver.experiments.e2`.\n"
        )
    lines = [
        "",
        f"## {stage} — {_now()}",
        "",
        "```",
        command_line(),
        "```",
        "",
        "```json",
        json.dumps(e1.strict_json({"config": cfg.to_dict(), **info}), indent=2, allow_nan=False),
        "```",
    ]
    with open(readme, "a") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------- conditions
@dataclass
class Condition:
    id: str
    path: Path
    status: str  # PENDING | EVALUATED (set after evaluation)
    checkpoint: Path
    population_path: Path
    population_sha256: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("path", "checkpoint", "population_path"):
            d[k] = str(d[k])
        return d


def inspect_conditions(cfg: E2Config) -> tuple[Population, str, list[Condition]]:
    """Load the shared population and classify every condition; refuse hash mismatches.

    A condition with ``population.npz`` and ``model/model.pt`` is ready; one with neither, or
    with the population only, is pending; a checkpoint without a population is an error. All
    present populations (and ``--data``'s ``meta.json``, when given) must share one content hash.
    """
    if not cfg.conditions:
        raise ValueError("no conditions given (--conditions ID=PATH ...)")
    conds: list[Condition] = []
    pops: dict[str, tuple[Population, str]] = {}
    for cid, path in cfg.conditions.items():
        ckpt, pop_path = path / "model" / "model.pt", path / "population.npz"
        sha = None
        if pop_path.exists():
            pop = Population.load(pop_path)
            sha = e1.population_fingerprint(pop)
            pops[cid] = (pop, sha)
        elif ckpt.exists():
            raise RuntimeError(f"condition {cid}: {ckpt} exists but {pop_path} is missing")
        if ckpt.exists():
            conds.append(Condition(cid, path, EVALUATED, ckpt, pop_path, sha, "checkpoint present"))
        else:
            why = (
                "no checkpoint yet"
                if pop_path.exists()
                else "directory has no population and no checkpoint"
            )
            conds.append(Condition(cid, path, PENDING, ckpt, pop_path, sha, why))
    if not pops:
        raise RuntimeError("no condition has a population.npz; nothing to evaluate against")
    hashes = {cid: sha for cid, (_, sha) in pops.items()}
    if len(set(hashes.values())) != 1:
        raise RuntimeError(
            "conditions do not share one population (content_sha256 differs): "
            + ", ".join(f"{cid}={sha[:12]}" for cid, sha in hashes.items())
        )
    fp = next(iter(hashes.values()))
    if cfg.data is not None:
        if not (cfg.data / "meta.json").exists():
            raise FileNotFoundError(f"--data {cfg.data} has no meta.json")
        have = (read_meta(cfg.data).get("population") or {}).get("content_sha256")
        if have != fp:
            raise RuntimeError(
                f"--data {cfg.data} was generated from a different population "
                f"(content_sha256 {have!r} != {fp!r})"
            )
    pop = next(iter(pops.values()))[0]
    return pop, fp, conds


# ---------------------------------------------------------------------- evaluation
def build_reference_agents(
    game: KuhnPoker, pop: Population, *, cfr_iterations: int
) -> dict[str, Agent]:
    """The checkpoint-independent exact references (no theta: population / prior only)."""
    from exsolver.agents import (
        BayesBRAgent,
        EquilibriumAgent,
        PluralityBRAgent,
        RandomAgent,
        ThompsonAgent,
    )

    agents: list[Agent] = [
        EquilibriumAgent(game, iterations=cfr_iterations, name=AGENT_NAMES["equilibrium"]),
        ThompsonAgent(game, pop, name=AGENT_NAMES["thompson"]),
        BayesBRAgent(game, pop, name=AGENT_NAMES["bayes"]),
        PluralityBRAgent(game, pop, name=AGENT_NAMES["plurality"]),
        RandomAgent(game, name=AGENT_NAMES["random"]),
    ]
    return {a.name: a for a in agents}


def build_transformer_agents(game: KuhnPoker, policy: Any, tokenizer: Any) -> dict[str, Agent]:
    from exsolver.agents import TransformerAgent

    return {
        AGENT_NAMES["transformer"].format(mode=mode): TransformerAgent(
            game, policy, tokenizer, mode=mode, name=AGENT_NAMES["transformer"].format(mode=mode)
        )
        for mode in e1.TRANSFORMER_MODES
    }


def _evaluate(
    game: KuhnPoker, pop: Population, agents: dict[str, Agent], cfg: E2Config, *, with_oracle: bool
) -> list[SessionMetrics]:
    from exsolver.agents import OracleBRAgent
    from exsolver.eval.reference import BRActionReference

    def oracle_factory(opp_profile):  # only the oracle receives the opponent profile
        return OracleBRAgent(game, opp_profile, name=AGENT_NAMES["oracle"])

    return e1.evaluate_population(
        game,
        pop,
        agents,
        oracle_factory=oracle_factory if with_oracle else None,
        posterior_factory=e1.build_posterior_factory(game, pop),
        H=cfg.hands,
        sessions_per_opp=cfg.eval_sessions_per_opp,
        seed=cfg.eval_seed,
        progress=cfg.progress,
        reference=BRActionReference(game, pop),
    )


def evaluate_references(game: KuhnPoker, pop: Population, cfg: E2Config) -> list[SessionMetrics]:
    """Equilibrium, Thompson, BayesBR, PluralityBR, Random and the per-opponent OracleBR, once."""
    agents = build_reference_agents(game, pop, cfr_iterations=cfg.cfr_iterations)
    return _evaluate(game, pop, agents, cfg, with_oracle=True)


def evaluate_condition(
    game: KuhnPoker, pop: Population, cond: Condition, cfg: E2Config
) -> tuple[list[SessionMetrics], dict[str, Any], dict[str, Any]]:
    """Transformer(sample) / (argmax) of one condition; returns ``(metrics, training, run_info)``."""
    from exsolver.model.inference import Policy
    from exsolver.train import load_checkpoint, resolve_device, tokenizer_from_checkpoint

    device = resolve_device(cfg.device)
    model, model_cfg, meta = load_checkpoint(cond.checkpoint, device)
    fp = e1.check_checkpoint(meta, model_cfg, pop)
    training = e1.training_provenance(meta)
    policy = Policy(model, tokenizer_from_checkpoint(meta), device)
    agents = build_transformer_agents(game, policy, policy.tokenizer)
    t0 = time.perf_counter()
    metrics = _evaluate(game, pop, agents, cfg, with_oracle=False)
    run_info = {
        "condition": cond.id,
        "checkpoint": str(cond.checkpoint),
        "checkpoint_step": meta.get("step"),
        "population_content_sha256": fp,
        "device": str(device),
        "eval_seconds": f"{time.perf_counter() - t0:.1f}",
    }
    return metrics, training, run_info


def condition_aggregate(
    refs: Sequence[SessionMetrics],
    cond_metrics: Sequence[SessionMetrics],
    pop: Population,
    cfg: E2Config,
    game: KuhnPoker,
    agents_for_sanity: dict[str, Agent] | None = None,
) -> dict[str, Any]:
    """One condition's aggregate with plain agent names: E1 criteria plus the E2-1 checks."""
    metrics = [*refs, *cond_metrics]
    agg = aggregate(metrics, reference=AGENT_NAMES["oracle"], archetype_names=pop.archetype_names)
    agg["n_population"] = len(pop)
    agg["criteria_e2"] = check_e2_criteria(
        group_by_agent(metrics), agg["H"], thresholds=cfg.thresholds
    )
    if agents_for_sanity:
        eq_name = agg["criteria"]["resolved_agents"].get("equilibrium")
        if eq_name in agents_for_sanity:
            agg["criteria"]["sanity_equilibrium_vs_nash"] = e1.equilibrium_sanity(
                game, agents_for_sanity[eq_name], H=min(cfg.hands, 8)
            )
    return agg


def combined_aggregate(
    refs: Sequence[SessionMetrics],
    per_condition: dict[str, Sequence[SessionMetrics]],
    conditions: Sequence[Condition],
    pop: Population,
    cfg: E2Config,
) -> dict[str, Any]:
    """Aggregate over references + condition-tagged transformers with the E2 criteria."""
    tagged: list[SessionMetrics] = list(refs)
    for cid, ms in per_condition.items():
        tagged += [replace(m, agent=tag_condition(m.agent, cid)) for m in ms]
    agg = aggregate(tagged, reference=AGENT_NAMES["oracle"], archetype_names=pop.archetype_names)
    agg["experiment"] = "E2"
    agg["n_population"] = len(pop)
    agg["conditions"] = {c.id: c.status for c in conditions}
    agg["criteria_e2"] = check_e2_criteria(
        group_by_agent(tagged),
        agg["H"],
        thresholds=cfg.thresholds,
        conditions=agg["conditions"],
        roles=cfg.roles,
    )
    return agg


# ---------------------------------------------------------------------- summary tables
def _share(ev: float | None, eq: float | None, oracle: float | None) -> float | None:
    if ev is None or eq is None or oracle is None or not (oracle - eq) > 0:
        return None
    return (ev - eq) / (oracle - eq)


def _mean(entry: dict[str, Any] | None) -> float | None:
    if not entry:
        return None
    v = entry.get("mean")
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def condition_rows(agg: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-agent numbers for the per-condition table (references first, then conditions)."""
    per = agg["per_agent"]
    eq = _mean(per.get(AGENT_NAMES["equilibrium"], {}).get("summary", {}).get("ev_all"))
    eq16 = _mean(per.get(AGENT_NAMES["equilibrium"], {}).get("summary", {}).get("ev_last16"))
    orc = _mean(per.get(AGENT_NAMES["oracle"], {}).get("summary", {}).get("ev_all"))
    orc16 = _mean(per.get(AGENT_NAMES["oracle"], {}).get("summary", {}).get("ev_last16"))
    rows = []
    for name in agg["agents"]:
        s = per[name]["summary"]
        klp = s.get("kl_policy_at") or {}
        kl_at = s.get("kl_at") or {}
        rows.append(
            {
                "agent": name,
                "ev_all": _mean(s["ev_all"]),
                "ev_all_se": s["ev_all"].get("se"),
                "ev_last16": _mean(s["ev_last16"]),
                "ev_last16_se": s["ev_last16"].get("se"),
                "share_all": _share(_mean(s["ev_all"]), eq, orc),
                "share_last16": _share(_mean(s["ev_last16"]), eq16, orc16),
                "kl_policy_at": {t: _mean(v) for t, v in klp.items()},
                "kl_identity_final": _mean(kl_at.get(str(agg["H"] - 1)))
                if kl_at
                else _mean(s["kl_final"]),
            }
        )
    return rows


def _f(v: Any, digits: int = 3, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"


def condition_table_md(agg: dict[str, Any]) -> str:
    rows = condition_rows(agg)
    ts = list((rows[0]["kl_policy_at"] if rows else {}).keys())
    header = [
        "agent",
        "EV all hands",
        "EV last 16",
        "share of exploitable value all / last 16",
        *[f"policy KL t={t}" for t in ts],
        f"identity KL t={agg['H'] - 1}",
    ]
    body = []
    for r in rows:
        body.append(
            [
                r["agent"],
                _f(r["ev_all"], 4, True) + (f" ± {r['ev_all_se']:.4f}" if r["ev_all_se"] else ""),
                _f(r["ev_last16"], 4, True)
                + (f" ± {r['ev_last16_se']:.4f}" if r["ev_last16_se"] else ""),
                f"{_f(r['share_all'], 2)} / {_f(r['share_last16'], 2)}",
                *[_f(r["kl_policy_at"].get(t)) for t in ts],
                _f(r["kl_identity_final"]),
            ]
        )
    return md_table(header, body)


def conditions_table_md(agg: dict[str, Any]) -> str:
    header = [
        "condition",
        "status",
        "directory",
        "checkpoint step",
        "steps × batch",
        "λ_opp",
        "data_dir",
        "reason",
    ]
    rows = []
    for cid, info in agg.get("per_condition", {}).items():
        tr = info.get("training") or {}
        rows.append(
            [
                cid,
                info["status"],
                str(info["path"]),
                str(tr.get("checkpoint_step", "–")),
                f"{tr.get('steps', '–')} × {tr.get('batch_size', '–')}" if tr else "–",
                str(tr.get("lambda_opp", "–")),
                str(tr.get("data_dir", "–")),
                info.get("reason", ""),
            ]
        )
    return md_table(header, rows)


def render_combined_md(agg: dict[str, Any], run_info: dict[str, Any]) -> str:
    n_per = agg.get("n_sessions_per_agent", {})
    n = max(n_per.values()) if n_per else agg.get("n_sessions", 0)
    parts = [
        "# E2 — summary",
        "",
        f"{n} sessions per agent = {agg['n_opponents']} opponents × {agg.get('n_sessions_per_opponent', '?')} "
        f"sessions, H = {agg['H']} hands per session. Exact references (Equilibrium, Thompson, BayesBR, "
        "PluralityBR, OracleBR, Random) evaluated once; every condition adds Transformer(sample) and "
        "Transformer(argmax) tagged `[condition]`. All EV / KL numbers are exact (game-tree traversal of "
        "the hand-t strategy); ± is one standard error across sessions; comparisons are paired (same "
        "opponent, same deals).",
        "",
        "## Conditions",
        "",
        conditions_table_md(agg),
        "",
        "## E2 criteria (docs/experiments/e2-decision-relevant-inference.md)",
        "",
        criteria_table(agg, "criteria_e2"),
        "",
        "## Agents and conditions",
        "",
        '"share of exploitable value" = (EV − EV_Equilibrium) / (EV_OracleBR − EV_Equilibrium) on the same hands.',
        "",
        condition_table_md(agg),
        "",
    ]
    for cid, info in agg.get("per_condition", {}).items():
        if info.get("training"):
            parts += [
                f"## Training configuration — condition {cid} (from the checkpoint)",
                "",
                training_table({"training": info["training"]}),
                "",
            ]
    parts += ["## Notes", ""] + [f"- {x}" for x in notes(agg)]
    pending = [
        cid for cid, info in agg.get("per_condition", {}).items() if info["status"] != EVALUATED
    ]
    if pending:
        parts.append(
            f"- Pending conditions ({', '.join(pending)}) have no checkpoint yet; re-run `eval` once the "
            "training queue has written them."
        )
    parts += [
        "",
        "## Figures",
        "",
        ", ".join(f"`{f}`" for f in COMBINED_FIGURES)
        + " (regenerable from `summary.json` with the `plots` stage); "
        "per-condition figures under `<out>/<condition>/`.",
        "",
        "## Run",
        "",
    ]
    parts += [f"- **{k}**: {v}" for k, v in run_info.items()]
    parts.append("")
    return "\n".join(parts)


def write_combined_outputs(agg: dict[str, Any], out: Path, run_info: dict[str, Any]) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    e1.dump_json(agg, out / "summary.json")
    paths = [fn(agg, out / name) for name, fn in COMBINED_FIGURES.items()]
    (out / "summary.md").write_text(render_combined_md(agg, run_info))
    paths += [out / "summary.md", out / "summary.json"]
    return paths


# ---------------------------------------------------------------------- stages
def build_smoke_conditions(cfg: E2Config) -> dict[str, Path]:
    """Two tiny conditions under ``--out`` (E1 stages: population, gen, train) sharing one dataset.

    ``A`` uses the E1 smoke settings (M = 16, 200 sessions, H = 8) with a 1 x 32 model for
    ``SMOKE_CONDITION_STEPS["A"]`` steps; ``B`` re-samples the same population (same seed and M,
    hence the same content hash) and trains on ``A``'s dataset for fewer steps.
    """
    base = e1.apply_smoke(
        e1.E1Config(out=cfg.condition_out("A"), device=cfg.device, progress=False)
    )
    base = replace(
        base,
        data=base.out / "data",
        workers=1,
        steps=SMOKE_CONDITION_STEPS["A"],
        d_model=SMOKE_MODEL["d_model"],
        n_layers=SMOKE_MODEL["n_layers"],
        n_heads=SMOKE_MODEL["n_heads"],
    )
    e1.stage_population(base)
    e1.stage_gen(base)
    e1.stage_train(base)
    # B reads A's dataset, so it cannot carry E1's --smoke flag (whose guard wants <out>/data)
    other = replace(base, smoke=False, out=cfg.condition_out("B"), steps=SMOKE_CONDITION_STEPS["B"])
    e1.stage_population(other)
    e1.stage_train(other)
    return {"A": base.out, "B": other.out}


def stage_eval(cfg: E2Config) -> dict[str, Any]:
    t0 = time.perf_counter()
    if cfg.smoke and not cfg.conditions:
        _log("smoke: building two tiny conditions with the E1 stages ...")
        cfg = replace(cfg, conditions=build_smoke_conditions(cfg))
    game = KuhnPoker()
    pop, fp, conds = inspect_conditions(cfg)
    ready = [c for c in conds if c.status == EVALUATED]
    pending = [c for c in conds if c.status == PENDING]
    _log(
        f"eval: population M={len(pop)} sha256 {fp[:12]}; conditions ready {[c.id for c in ready]}, "
        f"pending {[c.id for c in pending]}; {cfg.eval_sessions_per_opp} sessions x H={cfg.hands}"
    )
    t_ref = time.perf_counter()
    ref_agents = build_reference_agents(game, pop, cfr_iterations=cfg.cfr_iterations)
    refs = evaluate_references(game, pop, cfg)
    ref_seconds = time.perf_counter() - t_ref
    cfg.out.mkdir(parents=True, exist_ok=True)
    e1.save_sessions(refs, cfg.out / "references_sessions.npz")
    _log(f"eval: {len(refs)} reference sessions in {ref_seconds:.0f}s")

    per_condition: dict[str, list[SessionMetrics]] = {}
    per_condition_info: dict[str, dict[str, Any]] = {}
    for cond in conds:
        info: dict[str, Any] = {**cond.to_dict(), "training": None}
        if cond.status != EVALUATED:
            _log(f"eval: condition {cond.id} pending ({cond.reason})")
            per_condition_info[cond.id] = info
            continue
        metrics, training, run_info = evaluate_condition(game, pop, cond, cfg)
        per_condition[cond.id] = metrics
        agg_c = condition_aggregate(refs, metrics, pop, cfg, game, ref_agents)
        agg_c["experiment"] = f"E2 condition {cond.id}"
        agg_c["training"] = training
        agg_c["condition"] = cond.to_dict()
        run_c = {"command": command_line(), **run_info}
        agg_c["run"] = run_c
        out_c = cfg.condition_out(cond.id)
        paths = e1.write_eval_outputs(agg_c, out_c, run_info=run_c)
        e1.save_sessions(metrics, out_c / "sessions.npz")
        info.update({"training": training, "run": run_info, "outputs": [str(p) for p in paths]})
        info["criteria_e2"] = {
            k: v.get("pass")
            for k, v in agg_c["criteria_e2"].items()
            if isinstance(v, dict) and "pass" in v
        }
        per_condition_info[cond.id] = info
        _log(
            f"eval: condition {cond.id}: {len(metrics)} sessions in {run_info['eval_seconds']}s; E2-1 {info['criteria_e2']}"
        )

    agg = combined_aggregate(refs, per_condition, conds, pop, cfg)
    agg["per_condition"] = per_condition_info
    run_info = {
        "command": command_line(),
        "config": json.dumps(cfg.to_dict(), default=str),
        "population_content_sha256": fp,
        "data": None if cfg.data is None else str(cfg.data),
        "reference_seconds": f"{ref_seconds:.1f}",
        "references_sessions_file": str(cfg.out / "references_sessions.npz"),
    }
    agg["run"] = run_info
    paths = write_combined_outputs(agg, cfg.out, run_info)
    verdicts = {
        k: v.get("pass")
        for k, v in agg["criteria_e2"].items()
        if isinstance(v, dict) and "pass" in v
    }
    _log(f"eval: E2 criteria {verdicts}; wrote {[p.name for p in paths]}")
    record_command(
        cfg,
        "eval",
        {
            "population_content_sha256": fp,
            "conditions": {c.id: c.to_dict() for c in conds},
            "criteria_e2": verdicts,
            "outputs": [str(p) for p in paths],
            "elapsed_s": time.perf_counter() - t0,
        },
    )
    return agg


def stage_plots(cfg: E2Config) -> list[Path]:
    """Re-draw the combined figures and ``summary.md`` from ``<out>/summary.json``."""
    t0 = time.perf_counter()
    path = cfg.out / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"no {path}; run the eval stage first")
    agg = json.loads(path.read_text())
    paths = write_combined_outputs(agg, cfg.out, agg.get("run") or {})
    for cid in agg.get("per_condition", {}):
        if (cfg.condition_out(cid) / "summary.json").exists():
            paths += e1.stage_plots(e1.E1Config(out=cfg.condition_out(cid), progress=False))
    record_command(
        cfg,
        "plots",
        {
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
        "--data",
        type=Path,
        default=None,
        help="dataset whose meta.json population hash is cross-checked (e.g. data/e1a)",
    )
    p.add_argument(
        "--conditions",
        nargs="*",
        default=[],
        metavar="ID=PATH",
        help="condition directories (model/model.pt + population.npz)",
    )
    p.add_argument("--hands", type=int, default=E2Config.hands)
    p.add_argument("--eval-sessions-per-opp", type=int, default=E2Config.eval_sessions_per_opp)
    p.add_argument("--eval-seed", type=int, default=E2Config.eval_seed)
    p.add_argument("--device", default=E2Config.device, help="auto | cpu | mps")
    p.add_argument("--cfr-iterations", type=int, default=E2Config.cfr_iterations)
    p.add_argument(
        "--smoke",
        action="store_true",
        help=f"shrink to {SMOKE}; builds two tiny conditions when none are given",
    )
    p.add_argument("--no-progress", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> E2Config:
    out = args.out if args.out is not None else (SMOKE_OUT if args.smoke else DEFAULT_OUT)
    cfg = E2Config(
        out=out,
        data=args.data,
        conditions=parse_conditions(args.conditions),
        hands=args.hands,
        eval_sessions_per_opp=args.eval_sessions_per_opp,
        eval_seed=args.eval_seed,
        device=args.device,
        cfr_iterations=args.cfr_iterations,
        progress=not args.no_progress,
        smoke=args.smoke,
    )
    return apply_smoke(cfg) if args.smoke else cfg


def run_stage(stage: str, cfg: E2Config) -> Any:
    if stage == "eval":
        return stage_eval(cfg)
    if stage == "plots":
        return stage_plots(cfg)
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
