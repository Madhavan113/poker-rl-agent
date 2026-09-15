"""E1 runner: population provenance, CLI / smoke configuration, hash cross-checks, outputs.

Every test writes into ``tmp_path`` only -- never into ``runs/e1`` or ``data/e1a``.
"""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from exsolver.data.shards import write_meta
from exsolver.eval.aggregate import aggregate
from exsolver.eval.session import FixedStrategyAgent, entropy
from exsolver.experiments import e1_kuhn as e1
from exsolver.games import KuhnPoker
from exsolver.population import (
    KuhnPrior,
    Population,
    nash_theta,
    sample_population,
    theta_to_profile,
)
from exsolver.solvers import best_response, cfr_plus
from exsolver.strategy import merge_seats, uniform_strategy


class DummyPosterior:
    def __init__(self, m):
        self.probs = np.full(m, 1.0 / m)
        self.n = 0

    def update(self, hand):
        self.n += 1
        p = np.exp(-0.2 * self.n * np.arange(len(self.probs)))
        self.probs = p / p.sum()

    def entropy(self):
        return entropy(self.probs)


def _reject_constant(name):
    raise ValueError(f"non-strict JSON constant {name}")


def load_strict_json(path: Path):
    return json.loads(path.read_text(), parse_constant=_reject_constant)


def test_population_stage_writes_population_and_provenance_sidecar(tmp_path):
    cfg = e1.E1Config(out=tmp_path, m=8, progress=False)
    pop = e1.stage_population(cfg)
    assert cfg.population_path.exists() and cfg.population_sidecar.exists()
    side = json.loads(cfg.population_sidecar.read_text())
    assert side["seed"] == 0 and side["m"] == 8 and side["prior"] == "KuhnPrior"
    assert side["content_sha256"] == e1.population_fingerprint(pop)
    assert "created_at" in side and side["command"].startswith(
        "uv run python -m exsolver.experiments.e1_kuhn"
    )
    loaded = Population.load(cfg.population_path)
    expected = sample_population(
        KuhnPrior(), 8, np.random.default_rng(0)
    )  # seed 0, as the spec says
    np.testing.assert_array_equal(loaded.thetas, expected.thetas)
    np.testing.assert_array_equal(loaded.thetas, pop.thetas)
    # reuse with a consistent sidecar: skipped, and the README says so
    again = e1.stage_population(cfg)
    np.testing.assert_array_equal(again.thetas, pop.thetas)
    readme = (tmp_path / "README.md").read_text()
    assert (
        readme.count("## population") == 2 and "(skipped)" in readme and '"skipped": true' in readme
    )
    assert readme.startswith(f"# {tmp_path}")
    # seed / M mismatch -> refuse
    with pytest.raises(RuntimeError, match="seed"):
        e1.stage_population(e1.E1Config(out=tmp_path, m=8, seed=1, progress=False))
    with pytest.raises(RuntimeError, match="--m is 9"):
        e1.stage_population(e1.E1Config(out=tmp_path, m=9, progress=False))
    # sidecar missing: recompute for seed / M, reuse when the content matches, rewrite the sidecar
    cfg.population_sidecar.unlink()
    e1.stage_population(cfg)
    assert (
        json.loads(cfg.population_sidecar.read_text())["content_sha256"] == side["content_sha256"]
    )
    # sidecar missing and different content -> refuse
    cfg.population_sidecar.unlink()
    sample_population(KuhnPrior(), 8, np.random.default_rng(123)).save(cfg.population_path)
    with pytest.raises(RuntimeError, match="does not match"):
        e1.stage_population(cfg)
    # tampered sidecar hash -> refuse
    e1.stage_population(e1.E1Config(out=tmp_path / "b", m=4, progress=False))
    bad = json.loads((tmp_path / "b" / "population.json").read_text())
    bad["content_sha256"] = "0" * 64
    (tmp_path / "b" / "population.json").write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="content_sha256"):
        e1.stage_population(e1.E1Config(out=tmp_path / "b", m=4, progress=False))


def test_cli_defaults_smoke_directory_and_refusal(tmp_path, monkeypatch):
    # run inside a scratch tree with a dummy live run so a guard regression can never touch runs/e1
    monkeypatch.chdir(tmp_path)
    live = tmp_path / "runs" / "e1"
    live.mkdir(parents=True)
    (live / "population.npz").write_bytes(b"live run")
    smoke = e1.config_from_args(e1.build_parser().parse_args(["all", "--smoke"]))
    assert smoke.smoke is True
    assert smoke.out == Path("runs/e1_smoke") and smoke.data_dir == Path("runs/e1_smoke") / "data"
    assert (smoke.m, smoke.sessions, smoke.hands, smoke.steps, smoke.eval_sessions_per_opp) == (
        16,
        200,
        8,
        100,
        1,
    )
    custom = e1.config_from_args(e1.build_parser().parse_args(["all", "--smoke", "--out", "x"]))
    assert custom.out == Path("x") and custom.data_dir == Path("x") / "data"
    for bad_out in ("runs/e1", "runs/e1/", "./runs/e1"):
        with pytest.raises(ValueError, match="runs/e1"):
            e1.config_from_args(e1.build_parser().parse_args(["all", "--smoke", "--out", bad_out]))
    with pytest.raises(ValueError):
        e1.E1Config(out="runs/e1", smoke=True)
    with pytest.raises(ValueError):
        e1.apply_smoke(e1.E1Config())
    with pytest.raises(SystemExit):
        e1.main(["all", "--smoke", "--out", "runs/e1"])  # refused before anything is written
    assert sorted(p.name for p in live.iterdir()) == ["population.npz"]  # nothing written
    assert not (tmp_path / "runs" / "e1_smoke").exists()
    # N3: under --smoke the dataset directory must lie inside --out
    with pytest.raises(ValueError, match="--data"):
        e1.config_from_args(e1.build_parser().parse_args(["all", "--smoke", "--data", "data/e1a"]))
    with pytest.raises(ValueError, match="--data"):
        e1.config_from_args(
            e1.build_parser().parse_args(["all", "--smoke", "--out", "x", "--data", "y"])
        )
    inside = e1.config_from_args(
        e1.build_parser().parse_args(["all", "--smoke", "--out", "x", "--data", "x/mydata"])
    )
    assert inside.data_dir == Path("x/mydata")
    with pytest.raises(ValueError):
        e1.E1Config(out="x", data="data/e1a", smoke=True)
    assert e1.E1Config(out="x", data="x/data", smoke=True).data_dir == Path("x/data")
    assert not (tmp_path / "data").exists()

    full = e1.config_from_args(e1.build_parser().parse_args(["eval"]))
    assert (full.m, full.hands, full.sessions) == (256, 64, 100_000)
    assert (full.steps, full.batch) == (12_000, 32)  # deviates from the spec's 20k x 64 (MPS speed)
    assert (e1.SPEC_STEPS, e1.SPEC_BATCH) == (20_000, 64) and "MPS" in e1.TRAINING_DEVIATION
    assert (full.eval_sessions_per_opp, full.seed, full.eval_seed, full.device) == (4, 0, 1, "auto")
    assert full.out == Path("runs/e1") and full.data_dir == Path("data/e1a")
    assert full.smoke is False and full.retrain is False and full.max_train_minutes is None
    assert 1 <= full.workers <= 8
    over = e1.config_from_args(
        e1.build_parser().parse_args(
            ["train", "--data", "d", "--steps", "7", "--batch", "3", "--max-train-minutes", "5.5",
             "--device", "cpu", "--retrain", "--no-progress"]
        )
    )  # fmt: skip
    assert (
        over.data_dir == Path("d") and over.steps == 7 and over.batch == 3 and over.retrain is True
    )
    assert over.max_train_minutes == 5.5 and over.device == "cpu" and over.progress is False
    assert json.dumps(over.to_dict())
    with pytest.raises(SystemExit):
        e1.build_parser().parse_args(["bogus"])
    assert e1.STAGES == ("population", "gen", "train", "eval", "plots", "all")


def test_cap_steps_respects_the_wall_clock_budget():
    steps, rate = e1.cap_steps(20_000, 20, 10.0, 60.0)
    assert rate == pytest.approx(2.0)
    assert steps == int(2.0 * (3600 - 10))
    assert e1.cap_steps(100, 20, 1.0, 60.0)[0] == 100  # budget not binding
    assert e1.cap_steps(20_000, 20, 100.0, 1.0)[0] == 20  # probe alone exceeded the budget
    assert e1.cap_steps(20_000, 20, 14.3, 60.0)[0] == pytest.approx(
        20 / 14.3 * (3600 - 14.3), abs=1
    )


def test_strict_json_replaces_non_finite_values():
    obj = {"a": float("nan"), "b": [1.0, float("inf"), -float("inf")], "c": np.float64(2.5), "d": np.int64(3),
           "e": np.array([1.0, np.nan]), "f": np.bool_(True), "g": Path("x")}  # fmt: skip
    out = e1.strict_json(obj)
    assert out == {
        "a": None,
        "b": [1.0, None, None],
        "c": 2.5,
        "d": 3,
        "e": [1.0, None],
        "f": True,
        "g": "x",
    }
    assert json.dumps(out, allow_nan=False)


def test_dataset_status_cross_checks_the_population_hash(tmp_path):
    cfg = e1.E1Config(
        out=tmp_path / "run", data=tmp_path / "data", m=4, hands=6, sessions=10, progress=False
    )
    pop = e1.stage_population(cfg)
    assert e1.dataset_status(cfg, pop) == "missing"
    with pytest.raises(FileNotFoundError):
        e1.stage_train(cfg)
    cfg.data_dir.mkdir()
    np.savez(cfg.data_dir / "shard_00000.npz", opp_id=np.zeros(10, np.int32))  # placeholder shard
    good = {
        "H": 6,
        "n_sessions": 10,
        "L": 64,
        "population": {"content_sha256": e1.population_fingerprint(pop)},
    }
    write_meta(cfg.data_dir, good)
    assert e1.dataset_status(cfg, pop) == "match"
    write_meta(cfg.data_dir, {**good, "population": {"content_sha256": "0" * 64}})  # tampered hash
    assert e1.dataset_status(cfg, pop) == "population_mismatch"
    with pytest.raises(RuntimeError, match="different population"):
        e1.stage_train(cfg)
    with pytest.raises(RuntimeError, match="different population"):
        e1.stage_gen(cfg)
    write_meta(cfg.data_dir, {**good, "H": 7})
    assert e1.dataset_status(cfg, pop) == "mismatch"
    with pytest.raises(RuntimeError, match="different H"):
        e1.stage_gen(cfg)
    write_meta(cfg.data_dir, {"H": 6, "n_sessions": 10})  # no population record at all
    assert e1.dataset_status(cfg, pop) == "population_mismatch"


def test_stage_eval_refuses_checkpoints_from_another_population(tmp_path):
    from exsolver.data.tokenizer import TokenizerSpec
    from exsolver.model.transformer import ExploitTransformer, ModelConfig
    from exsolver.train import TrainConfig, load_checkpoint, save_checkpoint

    cfg = e1.E1Config(out=tmp_path, m=3, hands=4, device="cpu", progress=False)
    pop = e1.stage_population(cfg)
    fp = e1.population_fingerprint(pop)

    def save(n_opp, content_hash):
        mc = ModelConfig(vocab_size=27, n_opp=n_opp, d_model=8, n_layers=1, n_heads=1, max_len=64)
        data_meta = {
            "population": {"content_sha256": content_hash},
            "n_sessions": 9,
            "H": 4,
            "L": 64,
        }
        tc = TrainConfig("some/data", "o", steps=5, batch_size=2, lr=1e-3, device="cpu")
        save_checkpoint(
            cfg.checkpoint, ExploitTransformer(mc), mc, tc, TokenizerSpec(3, 2), data_meta, step=5
        )

    save(3, "f" * 64)
    with pytest.raises(RuntimeError, match="different population"):
        e1.stage_eval(cfg)
    save(4, fp)
    with pytest.raises(RuntimeError, match="opponent head"):
        e1.stage_eval(cfg)
    save(3, fp)
    _, mc, meta = load_checkpoint(cfg.checkpoint, "cpu")
    assert e1.check_checkpoint(meta, mc, pop) == fp
    prov = e1.training_provenance(meta)
    assert (
        prov["steps"] == 5
        and prov["batch_size"] == 2
        and prov["lr"] == 1e-3
        and prov["device"] == "cpu"
    )
    assert prov["data_dir"] == "some/data" and prov["population_content_sha256"] == fp
    assert (
        prov["checkpoint_step"] == 5
        and prov["data"]["n_sessions"] == 9
        and prov["model"]["d_model"] == 8
    )
    assert "batch 2 × 5 steps (spec: 64 × 20000)" in prov["spec_deviation"]
    assert (
        "MPS" in prov["spec_deviation"]
        and prov["spec_batch_size"] == 64
        and prov["spec_steps"] == 20_000
    )
    meta["train_config"]["batch_size"], meta["train_config"]["steps"] = 64, 20_000
    assert e1.training_provenance(meta)["spec_deviation"] is None


def test_evaluate_population_and_outputs(tmp_path):
    game = KuhnPoker()
    pop = sample_population(KuhnPrior(), 3, np.random.default_rng(2))
    eq = cfr_plus(game, 500)
    agents = {
        "Equilibrium": FixedStrategyAgent(eq, "Equilibrium"),
        "Random": FixedStrategyAgent(uniform_strategy(game), "Random"),
    }

    def oracle_factory(profile):
        br = merge_seats(best_response(game, profile, 0)[0], best_response(game, profile, 1)[0])
        return FixedStrategyAgent(br, "OracleBR")

    metrics = e1.evaluate_population(
        game, pop, agents, oracle_factory=oracle_factory, posterior_factory=lambda: DummyPosterior(3),
        H=6, sessions_per_opp=2, seed=1,
    )  # fmt: skip
    assert len(metrics) == 3 * 2 * 3
    assert {m.agent for m in metrics} == {"Equilibrium", "Random", "OracleBR"}
    assert all(m.H == 6 and not np.isnan(m.post_entropy).any() for m in metrics)
    assert len({(m.agent, m.opp_id, m.session) for m in metrics}) == len(metrics)

    agg = aggregate(metrics, reference="OracleBR", archetype_names=pop.archetype_names)
    agg["n_population"] = len(pop)
    agg["training"] = {"steps": 12, "batch_size": 32, "spec_deviation": e1.TRAINING_DEVIATION}
    out = tmp_path / "out"
    paths = e1.write_eval_outputs(agg, out, run_info={"command": "pytest"})
    names = {p.name for p in paths}
    assert {"summary.json", "summary.md", "ev_vs_hand.png", "entropy_vs_hand.png", "exploitability_vs_hand.png",
            "regret_cumulative.png", "probes.png"} <= names  # fmt: skip
    text = (out / "summary.json").read_text()
    assert "NaN" not in text and "Infinity" not in text  # strict JSON: undefined values are null
    loaded = load_strict_json(out / "summary.json")
    assert (
        loaded["agents"] == agg["agents"]
        and loaded["criteria"]["sanity_oracle_dominates"]["pass"] is True
    )
    assert (
        loaded["per_agent"]["Random"]["metrics"]["kl"]["mean"][0] is None
    )  # fixed agents have no belief
    md = (out / "summary.md").read_text()
    assert "before" in md and "last hand" in md and "1e-30" in md and "69.08" in md
    assert "## Training configuration" in md and "MPS" in md and "| steps | 12 |" in md
    assert "This run used batch 32 × 12 steps (spec: 64 × 20000)" in md

    e1.save_sessions(metrics, tmp_path / "sessions.npz")
    with np.load(tmp_path / "sessions.npz") as d:
        assert d["ev"].shape == (18, 6) and d["probe_bluff"].shape == (18, 6)
        assert set(d["agent"].tolist()) == {"Equilibrium", "Random", "OracleBR"}

    (out / "ev_vs_hand.png").unlink()
    e1.stage_plots(
        e1.E1Config(out=out, progress=False)
    )  # re-drawn from the strict summary.json alone
    assert (out / "ev_vs_hand.png").exists()
    assert "## plots" in (out / "README.md").read_text()

    sanity = e1.equilibrium_sanity(
        game, FixedStrategyAgent(theta_to_profile(nash_theta(0.1)), "Equilibrium")
    )
    assert (
        sanity["pass"] is True
        and math.isclose(sanity["ev_seat0"], -1 / 18)
        and math.isclose(sanity["ev_seat1"], 1 / 18)
    )
    assert (
        e1.equilibrium_sanity(game, FixedStrategyAgent(uniform_strategy(game), "Uniform"))["pass"]
        is False
    )
    with pytest.raises(FileNotFoundError):
        e1.stage_plots(e1.E1Config(out=tmp_path / "nowhere"))
    with pytest.raises(FileNotFoundError):
        e1.stage_eval(e1.E1Config(out=tmp_path / "nockpt", m=3, progress=False))


def test_tiny_pipeline_end_to_end_with_provenance(tmp_path):
    """population -> gen -> train (2 steps, 16-dim model) -> eval on 3 opponents, all in tmp_path."""
    pytest.importorskip("exsolver.data.generate")
    pytest.importorskip("exsolver.agents")
    pytest.importorskip("exsolver.bayes")
    cfg = e1.E1Config(
        out=tmp_path / "run", data=tmp_path / "data", m=3, hands=4, sessions=12, steps=2, batch=4,
        eval_sessions_per_opp=1, device="cpu", workers=1, d_model=16, n_layers=1, n_heads=2,
        progress=False, extra_train={"eval_size": 0},
    )  # fmt: skip
    agg = e1.run_stage("all", cfg)
    pop = Population.load(cfg.population_path)
    fp = e1.population_fingerprint(pop)
    assert e1.dataset_status(cfg, pop) == "match"
    assert (
        cfg.checkpoint.exists()
        and (cfg.out / "summary.json").exists()
        and (cfg.out / "sessions.npz").exists()
    )
    assert agg["training"]["steps"] == 2 and agg["training"]["batch_size"] == 4
    assert (
        agg["training"]["population_content_sha256"]
        == fp
        == agg["run"]["population_content_sha256"]
    )
    assert agg["n_population"] == 3 and agg["H"] == 4
    assert set(agg["agents"]) == {
        "Equilibrium",
        "Thompson",
        "BayesBR",
        "Transformer(sample)",
        "Transformer(argmax)",
        "Random",
        "OracleBR",
    }
    assert agg["criteria"]["sanity_oracle_dominates"]["pass"] is True
    loaded = load_strict_json(cfg.out / "summary.json")
    assert loaded["training"]["steps"] == 2
    readme = (cfg.out / "README.md").read_text()
    for stage in ("population", "gen", "train", "eval"):
        assert f"## {stage}" in readme
    md = (cfg.out / "summary.md").read_text()
    assert "## Training configuration" in md and fp in md
    # re-running skips gen and train and says so
    e1.stage_gen(cfg)
    e1.stage_train(cfg)
    readme = (cfg.out / "README.md").read_text()
    assert "## gen" in readme and readme.count("(skipped)") >= 2
    # a tampered dataset meta is refused by train (even with --retrain)
    meta_path = cfg.data_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["population"]["content_sha256"] = "0" * 64
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="different population"):
        e1.stage_train(replace(cfg, retrain=True))
