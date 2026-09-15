"""E2 runner: condition parsing and guards, hash checks, pending conditions, tiny end-to-end run.

Every test writes into ``tmp_path`` only -- never into ``runs/e1``, ``runs/e2`` or ``data/e1a``.
"""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from exsolver.data.shards import write_meta
from exsolver.experiments import e1_kuhn as e1
from exsolver.experiments import e2
from exsolver.population import KuhnPrior, Population, sample_population


def _reject_constant(name):
    raise ValueError(f"non-strict JSON constant {name}")


def load_strict_json(path: Path):
    return json.loads(path.read_text(), parse_constant=_reject_constant)


def test_parse_conditions_and_config_guards(tmp_path, monkeypatch):
    assert e2.parse_conditions(["A=runs/e1", "B=runs/e2/B"]) == {
        "A": Path("runs/e1"),
        "B": Path("runs/e2/B"),
    }
    for bad in (["A"], ["=x"], ["A="], ["A=x", "A=y"]):
        with pytest.raises(ValueError):
            e2.parse_conditions(bad)
    with pytest.raises(ValueError, match="condition id"):
        e2.E2Config(conditions={"A[1]": "x"})
    # CLI defaults
    full = e2.config_from_args(
        e2.build_parser().parse_args(["eval", "--conditions", "A=runs/e1", "B=runs/e2/B"])
    )
    assert full.out == Path("runs/e2") and full.data is None and full.smoke is False
    assert (full.hands, full.eval_sessions_per_opp, full.eval_seed, full.device) == (
        64,
        4,
        1,
        "auto",
    )
    assert list(full.conditions) == ["A", "B"] and full.roles.baseline == "A"
    assert full.thresholds.kl_policy == 0.05 and full.thresholds.match_gap == 0.01
    assert json.dumps(full.to_dict())
    # --smoke: default out, shrunk evaluation, refusal to write into protected directories
    monkeypatch.chdir(tmp_path)
    smoke = e2.config_from_args(e2.build_parser().parse_args(["eval", "--smoke"]))
    assert smoke.out == Path("runs/e2_smoke") and smoke.smoke is True
    assert (smoke.hands, smoke.eval_sessions_per_opp) == (8, 1) and smoke.conditions == {}
    for bad_out in ("runs/e1", "runs/e2", "runs/e2/x", "data/e1a", "./runs/e2/"):
        with pytest.raises(ValueError, match="--smoke"):
            e2.config_from_args(e2.build_parser().parse_args(["eval", "--smoke", "--out", bad_out]))
    with pytest.raises(ValueError, match="--data"):
        e2.E2Config(out="x", data="data/e1a", smoke=True)
    assert e2.E2Config(out="x", data="x/A/data", smoke=True).data == Path("x/A/data")
    with pytest.raises(SystemExit):
        e2.main(["eval", "--smoke", "--out", "runs/e2"])
    assert not (tmp_path / "runs").exists()  # nothing written
    with pytest.raises(SystemExit):
        e2.build_parser().parse_args(["train"])
    with pytest.raises(ValueError, match="no conditions"):
        e2.stage_eval(e2.E2Config(out=tmp_path / "o"))
    with pytest.raises(FileNotFoundError):
        e2.stage_plots(e2.E2Config(out=tmp_path / "nowhere"))


def _fake_checkpoint(path: Path, n_opp: int, content_hash: str, steps: int = 5) -> None:
    from exsolver.data.tokenizer import TokenizerSpec
    from exsolver.model.transformer import ExploitTransformer, ModelConfig
    from exsolver.train import TrainConfig, save_checkpoint

    mc = ModelConfig(vocab_size=27, n_opp=n_opp, d_model=8, n_layers=1, n_heads=1, max_len=64)
    data_meta = {"population": {"content_sha256": content_hash}, "n_sessions": 9, "H": 4, "L": 64}
    tc = TrainConfig("some/data", "o", steps=steps, batch_size=2, lr=1e-3, device="cpu")
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        path, ExploitTransformer(mc), mc, tc, TokenizerSpec(3, 2), data_meta, step=steps
    )


def test_inspect_conditions_pending_and_hash_mismatch(tmp_path):
    pop = sample_population(KuhnPrior(), 3, np.random.default_rng(0))
    fp = e1.population_fingerprint(pop)
    other = sample_population(KuhnPrior(), 3, np.random.default_rng(7))
    a, b, c, d = (tmp_path / x for x in "ABCD")
    for path in (a, b):
        path.mkdir()
        pop.save(path / "population.npz")
    _fake_checkpoint(a / "model" / "model.pt", 3, fp)
    cfg = e2.E2Config(out=tmp_path / "out", conditions={"A": a, "B": b, "C": c})
    loaded, sha, conds = e2.inspect_conditions(cfg)
    assert sha == fp and len(loaded) == 3
    assert [(x.id, x.status) for x in conds] == [
        ("A", "evaluated"),
        ("B", "pending"),
        ("C", "pending"),
    ]
    assert "no checkpoint" in conds[1].reason and "no population" in conds[2].reason
    assert conds[0].population_sha256 == fp and conds[2].population_sha256 is None
    # a different population in one condition is refused
    d.mkdir()
    other.save(d / "population.npz")
    with pytest.raises(RuntimeError, match="do not share one population"):
        e2.inspect_conditions(replace(cfg, conditions={"A": a, "D": d}))
    # a checkpoint without its population is a broken condition
    e_dir = tmp_path / "E"
    _fake_checkpoint(e_dir / "model" / "model.pt", 3, fp)
    with pytest.raises(RuntimeError, match="population.npz is missing"):
        e2.inspect_conditions(replace(cfg, conditions={"A": a, "E": e_dir}))
    # only pending conditions: nothing to evaluate against
    with pytest.raises(RuntimeError, match="no condition has a population"):
        e2.inspect_conditions(replace(cfg, conditions={"C": c}))
    # --data must carry the same population hash
    data = tmp_path / "data"
    data.mkdir()
    write_meta(data, {"H": 4, "n_sessions": 9, "population": {"content_sha256": fp}})
    assert e2.inspect_conditions(replace(cfg, data=data))[1] == fp
    write_meta(data, {"H": 4, "n_sessions": 9, "population": {"content_sha256": "0" * 64}})
    with pytest.raises(RuntimeError, match="different population"):
        e2.inspect_conditions(replace(cfg, data=data))
    with pytest.raises(FileNotFoundError):
        e2.inspect_conditions(replace(cfg, data=tmp_path / "nodata"))
    # a checkpoint from another population / with the wrong head is refused at evaluation
    _fake_checkpoint(a / "model" / "model.pt", 3, "f" * 64)
    with pytest.raises(RuntimeError, match="different population"):
        e2.stage_eval(replace(cfg, hands=2, eval_sessions_per_opp=1, device="cpu", progress=False))
    assert not (tmp_path / "out" / "summary.json").exists()
    _fake_checkpoint(a / "model" / "model.pt", 4, fp)
    with pytest.raises(RuntimeError, match="opponent head"):
        e2.stage_eval(replace(cfg, hands=2, eval_sessions_per_opp=1, device="cpu", progress=False))


@pytest.fixture(scope="module")
def tiny_conditions(tmp_path_factory):
    """Two 2- / 3-step checkpoints on one tiny population and dataset, plus a pending third."""
    pytest.importorskip("exsolver.data.generate")
    root = tmp_path_factory.mktemp("conds")
    base = e1.E1Config(
        out=root / "A", data=root / "data", m=3, hands=4, sessions=12, steps=2, batch=4,
        device="cpu", workers=1, d_model=16, n_layers=1, n_heads=2, progress=False,
        extra_train={"eval_size": 0},
    )  # fmt: skip
    e1.stage_population(base)
    e1.stage_gen(base)
    e1.stage_train(base)
    other = replace(base, out=root / "B", steps=3)
    e1.stage_population(other)
    e1.stage_train(other)
    return root, {"A": root / "A", "B": root / "B", "C": root / "C"}


def test_e2_end_to_end_on_tiny_conditions(tiny_conditions, tmp_path):
    root, conds = tiny_conditions
    cfg = e2.E2Config(
        out=tmp_path / "e2", data=root / "data", conditions=conds, hands=4,
        eval_sessions_per_opp=1, device="cpu", progress=False,
    )  # fmt: skip
    agg = e2.stage_eval(cfg)
    pop = Population.load(conds["A"] / "population.npz")
    fp = e1.population_fingerprint(pop)
    out = cfg.out
    # combined outputs
    for name in ("summary.json", "summary.md", "README.md", "references_sessions.npz",
                 "conditions_ev.png", "kl_policy_vs_hand.png", "entropy_vs_hand.png"):  # fmt: skip
        assert (out / name).exists(), name
    loaded = load_strict_json(out / "summary.json")
    assert loaded["experiment"] == "E2" and loaded["H"] == 4 and loaded["n_population"] == 3
    assert loaded["conditions"] == {"A": "evaluated", "B": "evaluated", "C": "pending"}
    assert loaded["run"]["population_content_sha256"] == fp
    refs = {"Equilibrium", "Thompson", "BayesBR", "PluralityBR", "Random", "OracleBR"}
    tagged = {f"Transformer({m})[{c}]" for m in ("sample", "argmax") for c in ("A", "B")}
    assert set(loaded["agents"]) == refs | tagged
    assert all(n == 3 for n in loaded["n_sessions_per_agent"].values())  # 3 opponents x 1 session
    # references were evaluated once: the Thompson sessions in both per-condition summaries agree
    a_sum = load_strict_json(out / "A" / "summary.json")
    b_sum = load_strict_json(out / "B" / "summary.json")
    assert (
        a_sum["per_agent"]["Thompson"]["metrics"]["ev"]
        == b_sum["per_agent"]["Thompson"]["metrics"]["ev"]
    )
    assert a_sum["experiment"] == "E2 condition A" and a_sum["training"]["steps"] == 2
    assert b_sum["training"]["steps"] == 3 and b_sum["condition"]["id"] == "B"
    assert set(a_sum["agents"]) == refs | {"Transformer(sample)", "Transformer(argmax)"}
    assert a_sum["criteria"]["sanity_oracle_dominates"]["pass"] is True
    assert a_sum["criteria"]["sanity_equilibrium_vs_nash"]["pass"] is True
    assert {
        "E2-1a_kl_policy",
        "E2-1b_sample_within_thompson",
        "E2-1c_argmax_within_plurality",
    } <= set(a_sum["criteria_e2"])
    for cond in ("A", "B"):
        for name in (
            "summary.md",
            "sessions.npz",
            "ev_vs_hand.png",
            "kl_policy_vs_hand.png",
            "probes.png",
        ):
            assert (out / cond / name).exists(), (cond, name)
        md = (out / cond / "summary.md").read_text()
        assert "## E2 criteria" in md and "PluralityBR" in md and fp in md
    assert not (out / "C").exists()  # pending: nothing written for it, and the run did not fail
    # policy KL: recorded for the mixed transformer, NaN for argmax / pure references
    per = loaded["per_agent"]
    assert all(v is not None for v in per["Transformer(sample)[A]"]["metrics"]["kl_policy"]["mean"])
    for name in ("Transformer(argmax)[A]", "Thompson", "BayesBR", "PluralityBR", "OracleBR"):
        assert all(v is None for v in per[name]["metrics"]["kl_policy"]["mean"]), name
        assert per[name]["summary"]["pure_hands_share"] == 1.0
    assert all(v is not None for v in per["Equilibrium"]["metrics"]["kl_policy"]["mean"])
    # E2 criteria: A's E2-1 evaluated, B's identity KL compared, C / D / E pending or absent
    crit = loaded["criteria_e2"]
    assert crit["E2-1a_kl_policy[A]"]["pass"] in (True, False)
    assert crit["E2-1b_sample_within_thompson[A]"]["n_paired"] == 3
    assert crit["E2-3a_identity_kl[B]<[A]"]["pass"] in (True, False)
    assert (
        crit["E2-3b_ev_change[C]vs[A]"]["pass"] is None
        and "pending" in crit["E2-3b_ev_change[C]vs[A]"]["note"]
    )
    assert (
        crit["E2-2_sample_within_bayes[E]"]["pass"] is None
        and "absent" in crit["E2-2_sample_within_bayes[E]"]["note"]
    )
    assert agg["criteria_e2"]["conditions"] == loaded["conditions"]
    # per-condition table and training config in the combined markdown
    md = (out / "summary.md").read_text()
    for needle in ("# E2 — summary", "## Conditions", "| C | pending |", "## E2 criteria",
                   "Transformer(sample)[A]", "share of exploitable value", "policy KL t=3",
                   "Training configuration — condition A", "| steps | 2 |", "Pending conditions (C)"):  # fmt: skip
        assert needle in md, needle
    assert (
        loaded["per_condition"]["C"]["status"] == "pending"
        and loaded["per_condition"]["A"]["training"]["batch_size"] == 4
    )
    text = (out / "summary.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    readme = (out / "README.md").read_text()
    assert readme.startswith(f"# {out}") and "## eval" in readme and fp in readme
    with np.load(out / "references_sessions.npz") as d:
        assert d["kl_policy"].shape == (18, 4) and d["n_pure_hands"].shape == (18,)
        assert set(d["agent"].tolist()) == refs
    with np.load(out / "A" / "sessions.npz") as d:
        assert set(d["agent"].tolist()) == {"Transformer(sample)", "Transformer(argmax)"}
    # condition directories are inputs: untouched
    assert sorted(p.name for p in conds["A"].iterdir()) == [
        "README.md",
        "model",
        "population.json",
        "population.npz",
    ]
    assert sorted(p.name for p in (conds["A"] / "model").iterdir()) == [
        "log.jsonl",
        "model.pt",
        "train_summary.json",
    ]
    # the rows of the per-condition table carry the share of exploitable value
    rows = {r["agent"]: r for r in e2.condition_rows(loaded)}
    assert rows["OracleBR"]["share_all"] == pytest.approx(1.0) and rows["Equilibrium"][
        "share_all"
    ] == pytest.approx(0.0)
    # plots stage redraws everything from the strict JSON
    (out / "conditions_ev.png").unlink()
    (out / "A" / "ev_vs_hand.png").unlink()
    e2.stage_plots(e2.E2Config(out=out, progress=False))
    assert (out / "conditions_ev.png").exists() and (out / "A" / "ev_vs_hand.png").exists()
    assert "## plots" in (out / "README.md").read_text()
    # deterministic: a second run reproduces the numbers exactly
    again = e2.stage_eval(replace(cfg, out=tmp_path / "e2_again"))
    assert (
        again["per_agent"]["Transformer(sample)[A]"]["metrics"]["ev"]["mean"]
        == agg["per_agent"]["Transformer(sample)[A]"]["metrics"]["ev"]["mean"]
    )
    assert (
        again["per_agent"]["Thompson"]["metrics"]["ev"]["mean"]
        == agg["per_agent"]["Thompson"]["metrics"]["ev"]["mean"]
    )


def test_all_conditions_pending_still_writes_a_summary(tiny_conditions, tmp_path):
    root, conds = tiny_conditions
    pending = tmp_path / "P"
    pending.mkdir()
    shutil.copy(conds["A"] / "population.npz", pending / "population.npz")
    cfg = e2.E2Config(
        out=tmp_path / "e2", conditions={"P": pending, "Q": tmp_path / "Q"}, hands=3,
        eval_sessions_per_opp=1, device="cpu", progress=False,
    )  # fmt: skip
    agg = e2.stage_eval(cfg)
    assert agg["conditions"] == {"P": "pending", "Q": "pending"}
    assert set(agg["agents"]) == {
        "Equilibrium",
        "Thompson",
        "BayesBR",
        "PluralityBR",
        "Random",
        "OracleBR",
    }
    assert all(
        v["pass"] is None
        for k, v in agg["criteria_e2"].items()
        if isinstance(v, dict) and "pass" in v
    )
    assert (cfg.out / "summary.md").exists() and (cfg.out / "conditions_ev.png").exists()
    assert "pending" in (cfg.out / "summary.md").read_text()
