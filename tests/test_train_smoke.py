"""Training machinery: in-context learning on synthetic sessions, CLI run, checkpoint round trip."""

import json
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import torch

from exsolver.data.shards import ShardDataset, to_tensors
from exsolver.data.synthetic import (
    SyntheticConfig,
    SyntheticKuhnSessions,
    no_context_baseline_accuracy,
    per_hand_accuracy,
)
from exsolver.model.inference import Policy
from exsolver.model.losses import compute_loss
from exsolver.model.transformer import ExploitTransformer, ModelConfig, mask_illegal
from exsolver.train import (
    TrainConfig,
    evaluate,
    load_checkpoint,
    lr_at,
    tokenizer_from_checkpoint,
    train,
    trim_padding,
)

N_SESSIONS = 3000
N_EVAL = 300
HANDS = 16


@pytest.fixture(scope="module")
def synthetic_data(tmp_path_factory: pytest.TempPathFactory) -> tuple[SyntheticKuhnSessions, Path]:
    gen = SyntheticKuhnSessions(SyntheticConfig(n_opponents=4, hands_per_session=HANDS), seed=0)
    root = tmp_path_factory.mktemp("synth")
    gen.write(root, n_sessions=N_SESSIONS, seed=1, shard_size=1000)
    return gen, root


def _read_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_lr_schedule() -> None:
    cfg = TrainConfig("d", "o", steps=1000, warmup=100, lr=1e-3, lr_min_ratio=0.1)
    assert lr_at(0, cfg) == pytest.approx(1e-5)
    assert lr_at(99, cfg) == pytest.approx(1e-3)
    assert lr_at(1000, cfg) == pytest.approx(1e-4)
    lrs = [lr_at(s, cfg) for s in range(100, 1001)]
    assert all(a >= b for a, b in pairwise(lrs))
    assert lr_at(5, TrainConfig("d", "o", steps=10, warmup=0, lr=1e-3)) < 1e-3


def test_trim_padding_preserves_positions_and_loss(
    synthetic_data: tuple[SyntheticKuhnSessions, Path],
) -> None:
    _, data_dir = synthetic_data
    ds = ShardDataset(data_dir)
    batch = ds.get_batch(np.arange(16))
    trimmed = trim_padding(batch)
    L, T = batch["tokens"].shape[1], trimmed["tokens"].shape[1]
    assert T < L and T % 8 == 0
    assert torch.equal(trimmed["tokens"], batch["tokens"][:, :T])
    assert torch.all(batch["tokens"][:, T:] == 0)  # only PAD was cut
    for key in ("action_mask", "belief_mask"):
        assert torch.equal(trimmed[key].nonzero(), batch[key].nonzero()), key
        assert not batch[key][:, T:].any()
    assert torch.equal(trimmed["action_target"], batch["action_target"][:, :T])
    assert torch.all(batch["action_target"][:, T:] == -1)
    assert torch.equal(trimmed["legal"], batch["legal"][:, :T]) and not batch["legal"][:, T:].any()
    for key in ("opp_id", "theta", "archetype"):
        assert torch.equal(trimmed[key], batch[key]), key

    torch.manual_seed(0)
    model = ExploitTransformer(
        ModelConfig(
            vocab_size=27, n_opp=4, theta_dim=12, d_model=32, n_layers=1, n_heads=2, max_len=L
        )
    )
    with torch.no_grad():
        full_loss, full_comps = compute_loss(model(batch["tokens"]), batch)
        trim_loss, trim_comps = compute_loss(model(trimmed["tokens"]), trimmed)
    assert torch.allclose(full_loss, trim_loss, atol=1e-5)
    for key, value in full_comps.items():
        assert torch.allclose(value.float(), trim_comps[key].float(), atol=1e-5), key

    # A batch containing a full-length session is returned untouched.
    full = {**batch, "tokens": batch["tokens"].clone()}
    full["tokens"][0, -1] = 3  # END_HAND in the last slot
    assert trim_padding(full) is full


def test_in_context_learning_beats_no_context_baseline(
    synthetic_data: tuple[SyntheticKuhnSessions, Path], tmp_path: Path
) -> None:
    gen, data_dir = synthetic_data
    cfg = TrainConfig(
        data_dir=str(data_dir),
        out_dir=str(tmp_path / "run"),
        steps=300,
        batch_size=64,
        lr=1e-3,
        warmup=30,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_len=256,
        eval_every=0,
        log_every=50,
        eval_size=N_EVAL,
        seed=0,
        device="cpu",
        progress=False,
    )
    summary = train(cfg)

    ds = ShardDataset(data_dir)
    train_ds, eval_ds = ds.split(N_EVAL)
    eval_shard = eval_ds.load_all()
    baseline = no_context_baseline_accuracy(gen.tokenizer, train_ds.load_all(), eval_shard)
    trained = summary["eval"]["acc_action"]
    # Labels are a balanced split per infoset, so no-context accuracy sits near 1/2 (a little
    # above: the current hand's opponent actions carry some information).
    assert 0.4 < baseline < 0.7, baseline
    # In-context inference of the opponent id lifts accuracy well above that (0.79 at calibration).
    assert trained > baseline + 0.10, (trained, baseline)
    assert summary["eval"]["acc_opp"] > 0.5  # the opponent head identifies the id too

    # The gain comes from context: accuracy rises with the number of completed hands.
    model, model_cfg, meta = load_checkpoint(summary["checkpoint"], "cpu")
    assert model_cfg.n_opp == 4 and model_cfg.theta_dim is None  # auto: discrete -> opp head
    batch = to_tensors(eval_shard.as_dict())
    with torch.no_grad():
        logits = model(batch["tokens"])["action_logits"]
    pred = mask_illegal(logits, batch["legal"]).argmax(-1).numpy()
    per_hand = per_hand_accuracy(eval_shard, pred, gen.tokenizer)
    assert per_hand.shape == (HANDS,)
    assert per_hand[-4:].mean() > per_hand[:2].mean() + 0.10, per_hand

    # Log file and checkpoint round trip.
    records = _read_log(Path(summary["log"]))
    assert records[0]["event"] == "start" and records[-1]["event"] == "end"
    assert [r["step"] for r in records if r.get("split") == "train"] == [
        50,
        100,
        150,
        200,
        250,
        300,
    ]
    assert any(r.get("split") == "eval" for r in records)
    reloaded = evaluate(model, eval_ds, torch.device("cpu"), 128)
    assert reloaded["acc_action"] == pytest.approx(trained, abs=1e-6)
    assert reloaded["n_action"] == eval_shard.action_mask.sum()

    # The Policy helper works on counterfactual prefixes built from decoded context.
    tok = tokenizer_from_checkpoint(meta)
    policy = Policy(model, tok, device="cpu")
    hands = tok.decode_session(eval_shard.tokens[0])
    prefixes = [tok.encode_prefix(hands[:8], 0, c) for c in range(3)]
    legal = np.array([[False, True, True]] * 3)
    probs = policy.policy_at(prefixes, legal)
    assert probs.shape == (3, 3) and np.allclose(probs.sum(-1), 1.0, atol=1e-6)
    assert np.all(probs[:, 0] == 0)
    belief = policy.belief_at([tok.encode_prefix(hands, 0, 0)])
    assert belief.shape == (1, 4) and belief.sum() == pytest.approx(1.0, abs=1e-6)


def test_cli_smoke_and_checkpoint_round_trip(
    synthetic_data: tuple[SyntheticKuhnSessions, Path], tmp_path: Path
) -> None:
    _, data_dir = synthetic_data
    out = tmp_path / "cli"
    common = dict(
        steps=6, batch_size=8, lr=1e-3, warmup=2, d_model=32, n_layers=1, n_heads=2, max_len=256,
        eval_every=3, log_every=2, eval_size=16, seed=3, device="cpu", theta_head="on",
    )  # fmt: skip
    cmd = [
        sys.executable,
        "-m",
        "exsolver.train",
        "--data",
        str(data_dir),
        "--out",
        str(out),
        "--no-progress",
    ]
    for key, value in common.items():
        cmd += ["--" + key.replace("_", "-"), str(value)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr
    printed = json.loads(res.stdout)
    assert printed["steps"] == 6 and printed["eval"]["acc_action"] >= 0.0

    assert (out / "log.jsonl").exists() and (out / "model.pt").exists()
    records = _read_log(out / "log.jsonl")
    assert [r["step"] for r in records if r.get("split") == "train"] == [2, 4, 6]
    assert [r["step"] for r in records if r.get("split") == "eval"] == [3, 6]
    assert records[0]["train_config"]["theta_head"] == "on"

    model, model_cfg, meta = load_checkpoint(out / "model.pt", "cpu")
    assert (model_cfg.d_model, model_cfg.n_layers, model_cfg.n_heads) == (32, 1, 2)
    assert model_cfg.vocab_size == 27 and model_cfg.n_opp == 4 and model_cfg.theta_dim == 12
    assert meta["tokenizer_spec"] == {"n_cards": 3, "max_result": 2}
    assert meta["step"] == 6 and meta["train_config"]["steps"] == 6
    assert meta["data_meta"]["vocab_size"] == 27 and meta["data_meta"]["H"] == HANDS
    assert not model.training
    with torch.no_grad():
        out_dict = model(torch.tensor([[1, 2, 4, 6]]))
    assert tuple(out_dict["theta_logits"].shape) == (1, 4, 12)

    # Same config in-process reproduces the CLI run bit-for-bit (seeded, CPU).
    cfg = TrainConfig(
        data_dir=str(data_dir), out_dir=str(tmp_path / "inproc"), progress=False, **common
    )
    summary = train(cfg)
    twin, _, _ = load_checkpoint(summary["checkpoint"], "cpu")
    for (name, a), (_, b) in zip(
        model.state_dict().items(), twin.state_dict().items(), strict=True
    ):
        assert torch.allclose(a, b, atol=1e-6), name
    assert summary["eval"]["loss"] == pytest.approx(printed["eval"]["loss"], abs=1e-5)
