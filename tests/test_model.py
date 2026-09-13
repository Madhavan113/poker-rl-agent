"""Model: shapes, causality (CPU and MPS), illegal masking, losses, batched inference."""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from exsolver.data.records import CALL, FOLD, RAISE, HandRecord, act_event
from exsolver.data.tokenizer import Tokenizer, TokenizerSpec
from exsolver.model.inference import Policy, check_legal, pack_prefixes
from exsolver.model.losses import compute_loss
from exsolver.model.transformer import (
    ExploitTransformer,
    ModelConfig,
    count_parameters,
    mask_illegal,
)

VOCAB = 27
MPS = torch.backends.mps.is_available()
KUHN_MAX_SESSION = 577  # 1 + 9 * 64 tokens


def make_model(**overrides: object) -> ExploitTransformer:
    kwargs: dict[str, object] = dict(
        vocab_size=VOCAB, n_opp=5, theta_dim=12, d_model=32, n_layers=2, n_heads=4, max_len=64
    )
    kwargs.update(overrides)
    torch.manual_seed(0)
    return ExploitTransformer(ModelConfig(**kwargs))  # type: ignore[arg-type]


def longest_kuhn_hands(n: int) -> list[HandRecord]:
    """``n`` check-bet-call hands (9 tokens each), alternating seats."""
    hands = []
    for t in range(n):
        seat = t % 2
        if seat == 0:
            events = [act_event(0, CALL), act_event(1, RAISE), act_event(0, CALL)]
        else:
            events = [act_event(1, CALL), act_event(0, RAISE), act_event(1, CALL)]
        mine, theirs = t % 3, (t + 1) % 3
        hands.append(HandRecord(seat, (mine,), events, (theirs,), 2 if mine > theirs else -2))
    return hands


def test_output_shapes_and_optional_heads() -> None:
    model = make_model()
    tokens = torch.randint(0, VOCAB, (3, 20))
    out = model(tokens, return_hidden=True)
    assert tuple(out["action_logits"].shape) == (3, 20, 3)
    assert tuple(out["opp_logits"].shape) == (3, 20, 5)
    assert tuple(out["theta_logits"].shape) == (3, 20, 12)
    assert tuple(out["hidden"].shape) == (3, 20, 32)
    bare = make_model(n_opp=None, theta_dim=None)(tokens)
    assert bare["opp_logits"] is None and bare["theta_logits"] is None
    assert tuple(bare["action_logits"].shape) == (3, 20, 3)
    with pytest.raises(ValueError, match="max_len"):
        model(torch.randint(0, VOCAB, (1, 65)))
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(vocab_size=VOCAB, d_model=30, n_heads=4)


def test_parameter_count() -> None:
    model = make_model()
    assert (
        model.num_parameters()
        == count_parameters(model)
        == sum(p.numel() for p in model.parameters())
    )
    default = ExploitTransformer(ModelConfig(vocab_size=VOCAB, n_opp=256))
    # 4 x 128 with 4 heads: ~0.8M parameters (12 d^2 per block + embeddings + heads).
    assert 0.7e6 < default.num_parameters() < 1.0e6
    assert ModelConfig.from_dict(default.cfg.to_dict()) == default.cfg


def test_causality() -> None:
    model = make_model().eval()
    torch.manual_seed(1)
    a = torch.randint(0, VOCAB, (2, 24))
    b = a.clone()
    p = 7
    b[:, p + 1 :] = torch.randint(0, VOCAB, (2, 24 - p - 1))
    assert not torch.equal(a, b)
    with torch.no_grad():
        out_a, out_b = model(a), model(b)
    for key in ("action_logits", "opp_logits", "theta_logits"):
        assert torch.allclose(out_a[key][:, : p + 1], out_b[key][:, : p + 1], atol=1e-5), key
    assert not torch.allclose(
        out_a["action_logits"][:, p + 1 :], out_b["action_logits"][:, p + 1 :]
    )


@pytest.mark.skipif(not MPS, reason="MPS not available")
def test_causality_on_mps_with_right_padding() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=VOCAB, n_opp=8, d_model=64, n_layers=2, n_heads=4, max_len=1024)
    model = ExploitTransformer(cfg).to("mps").eval()
    T = 600
    assert T >= KUHN_MAX_SESSION
    torch.manual_seed(1)
    full = torch.randint(1, VOCAB, (2, T))  # no PAD inside the "real" stream
    p = 300
    padded = full.clone()
    padded[:, p + 1 :] = 0  # right-PAD everything after p
    with torch.no_grad():
        out_full = model(full.to("mps"))
        out_padded = model(padded.to("mps"))
        out_prefix = model(full[:, : p + 1].to("mps"))
    for key in ("action_logits", "opp_logits"):
        a = out_full[key][:, : p + 1].cpu()
        assert torch.allclose(a, out_padded[key][:, : p + 1].cpu(), atol=1e-4), key
        assert torch.allclose(a, out_prefix[key].cpu(), atol=1e-4), key
    # The padded tail does differ from the unpadded tail (the test is not vacuous).
    assert not torch.allclose(
        out_full["action_logits"][:, p + 1 :], out_padded["action_logits"][:, p + 1 :]
    )


def test_illegal_masking() -> None:
    logits = torch.randn(4, 3)
    legal = torch.tensor(
        [[True, False, True], [False, True, False], [True, True, True], [False, False, True]]
    )
    masked = mask_illegal(logits, legal)
    probs = torch.softmax(masked, dim=-1)
    assert torch.all(probs[~legal] == 0)
    assert torch.allclose(probs.sum(-1), torch.ones(4))
    assert torch.equal(masked[legal], logits[legal])
    assert bool(legal[torch.arange(4), masked.argmax(-1)].all())
    assert torch.isfinite(F.cross_entropy(masked, torch.tensor([2, 1, 0, 2]))).all()
    half = mask_illegal(logits.half(), legal)
    assert torch.isfinite(half).all()


def test_compute_loss_matches_manual_and_respects_masks() -> None:
    torch.manual_seed(0)
    B, T, M, P = 2, 6, 5, 12
    out = {
        "action_logits": torch.randn(B, T, 3, requires_grad=True),
        "opp_logits": torch.randn(B, T, M),
        "theta_logits": torch.randn(B, T, P),
    }
    action_mask = torch.zeros(B, T, dtype=torch.bool)
    action_mask[0, 1] = action_mask[1, 3] = True
    action_target = torch.full((B, T), -1, dtype=torch.long)
    action_target[0, 1], action_target[1, 3] = RAISE, CALL
    legal = torch.zeros(B, T, 3, dtype=torch.bool)
    legal[0, 1] = torch.tensor([False, True, True])
    legal[1, 3] = torch.tensor([True, True, False])
    belief_mask = torch.zeros(B, T, dtype=torch.bool)
    belief_mask[:, 0] = True
    belief_mask[0, 5] = True
    batch = {
        "action_mask": action_mask,
        "action_target": action_target,
        "legal": legal,
        "belief_mask": belief_mask,
        "opp_id": torch.tensor([3, 1]),
        "theta": torch.rand(B, P),
    }
    total, comps = compute_loss(out, batch, lambda_opp=0.5, lambda_theta=0.25)

    sel_logits = mask_illegal(out["action_logits"], legal)[action_mask]
    manual_action = F.cross_entropy(sel_logits, action_target[action_mask])
    assert torch.allclose(comps["loss_action"], manual_action)
    manual_acc = (sel_logits.argmax(-1) == action_target[action_mask]).float().mean()
    assert torch.allclose(comps["acc_action"], manual_acc)
    assert int(comps["n_action"]) == 2 and int(comps["n_belief"]) == 3

    opp_target = batch["opp_id"][:, None].expand(B, T)[belief_mask]
    manual_opp = F.cross_entropy(out["opp_logits"][belief_mask], opp_target)
    assert torch.allclose(comps["loss_opp"], manual_opp)
    theta_target = batch["theta"][:, None, :].expand(B, T, P)[belief_mask]
    manual_theta = F.binary_cross_entropy_with_logits(
        out["theta_logits"][belief_mask], theta_target
    )
    assert torch.allclose(comps["loss_theta"], manual_theta)
    assert torch.allclose(total, manual_action + 0.5 * manual_opp + 0.25 * manual_theta)
    total.backward()
    grad = out["action_logits"].grad
    assert (
        grad is not None
        and torch.all(grad[~action_mask] == 0)
        and torch.any(grad[action_mask] != 0)
    )

    # Continuous-prior sessions (opp_id == -1) contribute nothing to the opponent loss.
    batch_cont = {**batch, "opp_id": torch.tensor([-1, -1])}
    _, comps_cont = compute_loss(out, batch_cont)
    assert float(comps_cont["loss_opp"]) == 0.0 and math.isnan(float(comps_cont["acc_opp"]))
    # No heads -> only the action loss.
    total_bare, comps_bare = compute_loss({"action_logits": out["action_logits"]}, batch)
    assert torch.allclose(total_bare, manual_action) and "loss_opp" not in comps_bare


def test_zero_masked_positions_loss_is_finite_with_zero_grads() -> None:
    model = make_model()
    tokens = torch.randint(0, VOCAB, (2, 12))
    out = model(tokens)
    B, T = tokens.shape
    batch = {
        "action_mask": torch.zeros(B, T, dtype=torch.bool),
        "action_target": torch.full((B, T), -1, dtype=torch.long),
        "legal": torch.zeros(B, T, 3, dtype=torch.bool),
        "belief_mask": torch.zeros(B, T, dtype=torch.bool),
        "opp_id": torch.tensor([0, 1]),
        "theta": torch.rand(B, 12),
    }
    total, comps = compute_loss(out, batch)
    assert torch.isfinite(total) and float(total.detach()) == 0.0
    for key in ("loss_action", "loss_opp", "loss_theta"):
        assert float(comps[key]) == 0.0, key
    assert math.isnan(float(comps["acc_action"])) and math.isnan(float(comps["acc_opp"]))
    assert int(comps["n_action"]) == 0 and int(comps["n_belief"]) == 0
    total.backward()
    for name, p in model.named_parameters():
        assert p.grad is None or torch.all(p.grad == 0), name


def test_pack_prefixes_and_check_legal() -> None:
    tokens, last = pack_prefixes([[1, 2, 3], np.array([4]), [5, 6]], pad=0)
    assert tokens.tolist() == [[1, 2, 3], [4, 0, 0], [5, 6, 0]]
    assert last.tolist() == [2, 0, 1]
    with pytest.raises(ValueError):
        pack_prefixes([[1], []], pad=0)
    assert check_legal([[0, 1, 1]], 1).dtype == bool
    with pytest.raises(ValueError, match="no action"):
        check_legal([[1, 0, 1], [0, 0, 0]], 2)
    with pytest.raises(ValueError, match="shape"):
        check_legal([[1, 1]], 1)


def test_policy_matches_unpadded_forward() -> None:
    tok = Tokenizer(TokenizerSpec(3, 2))
    model = make_model()
    policy = Policy(model, tok, device="cpu")
    ctx = [
        HandRecord(0, (2,), [act_event(0, RAISE), act_event(1, CALL)], (0,), 2),
        HandRecord(1, (0,), [act_event(1, CALL), act_event(0, RAISE), act_event(1, FOLD)], None, 1),
    ]
    prefixes = [
        tok.encode_prefix([], 0, 1),
        tok.encode_prefix(ctx[:1], 1, 2, [act_event(1, RAISE)]),
        tok.encode_prefix(ctx, 0, 0, [act_event(0, CALL), act_event(1, RAISE)]),
        np.array(tok.encode_prefix(ctx, 0, 2)),
    ]
    legal = np.array([[0, 1, 1], [1, 1, 0], [1, 1, 0], [0, 1, 1]], dtype=bool)
    probs = policy.policy_at(prefixes, legal)
    assert probs.shape == (4, 3) and probs.dtype == np.float32
    assert np.allclose(probs.sum(-1), 1.0, atol=1e-6)
    assert np.all(probs[~legal] == 0)
    for i, prefix in enumerate(prefixes):
        with torch.no_grad():
            logits = model(torch.tensor([list(prefix)]))["action_logits"][0, -1]
        expected = torch.softmax(mask_illegal(logits, torch.from_numpy(legal[i])), -1).numpy()
        assert np.allclose(probs[i], expected, atol=1e-5)

    belief = policy.belief_at(prefixes)
    assert belief.shape == (4, 5) and np.allclose(belief.sum(-1), 1.0, atol=1e-6)
    theta = policy.theta_at(prefixes)
    assert theta.shape == (4, 12) and np.all((theta > 0) & (theta < 1))
    everything = policy.all_at(prefixes, legal)
    assert np.allclose(everything["policy"], probs) and np.allclose(everything["belief"], belief)
    with pytest.raises(RuntimeError):
        Policy(make_model(n_opp=None), tok).belief_at(prefixes)


def test_policy_requires_legal_with_at_least_one_action() -> None:
    tok = Tokenizer(TokenizerSpec(3, 2))
    policy = Policy(make_model(), tok)
    prefixes = [tok.encode_prefix([], 0, 1), tok.encode_prefix([], 1, 2)]
    with pytest.raises(TypeError):
        policy.policy_at(prefixes)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="no action"):
        policy.policy_at(prefixes, np.array([[0, 1, 1], [0, 0, 0]], dtype=bool))
    with pytest.raises(ValueError, match="shape"):
        policy.policy_at(prefixes, np.array([[0, 1, 1]], dtype=bool))
    with pytest.raises(ValueError, match="no action"):
        policy.all_at(prefixes, np.zeros((2, 3), dtype=bool))


def test_policy_preserves_callers_train_mode() -> None:
    tok = Tokenizer(TokenizerSpec(3, 2))
    model = make_model(dropout=0.5)
    model.train()
    policy = Policy(model, tok)
    assert model.training  # constructing the wrapper leaves the module alone
    prefixes = [tok.encode_prefix([], 0, 1)]
    legal = np.array([[0, 1, 1]], dtype=bool)
    first = policy.policy_at(prefixes, legal)
    second = policy.policy_at(prefixes, legal)
    assert model.training  # mode restored after each query ...
    assert np.array_equal(first, second)  # ... and the forward ran without dropout
    model.eval()
    policy.policy_at(prefixes, legal)
    assert not model.training


def test_policy_long_prefixes_up_to_577_tokens() -> None:
    tok = Tokenizer(TokenizerSpec(3, 2))
    model = make_model(n_opp=4, max_len=1024)
    hands = longest_kuhn_hands(64)
    full = tok.encode_session(hands)
    assert len(full) == KUHN_MAX_SESSION
    prefixes = [
        tok.encode_prefix([], 0, 0),
        tok.encode_prefix(hands[:10], 1, 2, [act_event(1, RAISE)]),
        tok.encode_prefix(hands[:63], 0, 1, [act_event(0, CALL), act_event(1, RAISE)]),
        full.tolist(),  # belief read-out at the final END_HAND
    ]
    assert len(prefixes[2]) == 573 and max(len(p) for p in prefixes) == KUHN_MAX_SESSION
    legal = np.array([[0, 1, 1], [1, 1, 0], [1, 1, 0], [1, 1, 1]], dtype=bool)
    policy = Policy(model, tok)
    probs = policy.policy_at(prefixes, legal)
    belief = policy.belief_at(prefixes)
    for i, prefix in enumerate(prefixes):
        with torch.no_grad():
            out = model(torch.tensor([list(prefix)]))
        expected = torch.softmax(
            mask_illegal(out["action_logits"][0, -1], torch.from_numpy(legal[i])), -1
        )
        assert np.allclose(probs[i], expected.numpy(), atol=1e-5), i
        assert np.allclose(
            belief[i], torch.softmax(out["opp_logits"][0, -1], -1).numpy(), atol=1e-5
        ), i
