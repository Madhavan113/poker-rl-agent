"""Shard format: dtypes/shapes, validation, write/read round trip, dataset and collate."""

from pathlib import Path

import numpy as np
import pytest
import torch

from exsolver.data.records import FOLD, RAISE
from exsolver.data.shards import (
    SHARD_DTYPES,
    SHARD_KEYS,
    Shard,
    ShardDataset,
    collate,
    load_all,
    read_meta,
    shard_paths,
    write_meta,
)
from exsolver.data.synthetic import SyntheticConfig, SyntheticKuhnSessions

H = 8
L = 1 + 9 * H
TORCH_DTYPES = {
    "tokens": torch.long,
    "action_mask": torch.bool,
    "action_target": torch.long,
    "legal": torch.bool,
    "belief_mask": torch.bool,
    "opp_id": torch.long,
    "theta": torch.float32,
    "archetype": torch.long,
}


@pytest.fixture
def gen() -> SyntheticKuhnSessions:
    return SyntheticKuhnSessions(SyntheticConfig(n_opponents=4, hands_per_session=H), seed=0)


def test_shard_format_and_npz_round_trip(tmp_path: Path, gen: SyntheticKuhnSessions) -> None:
    shard = gen.generate(20, np.random.default_rng(0))
    for key, dtype in SHARD_DTYPES.items():
        assert getattr(shard, key).dtype == dtype, key
    assert shard.tokens.shape == (20, L)
    assert (
        shard.action_mask.shape == shard.action_target.shape == shard.belief_mask.shape == (20, L)
    )
    assert shard.legal.shape == (20, L, 3)
    assert shard.opp_id.shape == shard.archetype.shape == (20,)
    assert shard.theta.shape == (20, 12) and shard.P == 12
    shard.validate(L=L, vocab_size=gen.tokenizer.vocab_size)

    tok = gen.tokenizer
    assert np.array_equal(shard.action_mask, tok.decision_mask(shard.tokens))
    assert np.array_equal(shard.belief_mask, tok.belief_mask(shard.tokens))
    assert (shard.action_target[~shard.action_mask] == -1).all()
    assert shard.legal[shard.action_mask].any(axis=1).all()
    assert not shard.legal[~shard.action_mask].any()
    assert shard.belief_mask.sum(axis=1).tolist() == [H + 1] * 20  # BOS + one END_HAND per hand
    assert (shard.tokens[:, 0] == tok.BOS).all()
    assert set(shard.opp_id.tolist()) <= set(range(4))
    assert np.allclose(shard.theta, gen.thetas[shard.opp_id])
    # Every session decodes back into H hands whose agent decisions match the mask count.
    for s in range(shard.n):
        hands = tok.decode_session(shard.tokens[s])
        assert len(hands) == H
        n_dec = sum(1 for h in hands for ev in h.events if ev[0] == "act" and ev[1] == 0)
        assert n_dec == shard.action_mask[s].sum()

    path = shard.save(tmp_path / "s.npz")
    loaded = Shard.load(path)
    for key in SHARD_KEYS:
        assert getattr(loaded, key).dtype == SHARD_DTYPES[key]
        assert np.array_equal(getattr(loaded, key), getattr(shard, key)), key
    compressed = Shard.load(shard.save(tmp_path / "c.npz", compress=True))
    assert np.array_equal(compressed.tokens, shard.tokens)


def test_validate_rejects_violations(gen: SyntheticKuhnSessions) -> None:
    shard = gen.generate(4, np.random.default_rng(0))
    bad = Shard(**{**shard.as_dict(), "tokens": shard.tokens.astype(np.int32)})
    with pytest.raises(ValueError, match="dtype"):
        bad.validate()
    bad = shard.take(slice(None))
    bad.action_target = bad.action_target.copy()
    bad.action_target[0, 0] = 1  # position 0 is BOS, never a decision
    with pytest.raises(ValueError, match="-1 off"):
        bad.validate()
    bad = shard.take(slice(None))
    bad.legal = bad.legal.copy()
    p = np.flatnonzero(bad.action_mask[0])[0]
    bad.legal[0, p, bad.action_target[0, p]] = False  # target no longer legal
    with pytest.raises(ValueError, match="legal"):
        bad.validate()
    with pytest.raises(ValueError, match="L="):
        shard.validate(L=L + 1)
    with pytest.raises(ValueError, match="keys"):
        Shard.from_arrays(tokens=shard.tokens)


def test_write_shards_meta_and_dataset(tmp_path: Path, gen: SyntheticKuhnSessions) -> None:
    paths = gen.write(tmp_path, n_sessions=25, seed=3, shard_size=10)
    assert [p.name for p in paths] == ["shard_00000.npz", "shard_00001.npz", "shard_00002.npz"]
    assert shard_paths(tmp_path) == paths
    meta = read_meta(tmp_path)
    assert meta["game"] == "kuhn-synthetic" and meta["vocab_size"] == 27
    assert meta["L"] == L and meta["H"] == H and meta["seed"] == 3 and meta["n_opp"] == 4
    assert meta["n_cards"] == 3 and meta["max_result"] == 2
    assert meta["collection_mix"] == {"uniform": 1.0}

    everything = load_all(tmp_path)
    assert everything.n == 25
    everything.validate(L=L, vocab_size=27)

    ds = ShardDataset(tmp_path)
    assert len(ds) == 25 and ds.L == L and ds.P == 12 and ds.meta == meta
    sample = ds[13]
    assert set(sample) == set(SHARD_KEYS)
    for key, dtype in TORCH_DTYPES.items():
        assert sample[key].dtype == dtype, key
    assert tuple(sample["tokens"].shape) == (L,)
    assert tuple(sample["legal"].shape) == (L, 3)
    assert tuple(sample["theta"].shape) == (12,)
    assert sample["opp_id"].ndim == 0
    assert np.array_equal(sample["tokens"].numpy(), everything.tokens[13])

    idx = [3, 13, 22]  # one row from each shard file
    batch = collate([ds[i] for i in idx])
    fast = ds.get_batch(idx)
    for key in SHARD_KEYS:
        assert tuple(batch[key].shape) == (3, *sample[key].shape)
        assert batch[key].dtype == TORCH_DTYPES[key]
        assert torch.equal(batch[key], fast[key]), key
    assert np.array_equal(fast["opp_id"].numpy(), everything.opp_id[idx])
    assert np.array_equal(fast["theta"].numpy(), everything.theta[idx])

    train, held = ds.split(5)
    assert len(train) == 20 and len(held) == 5
    assert np.array_equal(held.get_batch([0, 4])["tokens"].numpy(), everything.tokens[[20, 24]])
    assert np.array_equal(ds.load_all().tokens, everything.tokens)
    assert np.array_equal(train.subset([1, 0]).load_all().opp_id, everything.opp_id[[1, 0]])


def test_write_is_deterministic(tmp_path: Path, gen: SyntheticKuhnSessions) -> None:
    gen.write(tmp_path / "a", n_sessions=12, seed=7, shard_size=5)
    gen.write(tmp_path / "b", n_sessions=12, seed=7, shard_size=5)
    a, b = load_all(tmp_path / "a"), load_all(tmp_path / "b")
    for key in SHARD_KEYS:
        assert np.array_equal(getattr(a, key), getattr(b, key)), key
    other = SyntheticKuhnSessions(SyntheticConfig(n_opponents=4, hands_per_session=H), seed=1)
    assert not np.array_equal(other.labels, gen.labels)


def test_meta_round_trip(tmp_path: Path) -> None:
    meta = {
        "game": "kuhn",
        "seed": np.int64(3),
        "mix": {"eq": 0.4},
        "path": Path("pop.npz"),
        "w": np.ones(2),
    }
    write_meta(tmp_path, meta)
    back = read_meta(tmp_path)
    assert back == {
        "game": "kuhn",
        "seed": 3,
        "mix": {"eq": 0.4},
        "path": "pop.npz",
        "w": [1.0, 1.0],
    }
    assert read_meta(tmp_path / "missing") == {}


def test_empty_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        ShardDataset(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_all(tmp_path)


def test_load_rejects_dtype_mismatch_and_missing_arrays(
    tmp_path: Path, gen: SyntheticKuhnSessions
) -> None:
    shard = gen.generate(3, np.random.default_rng(0))
    arrays = shard.as_dict()
    arrays["tokens"] = arrays["tokens"].astype(np.int32)
    np.savez(tmp_path / "bad.npz", **arrays)
    with pytest.raises(ValueError, match="dtype"):
        Shard.load(tmp_path / "bad.npz")
    del arrays["theta"]
    np.savez(tmp_path / "missing.npz", **arrays)
    with pytest.raises(ValueError, match="missing"):
        Shard.load(tmp_path / "missing.npz")
    # Casting is only ever explicit.
    cast = Shard.from_arrays(**{**shard.as_dict(), "tokens": shard.tokens.astype(np.int32)})
    assert cast.tokens.dtype == np.int16
    with pytest.raises(ValueError, match="dtype"):
        Shard(**{**shard.as_dict(), "opp_id": shard.opp_id.astype(np.int64)}).save(
            tmp_path / "x.npz"
        )


def _misalign(shard: Shard) -> Shard:
    """Shift the decision annotations one position right: format-valid, tokenizer-inconsistent."""
    bad = shard.take(slice(None))
    bad.action_mask = np.roll(bad.action_mask, 1, axis=1)
    bad.action_target = np.roll(bad.action_target, 1, axis=1)
    bad.legal = np.roll(bad.legal, 1, axis=1)
    return bad


def test_validate_with_tokenizer_rejects_misaligned_masks(gen: SyntheticKuhnSessions) -> None:
    tok = gen.tokenizer
    shard = gen.generate(4, np.random.default_rng(0))
    shard.validate(L=L, tokenizer=tok)

    bad = _misalign(shard)
    bad.validate()  # the format invariants alone cannot see the shift ...
    with pytest.raises(ValueError, match="action_mask"):
        bad.validate(tokenizer=tok)  # ... the tokenizer-aware check does

    bad = shard.take(slice(None))
    bad.belief_mask = bad.belief_mask.copy()
    bad.belief_mask[0, 0] = False
    with pytest.raises(ValueError, match="belief_mask"):
        bad.validate(tokenizer=tok)

    bad = shard.take(slice(None))
    bad.legal = bad.legal.copy()
    bad.legal[0, 0, 1] = True  # BOS position is never a decision
    with pytest.raises(ValueError, match="all-False"):
        bad.validate()

    bad = shard.take(slice(None))
    bad.tokens = bad.tokens.copy()
    bad.tokens[0, 0] = tok.HAND
    with pytest.raises(ValueError, match="BOS"):
        bad.validate(tokenizer=tok)

    bad = shard.take(slice(None))
    bad.tokens = bad.tokens.copy()
    bad.tokens[0, 1] = tok.vocab_size
    with pytest.raises(ValueError, match="vocabulary"):
        bad.validate(tokenizer=tok)


def test_dataset_validates_each_shard_on_first_load(
    tmp_path: Path, gen: SyntheticKuhnSessions
) -> None:
    gen.write(tmp_path, n_sessions=20, seed=3, shard_size=10)
    corrupt = shard_paths(tmp_path)[1]
    _misalign(Shard.load(corrupt)).save(corrupt)

    ds = ShardDataset(tmp_path)
    assert ds.opp_ids.shape == (20,) and ds.L == L and ds.P == 12
    assert ds._store._cache == {}  # meta + per-shard opp_id only; no full shard loaded yet
    _ = ds[0]  # shard 0 is fine
    with pytest.raises(ValueError, match="invalid shard .*shard_00001"):
        ds[15]
    with pytest.raises(ValueError):
        ds.get_batch([1, 15])
    raw = ShardDataset(tmp_path, validate=False)
    assert tuple(raw[15]["tokens"].shape) == (L,)


def test_tokens_do_not_depend_on_labels() -> None:
    cfg = SyntheticConfig(n_opponents=4, hands_per_session=H)
    a = SyntheticKuhnSessions(cfg, seed=0)
    b = SyntheticKuhnSessions(cfg, seed=5)
    assert not np.array_equal(a.labels, b.labels)  # different label tables ...
    assert np.array_equal(a.thetas, b.thetas)  # ... same behaviours
    sa = a.generate(30, np.random.default_rng(11))
    sb = b.generate(30, np.random.default_rng(11))
    for key in SHARD_KEYS:
        if key != "action_target":
            assert np.array_equal(getattr(sa, key), getattr(sb, key)), key
    assert not np.array_equal(sa.action_target, sb.action_target)
    # The taken action (context) is not the label: the uniform collection policy agrees with
    # the label about half the time, so labels cannot be read off the token stream.
    flat_pos = np.flatnonzero(sa.action_mask.ravel())
    taken = np.array([a.tokenizer.action_of_token(t) for t in sa.tokens.ravel()[flat_pos + 1]])
    agreement = (taken == sa.action_target.ravel()[flat_pos]).mean()
    assert 0.35 < agreement < 0.65, agreement


def test_tokens_do_not_depend_on_opp_id_beyond_behaviour() -> None:
    g = SyntheticKuhnSessions(SyntheticConfig(n_opponents=4, hands_per_session=H), seed=0)
    g.p_raise[1] = g.p_raise[0]  # give id 1 exactly id 0's behaviour ...
    g.p_call[1] = g.p_call[0]
    assert not np.array_equal(g.labels[0], g.labels[1])  # ... but its own labels
    hands0, dec0, _ = g.sample_session(np.random.default_rng(3), opp_id=0)
    hands1, dec1, _ = g.sample_session(np.random.default_rng(3), opp_id=1)
    assert hands0 == hands1
    assert np.array_equal(g.tokenizer.encode_session(hands0), g.tokenizer.encode_session(hands1))
    assert [legal for _, legal in dec0] == [legal for _, legal in dec1]
    assert [label for label, _ in dec0] != [label for label, _ in dec1]


def test_show_iff_showdown(gen: SyntheticKuhnSessions) -> None:
    tok = gen.tokenizer
    shard = gen.generate(50, np.random.default_rng(0))
    for s in range(shard.n):
        for hand in tok.decode_session(shard.tokens[s]):
            folded = hand.events[-1][2] == FOLD
            assert (hand.opp_cards is None) == folded
            if folded:
                assert abs(hand.result) == 1
            else:
                assert hand.opp_cards[0] != hand.my_cards[0]
                bet = any(ev[2] == RAISE for ev in hand.events)
                assert abs(hand.result) == (2 if bet else 1)
                assert (hand.result > 0) == (hand.my_cards[0] > hand.opp_cards[0])
    show_ids = [tok.show_token(c) for c in range(3)]
    n_show = np.isin(shard.tokens, show_ids).sum()
    n_no_show = (shard.tokens == tok.NO_SHOW).sum()
    assert n_show + n_no_show == shard.n * H
