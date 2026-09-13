"""Tokenizer: vocabulary order, encode/decode round trips, positions, prefixes."""

from types import SimpleNamespace

import numpy as np
import pytest

from exsolver.data.records import (
    CALL,
    FOLD,
    RAISE,
    HandRecord,
    SessionContext,
    act_event,
    board_event,
)
from exsolver.data.tokenizer import Tokenizer, TokenizerSpec, build_vocab

KUHN = TokenizerSpec(n_cards=3, max_result=2)
LEDUC = TokenizerSpec(n_cards=6, max_result=13)

KUHN_VOCAB = [
    "PAD", "BOS", "HAND", "END_HAND", "POS_0", "POS_1",
    "CARD_0", "CARD_1", "CARD_2",
    "BOARD_0", "BOARD_1", "BOARD_2",
    "ME_FOLD", "ME_CALL", "ME_RAISE", "OPP_FOLD", "OPP_CALL", "OPP_RAISE",
    "SHOW_0", "SHOW_1", "SHOW_2", "NO_SHOW",
    "RESULT_-2", "RESULT_-1", "RESULT_0", "RESULT_1", "RESULT_2",
]  # fmt: skip

# All four Kuhn line shapes, both seats, showdown and fold endings.
KUHN_HANDS = [
    HandRecord(0, (2,), [act_event(0, RAISE), act_event(1, CALL)], (0,), 2),
    HandRecord(1, (0,), [act_event(1, CALL), act_event(0, RAISE), act_event(1, FOLD)], None, 1),
    HandRecord(0, (1,), [act_event(0, CALL), act_event(1, RAISE), act_event(0, FOLD)], None, -1),
    HandRecord(1, (1,), [act_event(1, CALL), act_event(0, CALL)], (2,), -1),
    HandRecord(0, (0,), [act_event(0, CALL), act_event(1, RAISE), act_event(0, CALL)], (2,), -2),
    HandRecord(1, (2,), [act_event(1, RAISE), act_event(0, CALL)], (1,), 2),
]


def test_vocab_order_and_size_kuhn() -> None:
    tok = Tokenizer(KUHN)
    assert tok.names == KUHN_VOCAB
    assert tok.vocab_size == 27 == KUHN.vocab_size == len(build_vocab(KUHN))
    for i, name in enumerate(KUHN_VOCAB):
        assert tok.id(name) == i
        assert tok.name(i) == name
    assert (tok.PAD, tok.BOS, tok.HAND, tok.END_HAND, tok.POS_0, tok.POS_1) == (0, 1, 2, 3, 4, 5)
    assert tok.NO_SHOW == tok.id("NO_SHOW") == 21
    assert tok.result_token(-2) == 22 and tok.result_token(0) == 24 and tok.result_token(2) == 26


def test_vocab_order_and_size_leduc() -> None:
    tok = Tokenizer(LEDUC)
    names = tok.names
    assert tok.vocab_size == 58 == LEDUC.vocab_size
    assert names[:6] == ["PAD", "BOS", "HAND", "END_HAND", "POS_0", "POS_1"]
    assert names[6:12] == [f"CARD_{c}" for c in range(6)]
    assert names[12:18] == [f"BOARD_{c}" for c in range(6)]
    assert names[18:24] == ["ME_FOLD", "ME_CALL", "ME_RAISE", "OPP_FOLD", "OPP_CALL", "OPP_RAISE"]
    assert names[24:30] == [f"SHOW_{c}" for c in range(6)]
    assert names[30] == "NO_SHOW"
    assert names[31:] == [f"RESULT_{r}" for r in range(-13, 14)]
    assert tok.result_of_token(tok.id("RESULT_-13")) == -13
    assert tok.result_of_token(tok.id("RESULT_13")) == 13


def test_from_game_spec_duck_typing() -> None:
    game_spec = SimpleNamespace(name="kuhn", n_cards=3, n_rounds=1, max_result=2, ante=1)
    assert TokenizerSpec.from_game_spec(game_spec) == KUHN
    assert Tokenizer(game_spec).names == KUHN_VOCAB
    assert TokenizerSpec.from_game_spec(KUHN) is KUHN


def test_encode_hand_format_and_kuhn_length_bound() -> None:
    tok = Tokenizer(KUHN)
    ids = tok.encode_hand(KUHN_HANDS[0])
    assert [tok.name(i) for i in ids] == [
        "HAND", "POS_0", "CARD_2", "ME_RAISE", "OPP_CALL", "SHOW_0", "RESULT_2", "END_HAND",
    ]  # fmt: skip
    lengths = [len(tok.encode_hand(h)) for h in KUHN_HANDS]
    assert max(lengths) <= 9
    assert len(tok.encode_hand(KUHN_HANDS[4])) == 9  # check-bet-call is the longest line
    ids = tok.encode_hand(KUHN_HANDS[1])
    assert [tok.name(i) for i in ids] == [
        "HAND", "POS_1", "CARD_0", "OPP_CALL", "ME_RAISE", "OPP_FOLD", "NO_SHOW", "RESULT_1", "END_HAND",
    ]  # fmt: skip


def test_round_trip_kuhn_session() -> None:
    tok = Tokenizer(KUHN)
    flat = tok.encode_session(KUHN_HANDS)
    assert flat.dtype == np.int16
    assert flat[0] == tok.BOS
    n_real = 1 + sum(len(tok.encode_hand(h)) for h in KUHN_HANDS)
    assert flat.shape == (n_real,)
    assert tok.decode_session(flat) == KUHN_HANDS

    padded = tok.encode_session(KUHN_HANDS, L=64)
    assert padded.shape == (64,) and padded.dtype == np.int16
    assert np.array_equal(padded[:n_real], flat)
    assert (padded[n_real:] == tok.PAD).all()
    assert tok.decode_session(padded) == KUHN_HANDS
    # SessionContext is accepted wherever a list of hands is.
    assert np.array_equal(tok.encode_session(SessionContext(list(KUHN_HANDS))), flat)
    assert tok.decode_session(tok.encode_session([])) == []


def test_round_trip_leduc_board_and_multicard() -> None:
    tok = Tokenizer(LEDUC)
    leduc = HandRecord(
        seat=1,
        my_cards=(3,),
        events=[
            act_event(1, CALL),
            act_event(0, RAISE),
            act_event(1, CALL),
            board_event(5),
            act_event(1, RAISE),
            act_event(0, CALL),
        ],  # fmt: skip
        opp_cards=(4,),
        result=7,
    )
    folded = HandRecord(0, (0,), [act_event(0, RAISE), act_event(1, FOLD)], None, 1)
    multi = HandRecord(
        0, (0, 5), [act_event(0, CALL), board_event(2), act_event(1, CALL)], (1, 3), -13
    )
    hands = [leduc, folded, multi]
    tokens = tok.encode_session(hands, L=60)
    names = [tok.name(t) for t in tok.encode_hand(multi)]
    assert names[:4] == ["HAND", "POS_0", "CARD_0", "CARD_5"]
    assert names[-4:] == ["SHOW_1", "SHOW_3", "RESULT_-13", "END_HAND"]
    assert tok.decode_session(tokens) == hands


def test_encode_session_too_long_raises() -> None:
    tok = Tokenizer(KUHN)
    with pytest.raises(ValueError, match="longer than L"):
        tok.encode_session(KUHN_HANDS[:1], L=5)
    assert tok.encode_session(KUHN_HANDS[:1], L=9).shape == (9,)


def test_decision_and_belief_positions() -> None:
    tok = Tokenizer(KUHN)
    tokens = tok.encode_session(KUHN_HANDS[:2], L=32)
    # BOS(0) | HAND(1) POS_0(2) CARD_2(3) ME_RAISE(4) OPP_CALL(5) SHOW_0(6) RESULT_2(7) END_HAND(8)
    #        | HAND(9) POS_1(10) CARD_0(11) OPP_CALL(12) ME_RAISE(13) OPP_FOLD(14) NO_SHOW(15)
    #          RESULT_1(16) END_HAND(17) PAD...
    assert tok.decision_positions(tokens).tolist() == [3, 12]
    assert tok.belief_positions(tokens).tolist() == [0, 8, 17]
    for p in tok.decision_positions(tokens):
        assert tok.is_me_action(int(tokens[p + 1]))
    mask = tok.decision_mask(tokens)
    assert mask.shape == tokens.shape and mask.sum() == 2
    batch = np.stack([tokens, tokens])
    assert np.array_equal(tok.decision_mask(batch), np.stack([mask, mask]))
    assert np.array_equal(tok.belief_mask(batch)[1], tok.belief_mask(tokens))
    # Hand with two agent decisions (check, then call facing a bet).
    two = tok.encode_session([KUHN_HANDS[4]])
    assert tok.decision_positions(two).tolist() == [3, 5]
    assert [tok.name(two[p + 1]) for p in (3, 5)] == ["ME_CALL", "ME_CALL"]


def test_encode_prefix_is_prefix_of_session() -> None:
    tok = Tokenizer(KUHN)
    for i, hand in enumerate(KUHN_HANDS):
        ctx = KUHN_HANDS[:i]
        full = tok.encode_session([*ctx, hand])
        decisions = set(tok.decision_positions(full).tolist())
        for k in range(len(hand.events) + 1):
            prefix = tok.encode_prefix(ctx, hand.seat, hand.my_cards, hand.events[:k])
            assert full[: len(prefix)].tolist() == prefix
            is_agent_turn = k < len(hand.events) and hand.events[k][1] == 0
            assert (len(prefix) - 1 in decisions) == is_agent_turn
    assert tok.encode_prefix([], 0, 2) == [tok.BOS, tok.HAND, tok.POS_0, tok.card_token(2)]
    assert tok.encode_prefix(SessionContext(), 1, (1,)) == [
        tok.BOS,
        tok.HAND,
        tok.POS_1,
        tok.card_token(1),
    ]


def test_action_token_helpers() -> None:
    tok = Tokenizer(KUHN)
    for a in (FOLD, CALL, RAISE):
        me, opp = tok.me_action_token(a), tok.opp_action_token(a)
        assert tok.action_of_token(me) == a == tok.action_of_token(opp)
        assert tok.is_me_action(me) and not tok.is_opp_action(me)
        assert tok.is_opp_action(opp) and not tok.is_me_action(opp)
        assert tok.action_token(0, a) == me and tok.action_token(1, a) == opp
    assert tok.name(tok.me_action_token(RAISE)) == "ME_RAISE"
    assert tok.name(tok.opp_action_token(FOLD)) == "OPP_FOLD"
    with pytest.raises(ValueError):
        tok.action_of_token(tok.BOS)
    assert tok.card_of_token(tok.card_token(1)) == 1
    assert tok.card_of_token(tok.board_token(2)) == 2
    assert tok.card_of_token(tok.show_token(0)) == 0


def test_invalid_inputs_raise() -> None:
    tok = Tokenizer(KUHN)
    with pytest.raises(ValueError):
        tok.card_token(3)
    with pytest.raises(ValueError):
        tok.result_token(3)
    with pytest.raises(ValueError):
        tok.pos_token(2)
    with pytest.raises(ValueError):
        tok.me_action_token(3)
    with pytest.raises(ValueError):
        tok.encode_event(("deal", 1))
    with pytest.raises(ValueError):
        tok.encode_hand(HandRecord(0, (), [], None, 0))
    with pytest.raises(ValueError):
        tok.encode_hand(HandRecord(0, (0,), [], (), 0))


def test_decode_rejects_malformed_streams() -> None:
    tok = Tokenizer(KUHN)
    good = tok.encode_session(KUHN_HANDS[:1])
    with pytest.raises(ValueError, match="BOS"):
        tok.decode_session(good[1:])
    with pytest.raises(ValueError, match="truncated"):
        tok.decode_session(good[:-2])
    with pytest.raises(ValueError):
        tok.decode_session(np.stack([good, good]))
