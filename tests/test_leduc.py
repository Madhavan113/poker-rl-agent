import numpy as np
import pytest
from engine_helpers import walk

from exsolver.games import CALL, FOLD, RAISE, GameSpec, LeducPoker
from exsolver.solvers import CFRPlus, expected_value, exploitability
from exsolver.strategy import enumerate_infosets, seat_of_key

Js, Jh, Qs, Qh, Ks, Kh = range(6)
LEDUC_VALUE = -0.085603  # seat-0 game value; OpenSpiel 2.0.2 CFR+ on the default Leduc game


@pytest.fixture(scope="module")
def game() -> LeducPoker:
    return LeducPoker()


def test_spec_and_cards(game):
    assert game.spec == GameSpec(name="leduc", n_cards=6, n_rounds=2, max_result=13, ante=1)
    assert [game.card_name(c) for c in range(6)] == ["Js", "Jh", "Qs", "Qh", "Ks", "Kh"]


def test_infoset_count_is_936_and_stable(game):
    infosets = enumerate_infosets(game)
    assert len(infosets) == 936
    assert sum(seat_of_key(k) == 0 for k in infosets) == 468
    assert list(infosets) == list(enumerate_infosets(game))
    assert all(len(legal) in (2, 3) for legal in infosets.values())


def test_terminals_zero_sum_and_max_result(game):
    biggest = 0.0
    n_terminal = 0
    for s in walk(game):
        if not game.is_terminal(s):
            continue
        n_terminal += 1
        r = game.returns(s)
        assert r[0] + r[1] == 0
        assert float(r[0]).is_integer()
        biggest = max(biggest, abs(r[0]))
    assert biggest == 13 == game.spec.max_result
    assert n_terminal == 30 * (4 + 5 * 4 * 9)


def test_payoffs_by_hand(game):
    def play(c0, c1, board, r1, r2=""):
        s = game.state_from_deal((c0,), (c1,), board=(board,) if board is not None else ())
        for ch in r1:
            s = game.apply(s, {"c": CALL, "b": RAISE, "f": FOLD}[ch])
        if r2 is not None and game.is_chance(s):
            outcomes = game.chance_outcomes(s)
            assert len(outcomes) == 1 and outcomes[0][1] == 1.0
            s = outcomes[0][0]
            for ch in r2:
                s = game.apply(s, {"c": CALL, "b": RAISE, "f": FOLD}[ch])
        assert game.is_terminal(s)
        return game.returns(s)

    # round-1 folds: ante only, or ante + the folder's own raise
    assert play(Js, Kh, None, "bf", None) == (1, -1)
    assert play(Js, Kh, None, "cbf", None) == (-1, 1)
    assert play(Js, Kh, None, "bbf", None) == (-3, 3)
    assert play(Js, Kh, None, "cbbf", None) == (3, -3)
    # round-2 folds carry the round-1 pot
    assert play(Js, Kh, Qs, "bc", "bf") == (3, -3)
    assert play(Js, Kh, Qs, "bbc", "cbbf") == (1 + 4 + 4, -(1 + 4 + 4))
    # showdowns: high card, pair beats high card, split pot
    assert play(Ks, Jh, Qs, "cc", "cc") == (1, -1)
    assert play(Js, Kh, Jh, "cc", "cc") == (1, -1)  # seat 0 pairs the board
    assert play(Ks, Jh, Js, "bc", "bc") == (-(1 + 2 + 4), 1 + 2 + 4)
    assert play(Js, Jh, Kh, "bbc", "bbc") == (0, 0)
    assert play(Kh, Qs, Ks, "bbc", "bbc") == (13, -13)
    assert play(Qh, Ks, Kh, "cbbc", "cbbc") == (-13, 13)


def test_legal_actions_and_raise_cap(game):
    s = game.state_from_deal((Js,), (Kh,))
    assert game.current_player(s) == 0 and game.legal_actions(s) == [CALL, RAISE]
    s = game.apply(s, RAISE)
    assert game.current_player(s) == 1 and game.legal_actions(s) == [FOLD, CALL, RAISE]
    s = game.apply(s, RAISE)
    assert game.current_player(s) == 0 and game.legal_actions(s) == [FOLD, CALL]
    with pytest.raises(ValueError):
        game.apply(s, RAISE)
    s = game.apply(s, CALL)
    assert game.is_chance(s) and not game.is_terminal(s)
    s = game.chance_outcomes(s)[0][0]
    assert game.board_cards(s) == (Jh,)  # first remaining card after Js, Kh are dealt
    assert game.current_player(s) == 0 and game.legal_actions(s) == [CALL, RAISE]
    s = game.apply(game.apply(s, CALL), RAISE)
    assert game.current_player(s) == 0 and game.legal_actions(s) == [FOLD, CALL, RAISE]
    s = game.apply(s, RAISE)
    assert game.legal_actions(s) == [FOLD, CALL]
    for s in walk(game):
        if not game.is_chance(s) and not game.is_terminal(s):
            legal = game.legal_actions(s)
            assert legal == sorted(legal) and CALL in legal
            r = s[3] if s[2] < 0 else s[4]
            assert (RAISE in legal) == (r.count("b") < 2)
            assert (FOLD in legal) == r.endswith("b")


def test_infoset_key_format(game):
    s = game.state_from_deal((Js,), (Qs,), board=(Kh,))
    for a in (CALL, RAISE, CALL):
        s = game.apply(s, a)
    assert game.infoset_key(s, 1) == "1:Qs|cbc"
    s = game.chance_outcomes(s)[0][0]
    s = game.apply(s, CALL)
    assert game.infoset_key(s, 1) == "1:Qs|cbc/Kh|c"
    assert game.infoset_key(s, 0) == "0:Js|cbc/Kh|c"
    assert game.history(s) == "cbc/Kh|c"
    assert game.private_cards(s, 1) == (Qs,) and game.board_cards(s) == (Kh,)


def test_chance_outcomes_and_dealing(game):
    root = game.root()
    outcomes = game.chance_outcomes(root)
    assert len(outcomes) == 30 and all(p == pytest.approx(1 / 30) for _, p in outcomes)
    s = game.apply(game.apply(game.state_from_deal((Js,), (Kh,)), CALL), CALL)
    board = game.chance_outcomes(s)
    assert [b[2] for b, _ in board] == [Jh, Qs, Qh, Ks]
    assert all(p == 0.25 for _, p in board)
    pinned = game.apply(game.apply(game.state_from_deal((Js,), (Kh,), board=(Qh,)), CALL), CALL)
    assert game.chance_outcomes(pinned) == [((Js, Kh, Qh, "cc", "", -1), 1.0)]
    assert game.sample_chance(pinned, np.random.default_rng(0))[2] == Qh
    with pytest.raises(ValueError):
        game.state_from_deal((Js,), (Kh,), board=(Js,))
    rng = np.random.default_rng(0)
    for _ in range(50):
        d = game.deal(rng)
        assert d[0] != d[1] and d[2] == -1 and d[5] == -1
    assert game.deal(rng, {0: (Kh,)})[0] == Kh
    assert game.deal(rng, {1: (Js,)})[1] == Js
    assert game.deal(rng, {0: (Js,), 1: (Qs,)}) == game.state_from_deal((Js,), (Qs,))


def test_sample_chance(game):
    rng = np.random.default_rng(0)
    root = game.root()
    deals = {s for s, _ in game.chance_outcomes(root)}
    draws = {game.sample_chance(root, rng) for _ in range(600)}
    assert draws <= deals and len(draws) == 30
    board_node = game.apply(game.apply(game.state_from_deal((Js,), (Kh,)), RAISE), CALL)
    assert game.is_chance(board_node)
    boards = {s for s, _ in game.chance_outcomes(board_node)}
    draws = {game.sample_chance(board_node, rng) for _ in range(100)}
    assert draws <= boards and {s[2] for s in draws} == {Jh, Qs, Qh, Ks}
    pinned = game.apply(game.apply(game.state_from_deal((Js,), (Kh,), board=(Qs,)), RAISE), CALL)
    assert game.sample_chance(pinned, rng) == game.chance_outcomes(pinned)[0][0]
    for s in (
        game.state_from_deal((Js,), (Kh,)),  # seat 0 to act
        game.apply(game.state_from_deal((Js,), (Kh,)), RAISE),  # seat 1 to act
        game.apply(game.apply(game.state_from_deal((Js,), (Kh,)), RAISE), FOLD),  # terminal
        game.apply(game.chance_outcomes(board_node)[0][0], CALL),  # round 2, seat 1 to act
    ):
        assert not game.is_chance(s)
        with pytest.raises(ValueError):
            game.sample_chance(s, rng)


def test_cfr_plus_converges_towards_known_value(game):
    solver = CFRPlus(game)
    solver.run(20)
    early = solver.average_strategy()
    solver.run(480)
    late = solver.average_strategy()
    expl_early, expl_late = exploitability(game, early), exploitability(game, late)
    assert expl_late < expl_early
    assert expl_late < 2e-3
    assert expected_value(game, late, late) == pytest.approx(LEDUC_VALUE, abs=2e-3)
