"""play_hand / seat_infosets_with_prefixes: observation-only records, validation, determinism."""

import numpy as np
import pytest

from exsolver.data.records import ACT, BOARD
from exsolver.games import CALL, FOLD, RAISE, KuhnPoker, LeducPoker
from exsolver.play import check_row, play_hand, seat_infosets_with_prefixes
from exsolver.solvers import cfr_plus
from exsolver.strategy import enumerate_infosets, seat_infosets, seat_of_key, uniform_strategy
from tests.engine_helpers import pure_strategy, random_strategy


def replay(game, hand, decisions):
    """Rebuild the terminal state from the agent's record and check every step against the game.

    When the opponent's card stayed hidden any card consistent with the deal is used: fold
    payoffs do not depend on cards. Returns the terminal state.
    """
    my = hand.my_cards[0]
    board = [ev[1] for ev in hand.events if ev[0] == BOARD]
    if hand.opp_cards is not None:
        other = hand.opp_cards[0]
    else:
        other = next(c for c in range(game.spec.n_cards) if c != my and c not in board)
    cards0, cards1 = ((my,), (other,)) if hand.seat == 0 else ((other,), (my,))
    s = game.state_from_deal(cards0, cards1, tuple(board))
    dec = iter(decisions)
    for ev in hand.events:
        if ev[0] == BOARD:
            assert game.is_chance(s)
            ((s, _),) = game.chance_outcomes(s)  # the board is pinned -> a single outcome
            assert game.board_cards(s)[-1] == ev[1]
            continue
        _, actor, a = ev
        assert not game.is_chance(s)
        assert game.current_player(s) == (hand.seat if actor == 0 else 1 - hand.seat)
        if actor == 0:
            d = next(dec)
            assert d.infoset_key == game.infoset_key(s, hand.seat)
            assert d.taken == a
            legal = game.legal_actions(s)
            assert d.legal.tolist() == [x in legal for x in range(3)]
        s = game.apply(s, a)
    assert next(dec, None) is None
    assert game.is_terminal(s)
    return s


@pytest.mark.parametrize("game", [KuhnPoker(), LeducPoker()], ids=["kuhn", "leduc"])
def test_records_are_consistent_with_the_game(game):
    rng = np.random.default_rng(0)
    agent = random_strategy(game, rng)
    opp = random_strategy(game, rng)
    n_showdown = n_fold = n_board = 0
    for t in range(300):
        seat = t % 2
        hand, decisions = play_hand(game, seat, agent, opp, rng)
        assert hand.seat == seat
        assert hand.events[0][0] == ACT and hand.events[0][1] == (0 if seat == 0 else 1)
        folded = any(ev[0] == ACT and ev[2] == FOLD for ev in hand.events)
        assert (hand.opp_cards is None) == folded
        assert isinstance(hand.result, int) and abs(hand.result) <= game.spec.max_result
        s = replay(game, hand, decisions)
        assert game.returns(s)[seat] == hand.result
        assert hand.my_cards == game.private_cards(s, seat)
        if hand.opp_cards is not None:
            assert hand.opp_cards == game.private_cards(s, 1 - seat)
        n_showdown += hand.opp_cards is not None
        n_fold += folded
        n_board += any(ev[0] == BOARD for ev in hand.events)
    assert n_showdown > 20 and n_fold > 20
    if game.spec.n_rounds > 1:
        assert n_board > 20
        assert all(len([ev for ev in h.events if ev[0] == BOARD]) <= 1 for h in [hand])


def test_check_fold_agent_never_sees_a_showdown():
    game = KuhnPoker()
    agent = pure_strategy(game, lambda k, legal: FOLD if FOLD in legal else CALL)
    opp = pure_strategy(game, lambda k, legal: RAISE if RAISE in legal else CALL)
    rng = np.random.default_rng(1)
    for t in range(100):
        hand, decisions = play_hand(game, t % 2, agent, opp, rng)
        assert hand.opp_cards is None
        assert hand.result == -1
        assert all(seat_of_key(d.infoset_key) == t % 2 for d in decisions)
        if t % 2 == 0:  # check, opponent bets, fold
            assert [ev[2] for ev in hand.events] == [CALL, RAISE, FOLD]
        else:  # opponent bets, fold
            assert [ev[2] for ev in hand.events] == [RAISE, FOLD]


def test_bad_strategy_rows_raise_instead_of_renormalising():
    game = KuhnPoker()
    eq = cfr_plus(game, 50)
    rng = np.random.default_rng(2)
    illegal = dict(eq)
    illegal["0:J|"] = illegal["0:Q|"] = illegal["0:K|"] = np.array([0.5, 0.5, 0.0])  # FOLD illegal
    with pytest.raises(ValueError, match="illegal"):
        play_hand(game, 0, illegal, eq, rng)
    short = dict(eq)
    for k in ("0:J|", "0:Q|", "0:K|"):
        short[k] = np.array([0.0, 0.9, 0.05])
    with pytest.raises(ValueError, match="sums to"):
        play_hand(game, 0, short, eq, rng)
    with pytest.raises(KeyError, match="agent strategy has no entry"):
        play_hand(game, 0, {}, eq, rng)
    with pytest.raises(KeyError, match="opponent strategy has no entry"):
        play_hand(game, 0, eq, {}, rng)
    with pytest.raises(ValueError):
        play_hand(game, 2, eq, eq, rng)
    # row validation details
    check_row(np.array([0.0, 0.3, 0.7 + 5e-7]), [CALL, RAISE], "k")  # within tolerance
    with pytest.raises(ValueError):
        check_row(np.array([0.0, 0.3, 0.7 + 2e-6]), [CALL, RAISE], "k")
    with pytest.raises(ValueError):
        check_row(np.array([0.0, 1.2, -0.2]), [CALL, RAISE], "k")
    with pytest.raises(ValueError):
        check_row(np.array([0.5, 0.5]), [CALL, RAISE], "k")


def test_determinism_under_seed():
    game = KuhnPoker()
    rng = np.random.default_rng(3)
    agent, opp = random_strategy(game, rng), random_strategy(game, rng)

    def run(seed):
        r = np.random.default_rng(seed)
        return [play_hand(game, t % 2, agent, opp, r) for t in range(40)]

    a, b, c = run(7), run(7), run(8)
    for (ha, da), (hb, db) in zip(a, b, strict=True):
        assert ha == hb
        assert [(d.infoset_key, d.taken, d.legal.tolist()) for d in da] == [
            (d.infoset_key, d.taken, d.legal.tolist()) for d in db
        ]
    assert any(ha != hc for (ha, _), (hc, _) in zip(a, c, strict=True))


def test_separate_streams_give_every_agent_the_same_deals():
    game = KuhnPoker()
    rng = np.random.default_rng(4)
    opp = random_strategy(game, rng)
    strategies = [random_strategy(game, rng), random_strategy(game, rng), uniform_strategy(game)]
    deals = []
    for strat in strategies:
        deal_rng, opp_rng, act_rng = np.random.default_rng(11).spawn(3)
        cards = []
        for t in range(60):
            hand, _ = play_hand(
                game, t % 2, strat, opp, deal_rng, agent_rng=act_rng, opp_rng=opp_rng
            )
            cards.append(hand.my_cards)
        deals.append(cards)
    assert deals[0] == deals[1] == deals[2]
    # with a single shared stream the agents' different action counts shift the deals
    shared = []
    for strat in strategies[:2]:
        r = np.random.default_rng(11)
        shared.append([play_hand(game, t % 2, strat, opp, r)[0].my_cards for t in range(60)])
    assert shared[0] != shared[1]


@pytest.mark.parametrize("game", [KuhnPoker(), LeducPoker()], ids=["kuhn", "leduc"])
@pytest.mark.parametrize("seat", [0, 1])
def test_seat_infosets_with_prefixes_cover_the_seat(game, seat):
    table = seat_infosets_with_prefixes(game, seat)
    keys = [k for k, _, _ in table]
    assert keys == seat_infosets(game, seat)
    assert set(keys) == {k for k in enumerate_infosets(game) if seat_of_key(k) == seat}
    assert len(set(keys)) == len(keys)
    for key, my_cards, events in table:
        my = my_cards[0]
        board = [ev[1] for ev in events if ev[0] == BOARD]
        other = next(c for c in range(game.spec.n_cards) if c != my and c not in board)
        cards0, cards1 = ((my,), (other,)) if seat == 0 else ((other,), (my,))
        s = game.state_from_deal(cards0, cards1, tuple(board))
        for ev in events:
            if ev[0] == BOARD:
                ((s, _),) = game.chance_outcomes(s)
                assert game.board_cards(s)[-1] == ev[1]
            else:
                _, actor, a = ev
                assert game.current_player(s) == (seat if actor == 0 else 1 - seat)
                s = game.apply(s, a)
        assert not game.is_chance(s) and not game.is_terminal(s)
        assert game.current_player(s) == seat
        assert game.infoset_key(s, seat) == key
    # fresh copies every call: mutating the result must not leak into the cache
    table[0][2].append(("act", 9, 9))
    assert seat_infosets_with_prefixes(game, seat)[0][2] == table[0][2][:-1]
    with pytest.raises(ValueError):
        seat_infosets_with_prefixes(game, 2)


def test_kuhn_prefixes_match_the_spec_examples():
    game = KuhnPoker()
    by_key = {k: (cards, events) for k, cards, events in seat_infosets_with_prefixes(game, 0)}
    assert by_key["0:K|"] == ((2,), [])
    assert by_key["0:Q|cb"] == ((1,), [(ACT, 0, CALL), (ACT, 1, RAISE)])
    by_key = {k: (cards, events) for k, cards, events in seat_infosets_with_prefixes(game, 1)}
    assert by_key["1:J|c"] == ((0,), [(ACT, 1, CALL)])
    assert by_key["1:K|b"] == ((2,), [(ACT, 1, RAISE)])


def test_sampled_action_frequencies_match_the_strategy():
    game = KuhnPoker()
    rng = np.random.default_rng(5)
    agent = random_strategy(game, rng)
    opp = uniform_strategy(game)
    root_raises = np.zeros(3)
    root_hands = np.zeros(3)
    opp_bets = 0
    opp_chances = 0
    for _ in range(6000):
        hand, decisions = play_hand(game, 0, agent, opp, rng)
        c = hand.my_cards[0]
        root_hands[c] += 1
        root_raises[c] += decisions[0].taken == RAISE
        if decisions[0].taken == CALL:  # the opponent (seat 1) then acts after a check
            opp_chances += 1
            opp_bets += hand.events[1][2] == RAISE
    for c, name in enumerate("JQK"):
        p = agent[f"0:{name}|"][RAISE]
        n = root_hands[c]
        tol = 4 * np.sqrt(max(p * (1 - p), 1e-3) / n) + 0.005
        assert abs(root_raises[c] / n - p) < tol, (name, root_raises[c] / n, p)
    assert abs(opp_bets / opp_chances - 0.5) < 4 * np.sqrt(0.25 / opp_chances) + 0.005


@pytest.mark.parametrize("game", [KuhnPoker(), LeducPoker()], ids=["kuhn", "leduc"])
def test_decisions_align_one_to_one_with_me_tokens(game):
    from exsolver.data.tokenizer import Tokenizer

    tok = Tokenizer(game.spec)
    rng = np.random.default_rng(6)
    agent = random_strategy(game, rng)
    opp = random_strategy(game, rng)
    n_decisions = 0
    for t in range(300):
        hand, decisions = play_hand(game, t % 2, agent, opp, rng)
        ids = tok.encode_hand(hand)
        me = [tok.action_of_token(x) for x in ids if tok.is_me_action(x)]
        assert me == [d.taken for d in decisions]
        stream = np.asarray(tok.encode_session([hand]))
        pos = tok.decision_positions(stream)  # positions whose *next* token is ME_*
        assert len(pos) == len(decisions)
        assert [tok.action_of_token(stream[p + 1]) for p in pos] == [d.taken for d in decisions]
        for d in decisions:
            assert d.legal[d.taken]
        n_decisions += len(decisions)
    assert n_decisions > 300
