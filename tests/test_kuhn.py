import numpy as np
import pytest

from exsolver.games import CALL, FOLD, RAISE, GameSpec, KuhnPoker
from exsolver.games.tree import compile_tree
from exsolver.strategy import enumerate_infosets, seat_of_key, uniform_strategy
from tests.engine_helpers import walk

J, Q, K = 0, 1, 2
LEGAL = {"": [CALL, RAISE], "c": [CALL, RAISE], "b": [FOLD, CALL], "cb": [FOLD, CALL]}
PLAYER = {"": 0, "c": 1, "b": 1, "cb": 0}
TERMINAL = {"cc", "bc", "cbc", "bf", "cbf"}


@pytest.fixture(scope="module")
def game() -> KuhnPoker:
    return KuhnPoker()


def test_spec(game):
    assert game.spec == GameSpec(name="kuhn", n_cards=3, n_rounds=1, max_result=2, ante=1)
    assert [game.card_name(c) for c in range(3)] == ["J", "Q", "K"]


def test_root_is_chance_with_six_uniform_deals(game):
    root = game.root()
    assert game.is_chance(root) and not game.is_terminal(root)
    outcomes = game.chance_outcomes(root)
    assert len(outcomes) == 6
    assert all(p == pytest.approx(1 / 6) for _, p in outcomes)
    deals = {(s[0], s[1]) for s, _ in outcomes}
    assert deals == {(a, b) for a in range(3) for b in range(3) if a != b}
    assert all(not game.is_chance(s) for s, _ in outcomes)


def test_legal_actions_and_player_at_every_node(game):
    seen = set()
    for s in walk(game):
        if game.is_chance(s) or game.is_terminal(s):
            continue
        h = game.history(s)
        seen.add(h)
        assert game.legal_actions(s) == LEGAL[h]
        assert game.current_player(s) == PLAYER[h]
        assert not game.is_terminal(s)
    assert seen == set(LEGAL)


def test_terminals_are_zero_sum_with_expected_payoffs(game):
    histories = set()
    for s in walk(game):
        if not game.is_terminal(s):
            continue
        r = game.returns(s)
        histories.add(game.history(s))
        assert r[0] + r[1] == 0
        assert abs(r[0]) <= game.spec.max_result
        assert abs(r[0]) in (1, 2)
    assert histories == TERMINAL
    assert game.returns((K, J, "bc")) == (2, -2)
    assert game.returns((Q, K, "cbc")) == (-2, 2)
    assert game.returns((J, K, "cc")) == (-1, 1)
    assert game.returns((J, K, "bf")) == (1, -1)
    assert game.returns((K, J, "cbf")) == (-1, 1)
    assert max(abs(game.returns(s)[0]) for s in walk(game) if game.is_terminal(s)) == 2


def test_infoset_keys(game):
    s = game.state_from_deal((K,), (Q,))
    assert game.infoset_key(s, 0) == "0:K|"
    assert game.infoset_key(s, 1) == "1:Q|"
    s = game.apply(s, CALL)
    assert game.infoset_key(s, 1) == "1:Q|c"
    s = game.apply(s, RAISE)
    assert game.infoset_key(s, 0) == "0:K|cb"
    s = game.state_from_deal((Q,), (J,))
    assert game.infoset_key(game.apply(game.apply(s, CALL), RAISE), 0) == "0:Q|cb"
    assert game.infoset_key(game.apply(s, RAISE), 1) == "1:J|b"
    s = game.state_from_deal((J,), (K,))
    assert game.infoset_key(game.apply(s, CALL), 1) == "1:K|c"
    assert game.infoset_key(game.apply(s, RAISE), 1) == "1:K|b"


def test_private_and_board_cards(game):
    s = game.state_from_deal((Q,), (K,))
    assert game.private_cards(s, 0) == (Q,)
    assert game.private_cards(s, 1) == (K,)
    assert game.board_cards(s) == ()
    with pytest.raises(ValueError):
        game.state_from_deal((Q,), (K,), board=(J,))


def test_enumerate_infosets(game):
    infosets = enumerate_infosets(game)
    expected = {
        "0:J|": [CALL, RAISE], "0:Q|": [CALL, RAISE], "0:K|": [CALL, RAISE],
        "0:J|cb": [FOLD, CALL], "0:Q|cb": [FOLD, CALL], "0:K|cb": [FOLD, CALL],
        "1:J|c": [CALL, RAISE], "1:Q|c": [CALL, RAISE], "1:K|c": [CALL, RAISE],
        "1:J|b": [FOLD, CALL], "1:Q|b": [FOLD, CALL], "1:K|b": [FOLD, CALL],
    }  # fmt: skip
    assert infosets == expected
    assert list(infosets) == list(enumerate_infosets(game))  # stable order
    assert all(seat_of_key(k) == int(k[0]) for k in infosets)
    assert compile_tree(game).n_infosets == 12
    uni = uniform_strategy(game)
    assert set(uni) == set(expected)
    assert all(uni[k].sum() == pytest.approx(1.0) for k in uni)


def test_deal_and_state_from_deal(game):
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = game.deal(rng)
        assert s[0] != s[1] and s[2] == "" and not game.is_chance(s)
    s = game.deal(rng, {0: (K,)})
    assert s[0] == K and s[1] in (J, Q)
    s = game.deal(rng, {1: (J,)})
    assert s[1] == J and s[0] in (Q, K)
    assert game.deal(rng, {0: (J,), 1: (Q,)}) == (J, Q, "")
    assert game.deal(np.random.default_rng(3)) == game.deal(np.random.default_rng(3))
    assert game.sample_chance(game.root(), np.random.default_rng(1))[2] == ""
    assert game.state_from_deal((J,), (K,)) == (J, K, "")
    with pytest.raises(ValueError):
        game.state_from_deal((J,), (J,))
    with pytest.raises(ValueError):
        game.apply(game.state_from_deal((J,), (K,)), FOLD)


def test_sample_chance(game):
    root = game.root()
    outcomes = {s for s, _ in game.chance_outcomes(root)}
    rng = np.random.default_rng(0)
    draws = {game.sample_chance(root, rng) for _ in range(200)}
    assert draws <= outcomes and len(draws) == 6  # every deal shows up, nothing else does
    assert game.sample_chance(root, np.random.default_rng(4)) == game.sample_chance(
        root, np.random.default_rng(4)
    )
    dealt = game.state_from_deal((J,), (K,))
    for s in (dealt, game.apply(dealt, RAISE), game.apply(game.apply(dealt, RAISE), FOLD)):
        with pytest.raises(ValueError):
            game.sample_chance(s, rng)


def test_states_are_small_hashable_tuples(game):
    s = game.state_from_deal((J,), (K,))
    assert isinstance(s, tuple) and len(s) == 3
    assert hash(s) == hash((J, K, ""))
    assert KuhnPoker() == game and hash(KuhnPoker()) == hash(game)
    assert compile_tree(KuhnPoker()) is compile_tree(game)
