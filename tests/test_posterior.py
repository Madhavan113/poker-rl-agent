"""Exact posterior: bluffer identification, brute-force likelihoods, normalisation, -inf handling."""

import warnings

import numpy as np
import pytest

from exsolver.bayes import (
    ExactPosterior,
    hand_log_likelihoods,
    logsumexp,
    population_array,
    replay_hand,
)
from exsolver.data.records import BOARD, CALL, FOLD, RAISE, HandRecord, act_event, board_event
from exsolver.games import KuhnPoker, LeducPoker
from exsolver.games.tree import compile_tree
from exsolver.population import (
    KUHN_PARAM_NAMES,
    KuhnPrior,
    Population,
    sample_population,
    theta_to_profile,
)
from exsolver.strategy import strategy_to_array
from tests.engine_helpers import pure_strategy, random_strategy

J, Q, K = 0, 1, 2


def kuhn_theta(overrides: dict[str, float]) -> np.ndarray:
    """Theta with 0.5 everywhere except the given ``KUHN_PARAM_NAMES`` entries."""
    theta = np.full(12, 0.5)
    for name, value in overrides.items():
        theta[KUHN_PARAM_NAMES.index(name)] = value
    return theta


def naive_log_likelihoods(game, profiles, hand: HandRecord) -> np.ndarray:
    """Reference implementation: enumerate the deals through the ``Game`` API, one opponent at a time.

    ``L = sum_c P(c | my cards, board) * [c == revealed if shown] * prod_k theta(a_k | I_k(c))``.
    """
    agent, opp = hand.seat, 1 - hand.seat
    out = []
    for prof in profiles:
        total_w, lik = 0.0, 0.0
        for s0, p_deal in game.chance_outcomes(game.root()):
            if tuple(game.private_cards(s0, agent)) != tuple(hand.my_cards):
                continue
            s, w, prob, ok = s0, p_deal, 1.0, True
            for ev in hand.events:
                if ev[0] == BOARD:
                    match = [
                        (s2, p2)
                        for s2, p2 in game.chance_outcomes(s)
                        if game.board_cards(s2)[-1] == ev[1]
                    ]
                    if not match:
                        ok = False  # this candidate holds the board card
                        break
                    s, w = match[0][0], w * match[0][1]
                else:
                    _, actor, a = ev
                    if actor == 1:
                        prob *= prof[game.infoset_key(s, opp)][a]
                    s = game.apply(s, a)
            if not ok:
                continue
            total_w += w
            if hand.opp_cards is None or tuple(game.private_cards(s0, opp)) == tuple(
                hand.opp_cards
            ):
                lik += w * prob
        out.append(np.log(lik / total_w) if lik > 0 else -np.inf)
    return np.array(out)


def simulate_hand(game, rng, agent_seat, agent, opp) -> HandRecord:
    """Play one hand with the engine; reveal the opponent's cards unless somebody folded."""
    s = game.deal(rng)
    events, folded = [], False
    while not game.is_terminal(s):
        if game.is_chance(s):
            before = len(game.board_cards(s))
            s = game.sample_chance(s, rng)
            events.extend(board_event(c) for c in game.board_cards(s)[before:])
            continue
        p = game.current_player(s)
        row = (agent if p == agent_seat else opp)[game.infoset_key(s, p)]
        a = int(rng.choice(3, p=row / row.sum()))
        folded = folded or a == FOLD
        events.append(act_event(0 if p == agent_seat else 1, a))
        s = game.apply(s, a)
    opp_cards = None if folded else tuple(game.private_cards(s, 1 - agent_seat))
    return HandRecord(
        agent_seat,
        tuple(game.private_cards(s, agent_seat)),
        events,
        opp_cards,
        int(round(game.returns(s)[agent_seat])),
    )


@pytest.fixture(scope="module")
def kuhn() -> KuhnPoker:
    return KuhnPoker()


def test_showdown_identifies_the_bluffer(kuhn):
    never = kuhn_theta({"0:J|": 0.0, "1:J|c": 0.0})
    always = kuhn_theta({"0:J|": 1.0, "1:J|c": 1.0})
    pop = Population.from_thetas(np.stack([never, always]), np.array([0, 0]))
    post = ExactPosterior(kuhn, pop)
    np.testing.assert_allclose(post.probs, [0.5, 0.5])
    # agent at seat 1 holds K; the opponent (seat 0) bets, we call, the showdown reveals J
    post.update(HandRecord(1, (K,), [act_event(1, RAISE), act_event(0, CALL)], (J,), 2))
    np.testing.assert_array_equal(post.probs, [0.0, 1.0])
    assert post.entropy() == 0.0 and post.map_id() == 1
    # seat 0: we check, the opponent bets after the check, we call, J is shown
    post.reset()
    post.update(
        HandRecord(0, (K,), [act_event(0, CALL), act_event(1, RAISE), act_event(0, CALL)], (J,), 2)
    )
    np.testing.assert_array_equal(post.probs, [0.0, 1.0])
    # a J that checks behind is impossible for the always-bluffer
    post.reset()
    post.update(HandRecord(0, (K,), [act_event(0, CALL), act_event(1, CALL)], (J,), 1))
    np.testing.assert_array_equal(post.probs, [1.0, 0.0])
    # no showdown: J is one of two candidates (the other, K, bets w.p. 0.5 for both) -> no collapse
    post.reset()
    post.update(HandRecord(1, (Q,), [act_event(1, RAISE), act_event(0, FOLD)], None, -1))
    np.testing.assert_allclose(post.probs, [0.25, 0.75])
    assert 0.0 < post.entropy() < np.log(2)


@pytest.mark.parametrize("game_cls,n_hands", [(KuhnPoker, 60), (LeducPoker, 30)])
def test_matches_brute_force_enumeration(game_cls, n_hands):
    game = game_cls()
    rng = np.random.default_rng(0)
    for trial in range(4):
        profiles = [random_strategy(game, rng, concentration=0.5) for _ in range(4)]
        # pure profiles put exact zeros in the array so the -inf paths are exercised too
        profiles.append(pure_strategy(game, lambda k, legal: legal[-1]))
        profiles.append(pure_strategy(game, lambda k, legal: legal[0]))
        pop = population_array(game, profiles)
        assert pop.shape == (6, len(profiles[0]), 3)
        agent = random_strategy(game, rng)
        opp = profiles[trial % len(profiles)]
        for i in range(n_hands):
            hand = simulate_hand(game, rng, i % 2, agent, opp)
            got = hand_log_likelihoods(game, pop, hand)
            want = naive_log_likelihoods(game, profiles, hand)
            assert got.shape == (6,)
            np.testing.assert_array_equal(np.isinf(got), np.isinf(want))
            finite = np.isfinite(want)
            np.testing.assert_allclose(got[finite], want[finite], rtol=1e-12, atol=1e-12)
            assert np.isfinite(got[trial % len(profiles)])  # the true opponent can produce its hand


def test_probs_entropy_and_bookkeeping(kuhn):
    m = 12
    pop = sample_population(KuhnPrior(), m, np.random.default_rng(1))
    post = ExactPosterior(kuhn, pop)
    assert post.m == m
    assert post.entropy() == pytest.approx(np.log(m))
    assert post.probs.sum() == pytest.approx(1.0)
    rng = np.random.default_rng(2)
    theta = pop.profiles[3]
    agent = random_strategy(kuhn, rng)
    hands = [simulate_hand(kuhn, rng, t % 2, agent, theta) for t in range(40)]
    lls = np.stack([hand_log_likelihoods(kuhn, post.pop_array, h) for h in hands])
    for t, h in enumerate(hands):
        post.update(h)
        assert post.probs.sum() == pytest.approx(1.0, abs=1e-12)
        assert post.n_hands == t + 1
    joint = post.log_prior + lls.sum(axis=0)
    np.testing.assert_allclose(post.log_post, joint - logsumexp(joint), atol=1e-12)
    assert post.log_evidence == pytest.approx(float(logsumexp(joint)))
    p = post.probs
    assert post.entropy() == pytest.approx(-(p[p > 0] * np.log(p[p > 0])).sum())
    assert 0.0 <= post.entropy() <= np.log(m)
    assert post.map_id() == int(np.argmax(post.log_post))
    # copy is independent, reset returns to the prior
    twin = post.copy()
    twin.update(hands[0])
    assert twin.n_hands == post.n_hands + 1 and not np.allclose(twin.probs, post.probs)
    post.reset()
    assert post.n_hands == 0 and post.log_evidence == 0.0
    np.testing.assert_allclose(post.probs, 1.0 / m)
    # a point-mass prior has entropy exactly 0 (not -0.0) and never moves
    point = ExactPosterior(kuhn, pop, prior_weights=np.eye(m)[4])
    assert point.entropy() == 0.0 and str(point.entropy()) == "0.0" and point.map_id() == 4
    point.update_many(hands)
    np.testing.assert_array_equal(point.probs, np.eye(m)[4])
    # Population weights are the default prior; a bare list of profiles gets a uniform prior
    w = np.arange(1, m + 1, dtype=float)
    w /= w.sum()
    np.testing.assert_allclose(
        ExactPosterior(kuhn, Population.from_thetas(pop.thetas, pop.archetypes, w)).probs, w
    )
    np.testing.assert_allclose(ExactPosterior(kuhn, pop.profiles).probs, 1.0 / m)
    with pytest.raises(ValueError):
        ExactPosterior(kuhn, pop, prior_weights=np.ones(m - 1))
    with pytest.raises(ValueError):
        ExactPosterior(kuhn, pop, prior_weights=-np.ones(m))
    with pytest.raises(ValueError):
        ExactPosterior(kuhn, pop, prior_weights=np.zeros(m))


def test_impossible_opponents_get_zero_mass_without_nans(kuhn):
    always_bet = pure_strategy(kuhn, lambda k, legal: RAISE if RAISE in legal else CALL)
    always_check = pure_strategy(kuhn, lambda k, legal: CALL)
    coin = theta_to_profile(np.full(12, 0.5))
    post = ExactPosterior(kuhn, [always_bet, always_check, coin])
    # agent at seat 1 sees the opponent check: impossible for always_bet
    hand = HandRecord(1, (J,), [act_event(1, CALL), act_event(0, CALL)], (Q,), -1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no divide-by-zero / invalid-value warnings anywhere
        ll = hand_log_likelihoods(kuhn, post.pop_array, hand)
        post.update(hand)
        probs, ent = post.probs, post.entropy()
    assert ll[0] == -np.inf and np.isfinite(ll[1:]).all()
    assert probs[0] == 0.0 and np.isfinite(probs).all() and probs.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(probs, [0.0, 2 / 3, 1 / 3])  # check w.p. 1 versus w.p. 0.5
    assert np.isfinite(ent) and post.log_post[0] == -np.inf
    # a hand no remaining opponent can produce is a loud error that leaves the state untouched
    fold_or_check = pure_strategy(kuhn, lambda k, legal: legal[0])
    stuck = ExactPosterior(kuhn, [always_bet, fold_or_check])
    before = stuck.log_post
    impossible = HandRecord(
        1, (K,), [act_event(1, CALL), act_event(0, RAISE), act_event(1, CALL)], (Q,), 2
    )
    with pytest.raises(ValueError, match="zero probability"):
        stuck.update(impossible)
    np.testing.assert_array_equal(stuck.log_post, before)
    assert stuck.n_hands == 0
    # logsumexp of an all -inf row is -inf, not NaN
    assert logsumexp(np.array([-np.inf, -np.inf])) == -np.inf
    np.testing.assert_allclose(
        logsumexp(np.array([[0.0, 0.0], [-np.inf, 0.0]]), axis=1), [np.log(2), 0.0]
    )


def test_replay_details_and_rejections(kuhn):
    # unrevealed card: both candidates, equal deal weights, shared public action sequence
    rep = replay_hand(
        kuhn,
        HandRecord(
            0, (Q,), [act_event(0, CALL), act_event(1, RAISE), act_event(0, FOLD)], None, -1
        ),
    )
    assert (
        rep.n_candidates == 2 and rep.actions.tolist() == [RAISE] and rep.infosets.shape == (2, 1)
    )
    np.testing.assert_allclose(np.exp(rep.log_weights), 0.5)
    # revealed card: one candidate that keeps its conditional deal weight
    rep = replay_hand(kuhn, HandRecord(1, (K,), [act_event(1, RAISE), act_event(0, CALL)], (J,), 2))
    assert rep.n_candidates == 1 and np.exp(rep.log_weights[0]) == pytest.approx(0.5)
    # the opponent never acted: nothing to learn, log-likelihood 0 for everybody
    hand = HandRecord(1, (K,), [act_event(1, RAISE)], None, 0)
    rep = replay_hand(kuhn, hand)
    assert rep.actions.tolist() == [RAISE]
    pop = population_array(kuhn, [theta_to_profile(np.full(12, 0.3))])
    np.testing.assert_allclose(
        hand_log_likelihoods(kuhn, pop, HandRecord(0, (K,), [act_event(0, RAISE)], None, 0)), 0.0
    )
    with pytest.raises(ValueError, match="wrong player"):
        replay_hand(kuhn, HandRecord(0, (K,), [act_event(1, RAISE)], None, 0))
    with pytest.raises(ValueError, match="inconsistent"):
        replay_hand(kuhn, HandRecord(0, (K,), [act_event(0, RAISE), act_event(1, CALL)], (K,), 2))
    with pytest.raises(ValueError):  # illegal action rejected by the engine
        replay_hand(kuhn, HandRecord(0, (K,), [act_event(0, FOLD)], None, 0))
    with pytest.raises(ValueError, match="after the hand ended"):
        replay_hand(
            kuhn,
            HandRecord(
                0, (K,), [act_event(0, RAISE), act_event(1, FOLD), act_event(0, CALL)], None, 1
            ),
        )
    with pytest.raises(ValueError, match="shape"):
        hand_log_likelihoods(kuhn, np.zeros((2, 5, 3)), hand)


def test_leduc_board_candidates_and_weights():
    game = LeducPoker()
    # agent Js checks, opp checks, board Qh, agent bets, opp folds -> 4 candidates, 1/4 each
    hand = HandRecord(
        0, (0,),
        [act_event(0, CALL), act_event(1, CALL), board_event(3), act_event(0, RAISE), act_event(1, FOLD)],
        None, 1,
    )  # fmt: skip
    rep = replay_hand(game, hand)
    assert rep.n_candidates == 4 and rep.actions.tolist() == [CALL, FOLD]
    np.testing.assert_allclose(np.exp(rep.log_weights), 0.25)
    # revealed Kh keeps its weight of 1/4
    shown = HandRecord(
        0, (0,),
        [act_event(0, CALL), act_event(1, CALL), board_event(3), act_event(0, CALL), act_event(1, CALL)],
        (5,), -1,
    )  # fmt: skip
    rep = replay_hand(game, shown)
    assert rep.n_candidates == 1 and np.exp(rep.log_weights[0]) == pytest.approx(0.25)
    with pytest.raises(ValueError):  # the board card cannot be the revealed opponent card
        replay_hand(game, HandRecord(0, (0,), shown.events, (3,), -1))


def test_likelihood_is_invariant_to_the_agents_own_strategy(kuhn):
    """RESEARCH.md 2.2: only the opponent-seat entries of theta enter the likelihood (bit-identical)."""
    m = 10
    pop = sample_population(KuhnPrior(), m, np.random.default_rng(4))
    base = population_array(kuhn, pop.profiles)
    tree = compile_tree(kuhn)
    rng = np.random.default_rng(5)
    agent = random_strategy(kuhn, rng)
    seat_params = {0: slice(0, 6), 1: slice(6, 12)}  # KUHN_PARAM_NAMES: seat-0 entries first
    for t in range(40):
        seat = t % 2
        hand = simulate_hand(kuhn, rng, seat, agent, pop.profiles[t % m])
        # dense level: replace the agent-seat rows of every member by fresh random legal rows
        rows = tree.infoset_rows[seat]
        perturbed = base.copy()
        for i in range(m):
            dense = strategy_to_array(kuhn, random_strategy(kuhn, rng, (seat,)), (seat,))
            perturbed[i, rows] = dense[rows]
        assert not np.array_equal(perturbed, base)
        np.testing.assert_array_equal(
            hand_log_likelihoods(kuhn, perturbed, hand), hand_log_likelihoods(kuhn, base, hand)
        )
        # population level: perturbing the agent-seat parameters leaves the posterior bit-identical
        thetas = pop.thetas.copy()
        thetas[:, seat_params[seat]] = rng.uniform(size=(m, 6))
        other = Population.from_thetas(thetas, pop.archetypes, pop.weights)
        a, b = ExactPosterior(kuhn, pop), ExactPosterior(kuhn, other)
        a.update(hand)
        b.update(hand)
        np.testing.assert_array_equal(a.log_post, b.log_post)
        assert a.log_evidence == b.log_evidence
