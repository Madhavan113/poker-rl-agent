"""Agents: fixed baselines, exact-Bayes agents versus the oracle, transformer agent validity."""

import copy
import inspect

import numpy as np
import pytest
import torch

from exsolver.agents import (
    BayesBRAgent,
    EquilibriumAgent,
    FixedStrategyAgent,
    OracleBRAgent,
    RandomAgent,
    ThompsonAgent,
    TransformerAgent,
    check_seat,
)
from exsolver.bayes import ExactPosterior
from exsolver.data.records import SessionContext
from exsolver.data.tokenizer import Tokenizer
from exsolver.games import KuhnPoker
from exsolver.model.inference import Policy
from exsolver.model.transformer import ExploitTransformer, ModelConfig
from exsolver.play import play_hand, seat_infosets_with_prefixes
from exsolver.population import KUHN_PARAM_NAMES, KuhnPrior, sample_population, theta_to_profile
from exsolver.solvers import (
    best_response,
    best_response_value,
    expected_value,
    exploitability,
    mix_strategies,
)
from exsolver.strategy import enumerate_infosets, seat_infosets, uniform_strategy, validate_strategy

M = 12


@pytest.fixture(scope="module")
def game() -> KuhnPoker:
    return KuhnPoker()


@pytest.fixture(scope="module")
def pop():
    return sample_population(KuhnPrior(), M, np.random.default_rng(0))


@pytest.fixture(scope="module")
def equilibrium(game) -> EquilibriumAgent:
    return EquilibriumAgent(game)


def seat_ev(game, sigma, opp, seat) -> float:
    return expected_value(game, sigma, opp) if seat == 0 else -expected_value(game, opp, sigma)


def play_session(game, agent, opp, H, seed) -> SessionContext:
    """Drive ``agent`` the way the evaluator does: reset, then strategy / play / observe per hand."""
    rng = np.random.default_rng(seed)
    agent.reset(np.random.default_rng(seed + 1))
    ctx = SessionContext()
    for t in range(H):
        sigma = agent.strategy_for_hand(ctx, t % 2)
        validate_strategy(game, sigma, (t % 2,))
        hand, _ = play_hand(game, t % 2, sigma, opp, rng)
        agent.observe(hand)
        ctx.hands.append(hand)
    return ctx


def test_fixed_and_random_agents(game):
    uni = uniform_strategy(game)
    agent = FixedStrategyAgent(uni, "uni")
    ctx = SessionContext()
    for seat in (0, 1):
        st = agent.strategy_for_hand(ctx, seat)
        assert set(st) == set(seat_infosets(game, seat))
        for k, v in st.items():
            np.testing.assert_array_equal(v, uni[k])
            assert v is not uni[k]  # callers get copies
    assert agent.belief(ctx) is None and agent.name == "uni"
    agent.reset(np.random.default_rng(0))
    agent.observe(play_session(game, RandomAgent(game), uni, 1, seed=0).hands[0])
    rnd = RandomAgent(game)
    st = rnd.strategy_for_hand(ctx, 1)
    np.testing.assert_array_equal(st["1:J|b"], [0.5, 0.5, 0.0])
    np.testing.assert_array_equal(st["1:K|c"], [0.0, 0.5, 0.5])
    assert rnd.name == "random"
    with pytest.raises(ValueError):
        agent.strategy_for_hand(ctx, 2)
    with pytest.raises(ValueError):
        check_seat(-1)


def test_equilibrium_agent_is_a_nash_equilibrium(game, equilibrium):
    assert exploitability(game, equilibrium.profile) < 1e-3
    assert expected_value(game, equilibrium.profile, equilibrium.profile) == pytest.approx(
        -1 / 18, abs=1e-3
    )
    assert equilibrium.iterations == 2000 and equilibrium.name == "equilibrium"
    for seat in (0, 1):
        assert set(equilibrium.strategy_for_hand(SessionContext(), seat)) == set(
            seat_infosets(game, seat)
        )


def test_oracle_br_agent_attains_the_best_response_value(game, pop):
    ctx = SessionContext()
    for j in (0, 5, 11):
        theta = pop.profiles[j]
        oracle = OracleBRAgent(game, theta)
        assert oracle.name == "oracle_br" and oracle.belief(ctx) is None
        for seat in (0, 1):
            st = oracle.strategy_for_hand(ctx, seat)
            want = best_response_value(game, theta, seat)
            assert seat_ev(game, st, theta, seat) == pytest.approx(want, abs=1e-12)
            assert oracle.values[seat] == pytest.approx(want)


def test_point_mass_posterior_reproduces_the_oracle(game, pop):
    ctx = SessionContext()
    for j in range(0, M, 3):
        w = np.eye(M)[j]
        theta = pop.profiles[j]
        thompson = ThompsonAgent(game, pop, prior_weights=w, rng=np.random.default_rng(j))
        bayes = BayesBRAgent(game, pop, prior_weights=w)
        for seat in (0, 1):
            want = best_response_value(game, theta, seat)
            assert seat_ev(
                game, thompson.strategy_for_hand(ctx, seat), theta, seat
            ) == pytest.approx(want, abs=1e-12)
            assert seat_ev(game, bayes.strategy_for_hand(ctx, seat), theta, seat) == pytest.approx(
                want, abs=1e-12
            )
            assert bayes.last_value == pytest.approx(want, abs=1e-12)
        assert thompson.last_sample == j
        np.testing.assert_array_equal(thompson.belief(ctx), w)
        np.testing.assert_array_equal(bayes.belief(ctx), w)
        # hands generated by theta cannot move a point mass off theta
        after = play_session(game, bayes, theta, 8, seed=j)
        np.testing.assert_array_equal(bayes.belief(after), w)
        assert seat_ev(game, bayes.strategy_for_hand(after, 0), theta, 0) == pytest.approx(
            best_response_value(game, theta, 0), abs=1e-12
        )


def test_posterior_agents_track_the_exact_posterior(game, pop):
    theta = pop.profiles[7]
    bayes = BayesBRAgent(game, pop)
    ctx = play_session(game, bayes, theta, 32, seed=3)
    exact = ExactPosterior(game, pop)
    exact.update_many(ctx.hands)
    np.testing.assert_allclose(bayes.belief(ctx), exact.probs, atol=1e-12)
    # context-only path (never observed anything) and the Thompson agent agree with the exact posterior
    np.testing.assert_allclose(BayesBRAgent(game, pop).belief(ctx), exact.probs, atol=1e-12)
    thompson = ThompsonAgent(game, pop)
    thompson.reset(np.random.default_rng(0))
    np.testing.assert_allclose(thompson.belief(ctx), exact.probs, atol=1e-12)
    # a shorter context is recomputed from the prior, then the full one is caught up again
    short = SessionContext(list(ctx.hands[:10]))
    exact10 = ExactPosterior(game, pop)
    exact10.update_many(short.hands)
    np.testing.assert_allclose(bayes.belief(short), exact10.probs, atol=1e-12)
    np.testing.assert_allclose(bayes.belief(ctx), exact.probs, atol=1e-12)
    # observe() between hands is exactly the evaluator's call order and gives the same numbers
    inc = BayesBRAgent(game, pop)
    inc.reset(np.random.default_rng(0))
    for hand in ctx.hands:
        inc.observe(hand)
    np.testing.assert_allclose(inc.belief(ctx), exact.probs, atol=1e-12)
    bayes.reset(np.random.default_rng(0))
    np.testing.assert_allclose(bayes.belief(SessionContext()), 1.0 / M)


def test_thompson_samples_and_bayes_br_best_responds_to_the_mixture(game):
    maniac = theta_to_profile(
        np.array([0.9, 0.9, 0.95, 0.8, 0.9, 1.0, 0.9, 0.9, 0.95, 0.8, 0.9, 1.0])
    )
    rock = theta_to_profile(
        np.array([0.02, 0.05, 0.9, 0.02, 0.2, 0.95, 0.02, 0.05, 0.95, 0.02, 0.15, 1.0])
    )
    profiles = [maniac, rock]
    assert len(KUHN_PARAM_NAMES) == 12
    ctx = SessionContext()
    thompson = ThompsonAgent(game, profiles)
    with pytest.raises(RuntimeError, match="rng"):
        thompson.strategy_for_hand(ctx, 0)
    thompson.reset(np.random.default_rng(0))
    seen = set()
    for _ in range(50):
        st = thompson.strategy_for_hand(ctx, 0)
        i = thompson.last_sample
        seen.add(i)
        want, _ = best_response(game, profiles[i], 0)
        assert st.keys() == want.keys() and all(np.array_equal(st[k], want[k]) for k in st)
    assert seen == {0, 1}  # both opponents get sampled under a uniform posterior
    bayes = BayesBRAgent(game, profiles)
    for seat in (0, 1):
        probs = bayes.belief(ctx)
        np.testing.assert_allclose(probs, 0.5)
        mixed = mix_strategies(game, probs, profiles, 1 - seat)
        want, value = best_response(game, mixed, seat)
        got = bayes.strategy_for_hand(ctx, seat)
        assert got.keys() == want.keys() and all(np.array_equal(got[k], want[k]) for k in got)
        assert bayes.last_value == pytest.approx(value)
        # exact EV of the Bayes BR under the belief is the average over the two opponents
        avg = 0.5 * sum(seat_ev(game, got, th, seat) for th in profiles)
        assert avg == pytest.approx(value, abs=1e-12)


def test_transformer_agent_returns_valid_strategies_and_beliefs(game, pop):
    tok = Tokenizer(game.spec)
    torch.manual_seed(0)
    model = ExploitTransformer(
        ModelConfig(
            vocab_size=tok.vocab_size, n_opp=M, d_model=32, n_layers=1, n_heads=2, max_len=256
        )
    )
    policy = Policy(model, tok, device="cpu")
    ctx = play_session(game, RandomAgent(game), pop.profiles[2], 6, seed=5)
    legal_of = enumerate_infosets(game)
    for mode in ("sample", "argmax"):
        agent = TransformerAgent(game, policy, tok, mode=mode)
        assert agent.name == f"transformer_{mode}"
        agent.reset(np.random.default_rng(0))
        agent.observe(ctx.hands[0])
        for seat in (0, 1):
            st = agent.strategy_for_hand(ctx, seat)
            validate_strategy(game, st, (seat,))
            assert set(st) == set(seat_infosets(game, seat))
            views = seat_infosets_with_prefixes(game, seat)
            prefixes = [
                tok.encode_prefix(ctx.hands, seat, cards, events) for _, cards, events in views
            ]
            legal = np.array([[a in legal_of[k] for a in range(3)] for k, _, _ in views])
            ref = policy.policy_at(prefixes, legal)
            for i, (key, _, _) in enumerate(views):
                assert st[key].dtype == np.float64 and st[key].sum() == pytest.approx(
                    1.0, abs=1e-12
                )
                if mode == "sample":
                    np.testing.assert_allclose(st[key], ref[i], atol=1e-6)
                else:
                    assert st[key].sum() == 1.0 and st[key][int(np.argmax(ref[i]))] == 1.0
                    assert legal[i][int(np.argmax(st[key]))]
        for c in (ctx, SessionContext()):
            b = agent.belief(c)
            assert (
                b.shape == (M,)
                and b.dtype == np.float64
                and b.sum() == pytest.approx(1.0, abs=1e-6)
            )
            np.testing.assert_allclose(
                b, policy.belief_at([tok.encode_session(c.hands)])[0], atol=1e-7
            )
    # the counterfactual prefix of an infoset is its hand-local observation path
    agent = TransformerAgent(game, policy, tok)
    prefixes = agent.prefixes_for_hand(SessionContext(), 0)
    keys = [k for k, _, _ in seat_infosets_with_prefixes(game, 0)]
    names = [tok.name(t) for t in prefixes[keys.index("0:Q|cb")]]
    assert names == ["BOS", "HAND", "POS_0", "CARD_1", "ME_CALL", "OPP_RAISE"]
    names = [
        tok.name(t)
        for t in agent.prefixes_for_hand(SessionContext(), 1)[
            [k for k, _, _ in seat_infosets_with_prefixes(game, 1)].index("1:K|b")
        ]
    ]
    assert names == ["BOS", "HAND", "POS_1", "CARD_2", "OPP_RAISE"]
    # without an opponent head there is no belief; bad modes are rejected
    bare = Policy(
        ExploitTransformer(
            ModelConfig(vocab_size=tok.vocab_size, d_model=32, n_layers=1, n_heads=2, max_len=256)
        ),
        tok,
        device="cpu",
    )
    assert TransformerAgent(game, bare, tok).belief(ctx) is None
    with pytest.raises(ValueError):
        TransformerAgent(game, policy, tok, mode="greedy")
    # the agent has no channel for theta, the opponent id or labels
    assert set(inspect.signature(TransformerAgent).parameters) == {
        "game",
        "policy",
        "tokenizer",
        "mode",
        "name",
    }
    # a context longer than the model's window fails loudly instead of truncating
    long_ctx = play_session(game, RandomAgent(game), pop.profiles[2], 40, seed=6)
    with pytest.raises(ValueError, match="max_len"):
        agent.strategy_for_hand(long_ctx, 0)


def test_sync_recomputes_when_the_context_diverges(game, pop):
    """A context that does not extend the observed hands must not reuse the stale posterior."""
    theta = pop.profiles[1]
    h0, h1, h2, h3 = play_session(game, RandomAgent(game), theta, 4, seed=9).hands

    def exact(*hands):
        post = ExactPosterior(game, pop)
        post.update_many(hands)
        return post.probs

    for agent in (BayesBRAgent(game, pop), ThompsonAgent(game, pop)):
        agent.reset(np.random.default_rng(0))
        agent.observe(h0)
        # the failing scenario: observe h0, then ask about a context holding a different hand
        np.testing.assert_array_equal(agent.belief(SessionContext([h1])), exact(h1))
        # a context extending [h1] is caught up incrementally
        np.testing.assert_array_equal(agent.belief(SessionContext([h1, h2])), exact(h1, h2))
        assert agent.posterior.n_hands == 2
        # equal copies carry the same evidence: accepted as the same prefix, only h3 is new
        copies = SessionContext([copy.deepcopy(h1), copy.deepcopy(h2), h3])
        np.testing.assert_array_equal(agent.belief(copies), exact(h1, h2, h3))
        assert agent.posterior.n_hands == 3
        # observe, then a shorter context that differs at position 0 -> recomputed from the prior
        agent.observe(h0)
        np.testing.assert_array_equal(agent.belief(SessionContext([h0])), exact(h0))
        # same hands but fewer of them -> recomputed as well
        agent.observe(h1)
        agent.observe(h2)
        np.testing.assert_array_equal(agent.belief(SessionContext([h0, h1])), exact(h0, h1))
        # strategy_for_hand goes through the same synchronisation
        agent.strategy_for_hand(SessionContext([h2]), 0)
        np.testing.assert_array_equal(agent.belief(SessionContext([h2])), exact(h2))
        # the evaluator's order (observe before appending to ctx) is the fast path
        ctx = SessionContext([h2])
        agent.observe(h3)
        ctx.hands.append(h3)
        np.testing.assert_array_equal(agent.belief(ctx), exact(h2, h3))
        assert agent.posterior.n_hands == 2


def test_point_mass_strategies_equal_the_oracle_strategies(game, pop):
    ctx = SessionContext()
    for j in (1, 4, 10):
        w = np.eye(M)[j]
        oracle = OracleBRAgent(game, pop.profiles[j])
        bayes = BayesBRAgent(game, pop, prior_weights=w)
        thompson = ThompsonAgent(game, pop, prior_weights=w, rng=np.random.default_rng(0))
        for seat in (0, 1):
            want = oracle.strategy_for_hand(ctx, seat)
            for got in (bayes.strategy_for_hand(ctx, seat), thompson.strategy_for_hand(ctx, seat)):
                assert got.keys() == want.keys()
                for key in want:
                    np.testing.assert_array_equal(got[key], want[key], err_msg=f"{j} {key}")


def test_thompson_frequencies_follow_the_posterior_and_are_seed_deterministic(game):
    profiles = [theta_to_profile(np.full(12, 0.3)), theta_to_profile(np.full(12, 0.7))]
    prior = np.array([0.3, 0.7])
    agent = ThompsonAgent(game, profiles, prior_weights=prior, rng=np.random.default_rng(123))
    ctx = SessionContext()
    n = 20_000
    counts = np.zeros(2)
    for _ in range(n):
        agent.strategy_for_hand(ctx, 0)
        counts[agent.last_sample] += 1
    np.testing.assert_allclose(counts / n, prior, atol=0.01)
    a = ThompsonAgent(game, profiles, prior_weights=prior)
    b = ThompsonAgent(game, profiles, prior_weights=prior)
    a.reset(np.random.default_rng(7))
    b.reset(np.random.default_rng(7))
    for t in range(50):
        sa, sb = a.strategy_for_hand(ctx, t % 2), b.strategy_for_hand(ctx, t % 2)
        assert a.last_sample == b.last_sample
        assert sa.keys() == sb.keys() and all(np.array_equal(sa[k], sb[k]) for k in sa)


def test_transformer_single_pass_belief_matches_two_pass_and_is_cached(game, pop, monkeypatch):
    tok = Tokenizer(game.spec)
    torch.manual_seed(1)
    model = ExploitTransformer(
        ModelConfig(
            vocab_size=tok.vocab_size,
            n_opp=M,
            theta_dim=12,
            d_model=32,
            n_layers=1,
            n_heads=2,
            max_len=256,
        )
    )
    policy = Policy(model, tok, device="cpu")
    agent = TransformerAgent(game, policy, tok)
    ctx = play_session(game, RandomAgent(game), pop.profiles[4], 5, seed=8)
    legal_of = enumerate_infosets(game)
    calls = {"belief_at": 0, "all_at": 0}
    real_belief_at, real_all_at = policy.belief_at, policy.all_at

    def spy_belief_at(prefixes):
        calls["belief_at"] += 1
        return real_belief_at(prefixes)

    def spy_all_at(prefixes, legal=None):
        calls["all_at"] += 1
        return real_all_at(prefixes, legal)

    monkeypatch.setattr(policy, "belief_at", spy_belief_at)
    monkeypatch.setattr(policy, "all_at", spy_all_at)
    for seat in (0, 1):
        views = seat_infosets_with_prefixes(game, seat)
        prefixes = [tok.encode_prefix(ctx.hands, seat, cards, events) for _, cards, events in views]
        legal = np.array([[a in legal_of[k] for a in range(3)] for k, _, _ in views])
        two_pass_policy = policy.policy_at(prefixes, legal)
        two_pass_belief = real_belief_at([tok.encode_session(ctx.hands)])[0]
        st = agent.strategy_for_hand(ctx, seat)
        b = agent.belief(ctx)
        assert calls == {
            "belief_at": 0,
            "all_at": seat + 1,
        }  # one forward pass; belief from the cache
        np.testing.assert_allclose(b, two_pass_belief, atol=1e-6)
        assert b.shape == (M,) and b.sum() == pytest.approx(1.0, abs=1e-6)
        for i, (key, _, _) in enumerate(views):
            np.testing.assert_allclose(st[key], two_pass_policy[i], atol=1e-6)
        assert agent.belief(ctx) is not b  # callers get copies
    # a context the cache does not hold falls back to one belief_at call, then is cached
    other = SessionContext(list(ctx.hands[:3]))
    b3 = agent.belief(other)
    assert calls["belief_at"] == 1
    np.testing.assert_allclose(b3, real_belief_at([tok.encode_session(other.hands)])[0], atol=1e-7)
    agent.belief(other)
    assert calls["belief_at"] == 1
    # the empty context (BOS only) works through both paths; reset clears the cache
    agent.reset(np.random.default_rng(0))
    e = agent.belief(SessionContext())
    assert calls["belief_at"] == 2 and e.shape == (M,)
    agent.strategy_for_hand(SessionContext(), 1)
    np.testing.assert_allclose(agent.belief(SessionContext()), e, atol=1e-6)
    assert calls["belief_at"] == 2
    # collision case: same length and same last hand object but different earlier hands is a miss
    base = play_session(game, RandomAgent(game), pop.profiles[4], 4, seed=21).hands
    alt = play_session(game, RandomAgent(game), pop.profiles[6], 3, seed=22).hands
    ctx_a = SessionContext(list(base))
    ctx_b = SessionContext([alt[0], alt[1], alt[2], base[3]])
    truth_a = real_belief_at([tok.encode_session(ctx_a.hands)])[0]
    truth_b = real_belief_at([tok.encode_session(ctx_b.hands)])[0]
    assert not np.allclose(truth_a, truth_b, atol=1e-6)  # the stale vector would be wrong
    agent.strategy_for_hand(ctx_a, 0)  # caches the belief of ctx_a
    n_before = calls["belief_at"]
    np.testing.assert_allclose(agent.belief(ctx_b), truth_b, atol=1e-6)
    assert calls["belief_at"] == n_before + 1  # a genuine miss, recomputed
    np.testing.assert_allclose(
        agent.belief(ctx_a), truth_a, atol=1e-6
    )  # recomputed again, no stale hit
    assert calls["belief_at"] == n_before + 2
    # an equal copy of the cached context encodes to the same stream: a hit
    n_before = calls["belief_at"]
    np.testing.assert_allclose(
        agent.belief(SessionContext(copy.deepcopy(ctx_a.hands))), truth_a, atol=1e-6
    )
    assert calls["belief_at"] == n_before
