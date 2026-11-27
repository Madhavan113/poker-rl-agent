import json

import numpy as np
import pytest

from exsolver.bayes import ExactPosterior, log_likelihoods_from_replay, replay_hand, safe_log
from exsolver.bayes.online import discounted_posterior, em_fit, expected_visits, pad_replays
from exsolver.games import KuhnPoker, LeducPoker
from exsolver.games.base import FOLD, RAISE
from exsolver.games.tree import compile_tree
from exsolver.play import play_hand
from exsolver.population.adaptive import adapt, best_response_profile, mix_sequence_form
from exsolver.population.leduc_prior import MID, STRONG, WEAK, LeducPrior, strength_bucket
from exsolver.solvers import best_response_value, ev_dense
from exsolver.strategy import strategy_to_array
from tests.engine_helpers import random_strategy


@pytest.fixture(scope="module")
def leduc_prior() -> LeducPrior:
    return LeducPrior(cfr_iterations=300)


def test_strength_bucket():
    assert strength_bucket("0:Js|") == WEAK
    assert strength_bucket("1:Qh|c") == MID
    assert strength_bucket("0:Ks|cb") == STRONG
    assert strength_bucket("1:Qs|cbc/Qh|c") == STRONG  # pair
    assert strength_bucket("0:Kh|cc/Js|") == MID  # unpaired K
    assert strength_bucket("0:Qh|cc/Ks|b") == WEAK


def test_leduc_prior_profiles(leduc_prior):
    tree = leduc_prior.tree
    legal = tree.infoset_legal
    pop, ids = leduc_prior.sample_population(20, np.random.default_rng(0))
    assert pop.shape == (20, tree.n_infosets, 3) and ids.shape == (20,)
    np.testing.assert_allclose(pop.sum(-1), 1.0, atol=1e-12)
    assert np.all(pop[:, ~legal] == 0.0)
    assert np.all(pop[:, legal] > 0.0)  # smoothing: every legal action has positive probability
    # no bias, no noise: exactly the smoothed equilibrium
    zero = np.zeros((tree.n_infosets, 3))
    base = leduc_prior.dense_for(0, 1.0, zero)
    unif = legal / legal.sum(1, keepdims=True)
    np.testing.assert_allclose(base, 0.95 * leduc_prior.equilibrium + 0.05 * unif, atol=1e-12)
    # the maniac raises more than the base at every infoset where raising is legal and unforced
    maniac = leduc_prior.dense_for(leduc_prior.names.index("maniac"), 1.0, zero)
    can_raise = legal[:, RAISE] & ~legal[:, FOLD]
    assert np.all(maniac[can_raise, RAISE] > base[can_raise, RAISE])


@pytest.mark.parametrize("game", [KuhnPoker(), LeducPoker()], ids=["kuhn", "leduc"])
def test_mix_sequence_form_is_realization_equivalent(game):
    tree = compile_tree(game)
    rng = np.random.default_rng(1)
    a = np.stack([strategy_to_array(game, random_strategy(game, rng)) for _ in range(2)])
    b = np.stack([strategy_to_array(game, random_strategy(game, rng)) for _ in range(2)])
    mixed = mix_sequence_form(tree, a, b, 0.3)
    for i in range(2):
        ours = strategy_to_array(game, random_strategy(game, rng))
        for seat in (0, 1):
            want = 0.7 * ev_dense(game, seat, ours, a[i]) + 0.3 * ev_dense(game, seat, ours, b[i])
            assert ev_dense(game, seat, ours, mixed[i]) == pytest.approx(want, abs=1e-12)


def test_adapt_static_and_convergence():
    game = KuhnPoker()
    tree = compile_tree(game)
    rng = np.random.default_rng(2)
    theta = strategy_to_array(game, random_strategy(game, rng))[None]
    ours = strategy_to_array(game, random_strategy(game, rng))[None]
    np.testing.assert_array_equal(adapt(game, theta, ours, 0.0), theta)
    br = best_response_profile(game, ours)
    target = [
        best_response_value(game, tree.strategy_from_dense(ours[0], (s,)), 1 - s) for s in (0, 1)
    ]
    for s in (0, 1):  # the BR profile attains the BR value at both seats
        assert -ev_dense(game, s, ours[0], br[0]) == pytest.approx(target[s], abs=1e-12)
    gains = []
    for _ in range(40):
        theta = adapt(game, theta, ours, 0.3)
        gains.append(sum(-ev_dense(game, s, ours[0], theta[0]) for s in (0, 1)))
    assert np.all(np.diff(gains) >= -1e-12)  # the opponent's EV against us never falls
    assert gains[-1] == pytest.approx(sum(target), abs=1e-4)


def _hands(game, theta, n, seed):
    tree = compile_tree(game)
    rng = np.random.default_rng(seed)
    ours = tree.strategy_from_dense(strategy_to_array(game, random_strategy(game, rng)))
    opp = tree.strategy_from_dense(theta)
    recs = [play_hand(game, h % 2, ours, opp, rng)[0] for h in range(n)]
    return recs, [replay_hand(game, r) for r in recs]


def test_expected_visits_counts_every_opponent_action():
    game = LeducPoker()
    tree = compile_tree(game)
    theta = strategy_to_array(game, random_strategy(game, np.random.default_rng(3)))
    _, reps = _hands(game, theta, 200, seed=4)
    visits = expected_visits(theta, pad_replays(reps))
    assert visits.sum() == pytest.approx(sum(r.actions.size for r in reps), abs=1e-9)
    w = np.linspace(0.0, 1.0, 200)
    visits_w = expected_visits(theta, pad_replays(reps), w)
    assert visits_w.sum() == pytest.approx(
        sum(wi * r.actions.size for wi, r in zip(w, reps, strict=True))
    )
    assert np.all(visits[tree.infoset_rows[0]] >= 0.0)


def test_em_fit_recovers_kuhn_opponent():
    game = KuhnPoker()
    tree = compile_tree(game)
    theta = strategy_to_array(game, random_strategy(game, np.random.default_rng(5)))
    _, reps = _hands(game, theta, 20000, seed=6)
    hands = pad_replays(reps)
    legal = tree.infoset_legal.astype(float)
    ref = legal / legal.sum(1, keepdims=True)
    est = em_fit(hands, ref, strength=1.0, iterations=200)
    visits = expected_visits(est, hands)
    well = visits > 500
    assert well.sum() >= 8
    assert np.abs(est[well] - theta[well]).max() < 0.05
    # rows the data never reach stay at the reference
    untouched = visits == 0
    np.testing.assert_allclose(est[untouched], ref[untouched], atol=1e-12)


def test_discounted_posterior_matches_exact_posterior():
    game = KuhnPoker()
    rng = np.random.default_rng(7)
    pop = np.stack([strategy_to_array(game, random_strategy(game, rng)) for _ in range(5)])
    recs, reps = _hands(game, pop[2], 30, seed=8)
    ll = np.stack([log_likelihoods_from_replay(safe_log(pop), r) for r in reps])
    log_prior = np.log(np.full(5, 0.2))
    exact = ExactPosterior(game, [compile_tree(game).strategy_from_dense(p) for p in pop])
    exact.update_many(recs)
    np.testing.assert_allclose(discounted_posterior(log_prior, ll), exact.probs, atol=1e-12)
    np.testing.assert_allclose(discounted_posterior(log_prior, ll, np.zeros(30)), 0.2, atol=1e-12)


def test_e5_smoke_run(tmp_path):
    from exsolver.experiments.e5_nodelock import main

    agg = main(["--game", "kuhn", "--smoke", "--out", str(tmp_path)])
    for name in ("summary.json", "summary.md", "blocks.npz", "frontier.png", "README.md"):
        assert (tmp_path / name).exists()
    rows = json.loads((tmp_path / "summary.json").read_text())["rows"]
    assert len(rows) == len(agg["rows"]) == 44
    for r in rows:
        if r["method"] == "nash":
            assert r["gain_vs_nash"][0] == 0.0
            assert r["exploitability"][0] < 0.01
        assert np.isfinite(r["ev"][0]) and np.isfinite(r["exploitability"][0])
        assert 0.0 <= r["solver_gap_max"] < 0.1  # certified solver error, smoke-size solves
        if r["method"] in ("nash", "br"):
            assert r["solver_gap_max"] == 0.0
    # a second run with the same config resumes from the cached jobs and reproduces the summary
    again = main(["--game", "kuhn", "--smoke", "--out", str(tmp_path)])
    assert [r["ev"] for r in again["rows"]] == [r["ev"] for r in agg["rows"]]
    # block 0 is the equilibrium for every config: identical EV against the same opponents
    first = {
        r["set_name"] + str(r["eta"]): r["ev_by_block"][0] for r in rows if r["method"] == "nash"
    }
    for r in rows:
        assert r["ev_by_block"][0] == pytest.approx(first[r["set_name"] + str(r["eta"])], abs=1e-12)
