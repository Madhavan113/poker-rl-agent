"""run_session exactness, posterior / belief bookkeeping, probes, aggregate, plots."""

import json
import math

import numpy as np
import pytest

from exsolver.eval.aggregate import (
    Thresholds,
    aggregate,
    mean_se,
    pair_average,
    paired_difference,
    resolve_agent,
)
from exsolver.eval.plots import agent_styles, write_all_plots, write_summary_md
from exsolver.eval.session import (
    KUHN_PROBES,
    FixedStrategyAgent,
    entropy,
    kl_divergence,
    run_session,
)
from exsolver.games import CALL, RAISE, KuhnPoker
from exsolver.population import KuhnPrior, nash_theta, sample_population, theta_to_profile
from exsolver.solvers import best_response, best_response_value, cfr_plus
from exsolver.strategy import merge_seats, seat_entries, uniform_strategy
from tests.engine_helpers import random_strategy

V0 = -1.0 / 18.0


class DummyPosterior:
    """Deterministic stand-in for the exact posterior: entropy drops by 0.1 nats per update."""

    def __init__(self, m: int) -> None:
        self.m = m
        self.n = 0
        self.seen = []
        self.probs = np.full(m, 1.0 / m)

    def update(self, hand) -> None:
        self.n += 1
        self.seen.append(hand)
        self.probs = _probs_after(self.m, self.n)

    def entropy(self) -> float:
        return math.log(self.m) - 0.1 * self.n


def _probs_after(m: int, n: int) -> np.ndarray:
    p = np.exp(-0.3 * n * np.arange(m))
    return p / p.sum()


class BeliefAgent(FixedStrategyAgent):
    """Fixed strategy whose belief mirrors ``DummyPosterior`` (optionally blurred)."""

    def __init__(self, strategy, name, m, blur=0.0):
        super().__init__(strategy, name)
        self.m, self.blur = m, blur

    def reset(self, rng):
        super().reset(rng)
        self.n = 0

    def observe(self, hand):
        super().observe(hand)
        self.n += 1

    def belief(self, ctx):
        assert len(ctx.hands) == self.n
        p = _probs_after(self.m, self.n)
        return (1 - self.blur) * p + self.blur / self.m


@pytest.fixture(scope="module")
def kuhn():
    return KuhnPoker()


@pytest.fixture(scope="module")
def kuhn_eq(kuhn):
    return cfr_plus(kuhn, 2000)


def test_nash_vs_nash_ev_is_the_game_value_and_exploitability_zero(kuhn, kuhn_eq):
    agent = FixedStrategyAgent(theta_to_profile(nash_theta(0.2)), "Nash")
    opp = theta_to_profile(nash_theta(0.1))
    m = run_session(kuhn, agent, opp, 10, np.random.default_rng(0), equilibrium_value_seat0=V0)
    assert m.H == 10 and m.agent == "Nash"
    np.testing.assert_array_equal(m.seats, [t % 2 for t in range(10)])
    np.testing.assert_allclose(m.ev[0::2], -1 / 18, atol=1e-9)
    np.testing.assert_allclose(m.ev[1::2], +1 / 18, atol=1e-9)
    np.testing.assert_allclose(m.expl, 0.0, atol=1e-9)
    assert set(np.unique(m.realized)) <= {-2.0, -1.0, 1.0, 2.0}
    assert (
        np.all(np.isnan(m.post_entropy))
        and np.all(np.isnan(m.agent_entropy))
        and np.all(np.isnan(m.kl))
    )
    assert set(m.probes) == set(KUHN_PROBES)
    assert len(agent.observed) == 10
    # the CFR+ average strategy (2000 iterations) satisfies the spec's sanity check
    m2 = run_session(
        kuhn,
        FixedStrategyAgent(kuhn_eq, "Equilibrium"),
        opp,
        8,
        np.random.default_rng(0),
        equilibrium_value_seat0=V0,
    )
    np.testing.assert_allclose(m2.ev[0::2], -1 / 18, atol=1e-4)
    np.testing.assert_allclose(m2.ev[1::2], +1 / 18, atol=1e-4)
    assert np.all(m2.expl < 1e-3)


def test_best_response_agent_ev_equals_best_response_value(kuhn):
    pop = sample_population(KuhnPrior(), 3, np.random.default_rng(5))
    for opp in pop.profiles:
        br0, v0 = best_response(kuhn, opp, 0)
        br1, v1 = best_response(kuhn, opp, 1)
        agent = FixedStrategyAgent(merge_seats(br0, br1), "OracleBR")
        m = run_session(kuhn, agent, opp, 6, np.random.default_rng(1), equilibrium_value_seat0=V0)
        np.testing.assert_allclose(m.ev[0::2], v0, atol=1e-12)
        np.testing.assert_allclose(m.ev[1::2], v1, atol=1e-12)
        assert v0 == pytest.approx(best_response_value(kuhn, opp, 0))
        assert v1 == pytest.approx(best_response_value(kuhn, opp, 1))
        assert np.all(m.expl > 0)  # a pure best response is exploitable


def test_posterior_gets_exactly_h_updates_and_entropy_is_recorded_before_each_hand(kuhn, kuhn_eq):
    post = DummyPosterior(7)
    agent = FixedStrategyAgent(kuhn_eq, "Equilibrium")
    H = 9
    m = run_session(
        kuhn,
        agent,
        kuhn_eq,
        H,
        np.random.default_rng(2),
        posterior=post,
        equilibrium_value_seat0=V0,
    )
    assert post.n == H
    np.testing.assert_allclose(m.post_entropy, [math.log(7) - 0.1 * t for t in range(H)])
    assert post.seen == agent.observed  # the evaluator and the agent saw the same hands, in order
    assert np.all(np.isnan(m.kl))  # the fixed agent has no belief


def test_kl_is_zero_for_identical_beliefs_and_positive_otherwise(kuhn, kuhn_eq):
    exact = BeliefAgent(kuhn_eq, "Exact", 5)
    m = run_session(
        kuhn,
        exact,
        kuhn_eq,
        8,
        np.random.default_rng(3),
        posterior=DummyPosterior(5),
        equilibrium_value_seat0=V0,
    )
    np.testing.assert_allclose(m.kl, 0.0, atol=1e-12)
    ent = [entropy(_probs_after(5, t)) for t in range(8)]
    np.testing.assert_allclose(m.agent_entropy, ent)
    assert m.agent_entropy[0] == pytest.approx(math.log(5))
    blurred = BeliefAgent(kuhn_eq, "Blurred", 5, blur=0.3)
    m2 = run_session(
        kuhn,
        blurred,
        kuhn_eq,
        8,
        np.random.default_rng(3),
        posterior=DummyPosterior(5),
        equilibrium_value_seat0=V0,
    )
    assert m2.kl[0] == pytest.approx(0.0, abs=1e-12)  # both uniform before the first hand
    assert np.all(m2.kl[1:] > 0)
    # guards
    assert kl_divergence([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert 0 < kl_divergence([0.5, 0.5], [1.0, 0.0]) < 100  # zero in q is floored, not inf
    assert entropy([0.25] * 4) == pytest.approx(math.log(4))
    with pytest.raises(ValueError):
        kl_divergence([1.0], [0.5, 0.5])


def test_probes_are_nan_on_the_wrong_seat(kuhn):
    rng = np.random.default_rng(4)
    sigma = random_strategy(kuhn, rng)
    agent = FixedStrategyAgent(sigma, "Fixed")
    m = run_session(
        kuhn,
        agent,
        sigma,
        6,
        rng,
        equilibrium_value_seat0=V0,
        probe_keys={"b0": ("0:J|", RAISE), "c1": ("1:Q|b", CALL)},
    )
    np.testing.assert_allclose(m.probes["b0"][0::2], sigma["0:J|"][RAISE])
    assert np.all(np.isnan(m.probes["b0"][1::2]))
    np.testing.assert_allclose(m.probes["c1"][1::2], sigma["1:Q|b"][CALL])
    assert np.all(np.isnan(m.probes["c1"][0::2]))
    # default Kuhn probes carry one key per seat, so they are defined at every hand
    m = run_session(kuhn, agent, sigma, 6, rng, equilibrium_value_seat0=V0)
    np.testing.assert_allclose(m.probes["bluff"][0::2], sigma["0:J|"][RAISE])
    np.testing.assert_allclose(m.probes["bluff"][1::2], sigma["1:J|c"][RAISE])
    np.testing.assert_allclose(m.probes["calldown"][0::2], sigma["0:Q|cb"][CALL])
    np.testing.assert_allclose(m.probes["calldown"][1::2], sigma["1:Q|b"][CALL])
    assert (
        run_session(kuhn, agent, sigma, 2, rng, equilibrium_value_seat0=V0, probe_keys={}).probes
        == {}
    )
    with pytest.raises(ValueError):
        run_session(
            kuhn,
            agent,
            sigma,
            2,
            rng,
            equilibrium_value_seat0=V0,
            probe_keys={"x": (("0:J|", RAISE), ("0:Q|", RAISE))},
        )


def test_run_session_rejects_incomplete_or_illegal_strategies(kuhn, kuhn_eq):
    half = FixedStrategyAgent(seat_entries(kuhn_eq, 0), "HalfAgent")
    with pytest.raises(KeyError, match="HalfAgent"):
        run_session(kuhn, half, kuhn_eq, 2, np.random.default_rng(0), equilibrium_value_seat0=V0)
    bad = dict(kuhn_eq)
    bad["0:J|"] = np.array([0.5, 0.5, 0.0])
    with pytest.raises(ValueError, match="hand 0"):
        run_session(
            kuhn,
            FixedStrategyAgent(bad, "Bad"),
            kuhn_eq,
            2,
            np.random.default_rng(0),
            equilibrium_value_seat0=V0,
        )
    with pytest.raises(ValueError):
        run_session(
            kuhn,
            FixedStrategyAgent(kuhn_eq),
            kuhn_eq,
            0,
            np.random.default_rng(0),
            equilibrium_value_seat0=V0,
        )


def test_run_session_is_deterministic_and_deals_are_shared_across_agents(kuhn, kuhn_eq):
    opp = random_strategy(kuhn, np.random.default_rng(6))
    a1 = FixedStrategyAgent(kuhn_eq, "Eq")
    m1 = run_session(kuhn, a1, opp, 12, np.random.default_rng(9), equilibrium_value_seat0=V0)
    a2 = FixedStrategyAgent(kuhn_eq, "Eq")
    m2 = run_session(kuhn, a2, opp, 12, np.random.default_rng(9), equilibrium_value_seat0=V0)
    np.testing.assert_array_equal(m1.realized, m2.realized)
    assert [h.events for h in a1.observed] == [h.events for h in a2.observed]
    a3 = FixedStrategyAgent(uniform_strategy(kuhn), "Uniform")
    run_session(kuhn, a3, opp, 12, np.random.default_rng(9), equilibrium_value_seat0=V0)
    assert [h.my_cards for h in a3.observed] == [h.my_cards for h in a1.observed]


def _fake_eval(kuhn, kuhn_eq, H=12, n_opp=3, k=2, with_transformer=False):
    pop = sample_population(KuhnPrior(), n_opp, np.random.default_rng(8))
    nash = theta_to_profile(nash_theta(0.25))
    metrics = []
    for opp_id, opp in enumerate(pop.profiles):
        br0, _ = best_response(kuhn, opp, 0)
        br1, _ = best_response(kuhn, opp, 1)
        agents = {
            "OracleBR": FixedStrategyAgent(merge_seats(br0, br1), "OracleBR"),
            "Equilibrium": FixedStrategyAgent(nash, "Equilibrium"),
            "Random": FixedStrategyAgent(uniform_strategy(kuhn), "Random"),
        }
        if with_transformer:
            agents["Transformer(sample)"] = BeliefAgent(
                merge_seats(br0, br1), "Transformer(sample)", n_opp, blur=0.05
            )
            agents["BayesBR"] = FixedStrategyAgent(kuhn_eq, "BayesBR")
        for s in range(k):
            for agent in agents.values():
                metrics.append(
                    run_session(
                        kuhn,
                        agent,
                        opp,
                        H,
                        np.random.default_rng([1, opp_id, s]),
                        posterior=DummyPosterior(n_opp),
                        equilibrium_value_seat0=V0,
                        opp_id=opp_id,
                        archetype=int(pop.archetypes[opp_id]),
                        session=s,
                    )
                )
    return pop, metrics


def test_aggregate_shapes_standard_errors_and_regret(kuhn, kuhn_eq):
    H, n_opp, k = 12, 3, 2
    pop, metrics = _fake_eval(kuhn, kuhn_eq, H, n_opp, k)
    agg = aggregate(metrics, reference="OracleBR", archetype_names=pop.archetype_names)
    assert agg["H"] == H and agg["n_opponents"] == n_opp and agg["n_sessions_per_opponent"] == k
    assert agg["agents"] == ["OracleBR", "Equilibrium", "Random"] and agg["reference"] == "OracleBR"
    for name in agg["agents"]:
        block = agg["per_agent"][name]
        assert block["n_sessions"] == n_opp * k
        for metric in ("ev", "realized", "expl", "post_entropy", "agent_entropy", "kl"):
            c = block["metrics"][metric]
            assert len(c["mean"]) == len(c["se"]) == len(c["n"]) == H
        assert len(block["pairs"]["ev"]["mean"]) == H // 2
        assert len(block["ev_cumulative"]["mean"]) == H
        assert (
            block["regret_cumulative"] is not None
            and block["regret_cumulative"]["n_paired"] == n_opp * k
        )
    # standard error formula, checked on the Random agent's EV at every hand
    ev = np.stack([m.ev for m in metrics if m.agent == "Random"])
    c = agg["per_agent"]["Random"]["metrics"]["ev"]
    np.testing.assert_allclose(c["mean"], ev.mean(axis=0))
    np.testing.assert_allclose(c["se"], ev.std(axis=0, ddof=1) / np.sqrt(ev.shape[0]))
    # pair averages combine consecutive hands, removing the seat alternation
    pair = agg["per_agent"]["Random"]["pairs"]["ev"]["mean"]
    np.testing.assert_allclose(pair, ev.reshape(ev.shape[0], H // 2, 2).mean(axis=2).mean(axis=0))
    # cumulative regret versus the oracle is non-negative and zero for the oracle itself
    for name in ("Equilibrium", "Random"):
        reg = np.asarray(agg["per_agent"][name]["regret_cumulative"]["mean"])
        assert np.all(reg >= -1e-12) and np.all(np.diff(reg) >= -1e-12)
        assert agg["per_agent"][name]["summary"]["regret_final"]["mean"] >= 0
    np.testing.assert_allclose(
        agg["per_agent"]["OracleBR"]["regret_cumulative"]["mean"], 0.0, atol=1e-12
    )
    diff = paired_difference(
        [m for m in metrics if m.agent == "OracleBR"], [m for m in metrics if m.agent == "Random"]
    )
    assert diff.shape == (n_opp * k, H) and np.all(diff >= -1e-12)
    # criteria: sanity checks pass, hypothesis checks are n/a without a transformer
    crit = agg["criteria"]
    assert crit["sanity_oracle_dominates"]["pass"] is True
    assert crit["sanity_equilibrium_exploitability"]["pass"] is True
    assert crit["H1_kl"]["pass"] is None and crit["H2_within_bayes"]["pass"] is None
    # by-archetype blocks partition the sessions
    by = agg["per_agent"]["Random"]["by_archetype"]
    assert sum(b["n_sessions"] for b in by.values()) == n_opp * k
    assert set(by) <= set(pop.archetype_names)
    # summary entries
    s = agg["per_agent"]["Equilibrium"]["summary"]
    assert s["ev_all"]["mean"] == pytest.approx(
        np.mean([m.ev.mean() for m in metrics if m.agent == "Equilibrium"])
    )
    assert s["expl_mean"]["mean"] == pytest.approx(0.0, abs=1e-9)
    json.dumps(agg)  # serialisable


def test_aggregate_with_transformer_evaluates_hypotheses(kuhn, kuhn_eq):
    pop, metrics = _fake_eval(kuhn, kuhn_eq, H=12, with_transformer=True)
    agg = aggregate(
        metrics, archetype_names=pop.archetype_names, thresholds=Thresholds(h1_t=6, h2_t_from=4)
    )
    crit = agg["criteria"]
    assert crit["resolved_agents"] == {
        "transformer": "Transformer(sample)",
        "bayes": "BayesBR",
        "equilibrium": "Equilibrium",
        "oracle": "OracleBR",
    }
    assert crit["H1_kl"]["pass"] is not None and crit["H1_kl"]["t"] == 6
    assert crit["H1_entropy_tracking"]["pass"] is not None
    assert crit["H2_within_bayes"]["pass"] is True  # the fake transformer plays the oracle BR
    assert (
        crit["H2_beats_equilibrium"]["pass"] is not None
        and crit["H2_beats_equilibrium"]["t_from"] == 4
    )
    assert crit["sanity_oracle_dominates"]["pass"] is True
    agg_short = aggregate(
        metrics[:], archetype_names=pop.archetype_names
    )  # spec thresholds exceed H=12
    assert (
        agg_short["criteria"]["H1_kl"]["scaled_to_H"] is True
        and agg_short["criteria"]["H1_kl"]["t"] == 11
    )


def test_mean_se_pair_average_and_name_resolution():
    x = np.array([[1.0, np.nan, 3.0], [3.0, np.nan, 5.0], [np.nan, np.nan, 7.0]])
    m, s, n = mean_se(x)
    np.testing.assert_allclose(m, [2.0, np.nan, 5.0])
    np.testing.assert_allclose(s, [np.sqrt(2.0) / np.sqrt(2), np.nan, 2.0 / np.sqrt(3)])
    assert n.tolist() == [2, 0, 3]
    np.testing.assert_allclose(pair_average(np.array([[1.0, 3.0, 5.0, 7.0, 9.0]])), [[2.0, 6.0]])
    names = ["OracleBR", "Transformer(sample)", "Transformer(argmax)", "bayes_br"]
    assert resolve_agent(names, "Transformer(sample)") == "Transformer(sample)"
    assert resolve_agent(names, "transformer sample") == "Transformer(sample)"
    assert resolve_agent(names, "BayesBR") == "bayes_br"  # underscores are ignored
    assert resolve_agent(["oracle_br", "x"], "OracleBR") == "oracle_br"
    assert resolve_agent(names, "bayes br") == "bayes_br"
    assert resolve_agent(names, "transformer") is None  # ambiguous
    with pytest.raises(ValueError):
        aggregate([])


def test_plots_and_summary_are_written(kuhn, kuhn_eq, tmp_path):
    pop, metrics = _fake_eval(kuhn, kuhn_eq, H=8, with_transformer=True)
    agg = aggregate(metrics, archetype_names=pop.archetype_names)
    agg["n_population"] = len(pop)
    paths = write_all_plots(agg, tmp_path)
    assert sorted(p.name for p in paths) == sorted(
        [
            "ev_vs_hand.png",
            "entropy_vs_hand.png",
            "exploitability_vs_hand.png",
            "regret_cumulative.png",
            "probes.png",
            "kl_policy_vs_hand.png",
        ]
    )
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000
    md = write_summary_md(agg, tmp_path / "summary.md", extra={"command": "test"})
    text = md.read_text()
    for name in agg["agents"]:
        assert name in text
    assert "PASS" in text and "## Success criteria" in text and "command" in text
    # figures are regenerable from the JSON form of the aggregate
    agg2 = json.loads(json.dumps(agg))
    write_all_plots(agg2, tmp_path / "again")
    assert (tmp_path / "again" / "ev_vs_hand.png").exists()
    # colour follows the entity: the transformer keeps slot 1 whatever else is drawn
    a = agent_styles(["Transformer(sample)", "OracleBR"])
    b = agent_styles(["Random", "BayesBR", "Transformer(sample)"])
    assert a["Transformer(sample)"].color == b["Transformer(sample)"].color
    assert a["OracleBR"].band is False


class BadBeliefAgent(FixedStrategyAgent):
    def __init__(self, strategy, belief):
        super().__init__(strategy, "Bad")
        self._belief = belief

    def belief(self, ctx):
        return self._belief


@pytest.mark.parametrize(
    "bad, msg",
    [
        ([np.nan, 0.5, 0.5], "finite"),
        ([np.inf, 0.0, 0.0], "finite"),
        ([-0.1, 0.6, 0.5], "negative"),
        ([0.5, 0.4, 0.05], "sums to"),
        ([0.5, 0.5], "length"),
        ([[0.5, 0.5, 0.0]], "1-D"),
    ],
)
def test_run_session_rejects_invalid_beliefs(kuhn, kuhn_eq, bad, msg):
    agent = BadBeliefAgent(kuhn_eq, np.array(bad))
    with pytest.raises(ValueError, match=msg):
        run_session(
            kuhn, agent, kuhn_eq, 2, np.random.default_rng(0),
            posterior=DummyPosterior(3), equilibrium_value_seat0=V0,
        )  # fmt: skip
    # a valid belief passes, and without an exact posterior only the length check is waived
    ok = BadBeliefAgent(kuhn_eq, np.array([0.2, 0.3, 0.5]))
    m = run_session(
        kuhn,
        ok,
        kuhn_eq,
        2,
        np.random.default_rng(0),
        posterior=DummyPosterior(3),
        equilibrium_value_seat0=V0,
    )
    assert np.all(np.isfinite(m.kl))
    two = BadBeliefAgent(kuhn_eq, np.array([0.5, 0.5]))
    m = run_session(kuhn, two, kuhn_eq, 2, np.random.default_rng(0), equilibrium_value_seat0=V0)
    assert m.agent_entropy[0] == pytest.approx(math.log(2)) and np.all(np.isnan(m.kl))
    with pytest.raises(ValueError, match="sums to"):
        run_session(
            kuhn,
            BadBeliefAgent(kuhn_eq, np.array([0.5, 0.5 + 2e-6])),
            kuhn_eq,
            1,
            np.random.default_rng(0),
            equilibrium_value_seat0=V0,
        )


def test_kl_floor_caps_zero_mass_misses():
    from exsolver.eval.session import BELIEF_FLOOR, BELIEF_FLOOR_CAP_NATS

    assert BELIEF_FLOOR == 1e-30 and BELIEF_FLOOR_CAP_NATS == pytest.approx(69.0776, abs=1e-3)
    assert kl_divergence([1.0, 0.0], [0.0, 1.0]) == pytest.approx(BELIEF_FLOOR_CAP_NATS)
    assert kl_divergence([0.5, 0.5], [0.0, 1.0]) == pytest.approx(
        0.5 * BELIEF_FLOOR_CAP_NATS + math.log(0.5)
    )


def test_paired_difference_requires_identical_session_keys(kuhn, kuhn_eq):
    pop, metrics = _fake_eval(kuhn, kuhn_eq, H=6)
    oracle = [m for m in metrics if m.agent == "OracleBR"]
    rnd = [m for m in metrics if m.agent == "Random"]
    assert paired_difference(oracle, rnd).shape == (len(oracle), 6)
    with pytest.raises(ValueError, match="paired"):
        paired_difference(oracle, rnd[:-1])
    with pytest.raises(ValueError, match="paired"):
        aggregate(metrics[:-1], archetype_names=pop.archetype_names)  # one agent misses a session


def test_summary_notes_and_strict_json_nulls(kuhn, kuhn_eq, tmp_path):
    pop, metrics = _fake_eval(kuhn, kuhn_eq, H=6, with_transformer=True)
    agg = aggregate(metrics, archetype_names=pop.archetype_names)
    text = write_summary_md(agg, tmp_path / "s.md").read_text()
    assert "before" in text and "last hand" in text and "1e-30" in text and "69.08" in text
    assert "pointwise" in text
    # plots accept strict JSON (nulls for undefined values) as produced by the runner
    as_json = json.loads(json.dumps(agg).replace("NaN", "null"))
    write_all_plots(as_json, tmp_path / "plots")
    assert (tmp_path / "plots" / "entropy_vs_hand.png").exists()


class ChangingBeliefAgent(FixedStrategyAgent):
    """Belief length changes between hands (a bug a real agent could have)."""

    def __init__(self, strategy):
        super().__init__(strategy, "Changing")
        self.calls = 0

    def belief(self, ctx):
        self.calls += 1
        return np.array([0.5, 0.5]) if self.calls == 1 else np.array([0.2, 0.3, 0.5])


def test_belief_length_is_checked_without_a_posterior(kuhn, kuhn_eq):
    two = BadBeliefAgent(kuhn_eq, np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="length 2, expected M=3"):
        run_session(
            kuhn, two, kuhn_eq, 2, np.random.default_rng(0), equilibrium_value_seat0=V0, n_opp=3
        )
    m = run_session(
        kuhn, two, kuhn_eq, 2, np.random.default_rng(0), equilibrium_value_seat0=V0, n_opp=2
    )
    assert np.all(np.isfinite(m.agent_entropy))
    with pytest.raises(ValueError, match="expected M=2"):  # pinned by the first belief
        run_session(
            kuhn,
            ChangingBeliefAgent(kuhn_eq),
            kuhn_eq,
            3,
            np.random.default_rng(0),
            equilibrium_value_seat0=V0,
        )
    with pytest.raises(ValueError, match="n_opp"):  # evaluator misconfiguration is loud too
        run_session(
            kuhn,
            two,
            kuhn_eq,
            1,
            np.random.default_rng(0),
            posterior=DummyPosterior(3),
            equilibrium_value_seat0=V0,
            n_opp=2,
        )


def test_entropy_and_kl_validate_their_inputs():
    with pytest.raises(ValueError, match="finite"):
        entropy([0.5, np.nan])
    with pytest.raises(ValueError, match="non-negative"):
        entropy([1.5, -0.5])
    with pytest.raises(ValueError, match="finite"):
        kl_divergence([np.nan, 1.0], [0.5, 0.5])
    with pytest.raises(ValueError, match="non-negative"):
        kl_divergence([0.5, 0.5], [-0.1, 1.1])
    with pytest.raises(ValueError, match="finite"):
        kl_divergence([0.5, 0.5], [np.inf, 0.0])
    assert entropy([1.0, 0.0]) == 0.0 and kl_divergence([0.5, 0.5], [0.5, 0.5]) == 0.0


# ---------------------------------------------------------------------- E2: policy-level KL
from exsolver.eval.aggregate import (  # noqa: E402
    E2Roles,
    E2Thresholds,
    check_e2_criteria,
    group_by_agent,
    tag_condition,
)
from exsolver.eval.reference import BRActionReference, infoset_reach  # noqa: E402
from exsolver.eval.session import (  # noqa: E402
    BELIEF_FLOOR_CAP_NATS,
    PURE_STRATEGY_REASON,
    SessionMetrics,
    kl_rows,
    policy_kl,
)
from exsolver.games.tree import compile_tree  # noqa: E402
from tests.test_agents import OPP_A, OPP_B  # noqa: E402


class FrozenPosterior:
    """A posterior that never moves: pi*_t is then constant across the session."""

    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype=np.float64)

    def update(self, hand):
        pass

    def entropy(self):
        return entropy(self.probs)


def _pi_star_profile(ref: BRActionReference, probs) -> dict:
    return {**ref.action_posterior_strategy(probs, 0), **ref.action_posterior_strategy(probs, 1)}


def test_infoset_reach_hand_computed(kuhn):
    """Seat 0 checks with probability q_c, the opponent bets after a check with probability b.

    reach(0:c|)   = 1/3                         (two opponent cards, 1/6 each)
    reach(0:c|cb) = sum_{c' != c} 1/6 * q_c * b = q_c * b / 3
    """
    tree = compile_tree(kuhn)
    q = {"J": 0.9, "Q": 0.4, "K": 0.0}
    b = 0.5
    sigma = np.zeros((tree.n_infosets, 3))
    keys = tree.infoset_keys
    for c, qc in q.items():
        sigma[keys.index(f"0:{c}|")] = [0.0, qc, 1.0 - qc]
        sigma[keys.index(f"0:{c}|cb")] = [0.5, 0.5, 0.0]  # irrelevant for seat-0 reach
        sigma[keys.index(f"1:{c}|c")] = [0.0, 1.0 - b, b]
        sigma[keys.index(f"1:{c}|b")] = [0.5, 0.5, 0.0]
    reach = infoset_reach(tree, sigma, 0)
    got = {keys[i]: reach[j] for j, i in enumerate(tree.infoset_rows[0])}
    for c, qc in q.items():
        assert got[f"0:{c}|"] == pytest.approx(1 / 3)
        assert got[f"0:{c}|cb"] == pytest.approx(qc * b / 3)
    assert got["0:K|cb"] == 0.0  # K never checks: unreachable
    # seat 1: 1:c|c reached when seat 0 checks holding c' != c; 1:c|b when it bets
    reach1 = infoset_reach(tree, sigma, 1)
    got1 = {keys[i]: reach1[j] for j, i in enumerate(tree.infoset_rows[1])}
    assert got1["1:J|c"] == pytest.approx((q["Q"] + q["K"]) / 6)
    assert got1["1:J|b"] == pytest.approx((2 - q["Q"] - q["K"]) / 6)
    assert sum(got1[f"1:{c}|c"] + got1[f"1:{c}|b"] for c in "JQK") == pytest.approx(1.0)


def test_reference_weights_sum_to_one_and_unreachable_infosets_weigh_zero(kuhn):
    ref = BRActionReference(kuhn, [OPP_A, OPP_B])
    keys = ref.tree.infoset_keys
    # against A alone the opponent never bets after a check: every 0:c|cb is unreachable
    pi, w = ref.policy_target(0, [1.0, 0.0])
    assert w.sum() == pytest.approx(1.0) and np.all(w >= 0)
    names = [keys[i] for i in ref.tree.infoset_rows[0]]
    for name, wi in zip(names, w, strict=True):
        if name.endswith("cb"):
            assert wi == 0.0, name
        else:
            assert wi == pytest.approx(1 / 3), name  # the three root infosets share the mass
    # 50/50 posterior: mixture bets with J and K after a check w.p. 1/2; pi* checks with J and Q
    pi, w = ref.policy_target(0, [0.5, 0.5])
    by = dict(zip(names, w, strict=True))
    raw = {"0:J|": 1 / 3, "0:Q|": 1 / 3, "0:K|": 1 / 3}
    # holding J the opponent has Q (never bets) or K (bets w.p. 1/2): reach = 1/6 * 1 * 1/2
    raw["0:J|cb"] = 1 / 12
    # holding Q the opponent has J or K, each betting w.p. 1/2: reach = 2 * 1/6 * 1/2
    raw["0:Q|cb"] = 1 / 6
    raw["0:K|cb"] = 0.0  # pi* bets with K
    total = sum(raw.values())
    for name, value in raw.items():
        assert by[name] == pytest.approx(value / total), name
    assert w.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(pi[names.index("0:J|cb")], [0.5, 0.5, 0.0])
    with pytest.raises(ValueError, match="population has 2"):
        ref.policy_target(0, [1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        ref.policy_target(2, [0.5, 0.5])


def test_policy_kl_is_zero_for_pi_star_and_nan_for_pure_strategies(kuhn):
    ref = BRActionReference(kuhn, [OPP_A, OPP_B])
    probs = np.array([0.5, 0.5])
    agent = FixedStrategyAgent(_pi_star_profile(ref, probs), "PiStar")
    H = 8
    m = run_session(
        kuhn,
        agent,
        OPP_B,
        H,
        np.random.default_rng(0),
        posterior=FrozenPosterior(probs),
        equilibrium_value_seat0=V0,
        reference=ref,
    )
    # seat 0 (even hands): pi* is mixed at 0:J|cb, the agent plays it exactly -> KL 0; seat 1
    # (odd hands): the two best responses coincide, pi* is pure there and so is the agent's
    # strategy -> NaN by the pure-strategy rule (counted in n_pure_hands)
    np.testing.assert_array_equal(m.kl_policy[0::2], 0.0)
    np.testing.assert_array_equal(m.kl_policy_unweighted[0::2], 0.0)
    assert np.all(np.isnan(m.kl_policy[1::2])) and np.all(np.isnan(m.kl_policy_unweighted[1::2]))
    assert m.n_pure_hands == H // 2 and m.H == H
    # a pure strategy (the oracle BR to A) is NaN at every hand, with the documented reason
    pure = FixedStrategyAgent(
        merge_seats(best_response(kuhn, OPP_A, 0)[0], best_response(kuhn, OPP_A, 1)[0]), "Pure"
    )
    m2 = run_session(
        kuhn,
        pure,
        OPP_B,
        H,
        np.random.default_rng(0),
        posterior=FrozenPosterior(probs),
        equilibrium_value_seat0=V0,
        reference=ref,
    )
    assert np.all(np.isnan(m2.kl_policy)) and np.all(np.isnan(m2.kl_policy_unweighted))
    assert m2.n_pure_hands == H and "pure" in PURE_STRATEGY_REASON
    # the floor cap the pure strategy would have hit, computed on the rows directly: at 0:J|cb
    # pi* = [0.5, 0.5, 0] and the pure BR(A) plays CALL, so the FOLD half is charged the cap:
    # 0.5 (log 0.5 - log 1e-30) + 0.5 (log 0.5 - log 1) = 0.5 * 69.078 + log 0.5 = 33.846 nats
    pi, w = ref.policy_target(0, probs)
    rows = ref.tree.dense_from_strategy(pure.strategy, (0,))[ref.tree.infoset_rows[0]]
    kl = kl_rows(pi, rows)
    j = [ref.tree.infoset_keys[i] for i in ref.tree.infoset_rows[0]].index("0:J|cb")
    assert kl[j] == pytest.approx(0.5 * BELIEF_FLOOR_CAP_NATS + math.log(0.5))
    assert kl[j] == pytest.approx(33.8456, abs=1e-4)
    assert np.all(kl[np.arange(kl.size) != j] == 0.0)  # BR(A) agrees with pi* elsewhere
    assert policy_kl(pi, w, rows) == (pytest.approx(math.nan, nan_ok=True),) * 2
    # a mixed strategy that is wrong is charged a finite, positive amount
    uni = FixedStrategyAgent(uniform_strategy(kuhn), "Uniform")
    m3 = run_session(
        kuhn,
        uni,
        OPP_B,
        H,
        np.random.default_rng(0),
        posterior=FrozenPosterior(probs),
        equilibrium_value_seat0=V0,
        reference=ref,
    )
    assert np.all(m3.kl_policy > 0) and np.all(m3.kl_policy < 1.0) and m3.n_pure_hands == 0
    # weighted by hand at seat 0: pi* is one-hot except at 0:J|cb -> KL(one-hot || uniform) = log 2
    # at every root infoset (weight 1/3 each before renormalisation) and 0.5 log 0.5 - ... = 0 at
    # 0:J|cb (pi* = uniform there), reach(0:Q|cb) = 1/6 with KL log 2
    rows_u = ref.tree.dense_from_strategy(uniform_strategy(kuhn), (0,))[ref.tree.infoset_rows[0]]
    names = [ref.tree.infoset_keys[i] for i in ref.tree.infoset_rows[0]]
    per = dict(zip(names, kl_rows(pi, rows_u), strict=True))
    assert per["0:J|cb"] == pytest.approx(0.0)
    for name in ("0:J|", "0:Q|", "0:K|", "0:Q|cb", "0:K|cb"):
        assert per[name] == pytest.approx(math.log(2)), name
    total = 1 + 1 / 12 + 1 / 6
    want = (3 * (1 / 3) * math.log(2) + (1 / 6) * math.log(2)) / total
    assert m3.kl_policy[0] == pytest.approx(want)
    assert m3.kl_policy_unweighted[0] == pytest.approx(5 * math.log(2) / 6)
    # a reference without a posterior is refused; the reference is never given to the agent
    with pytest.raises(ValueError, match="posterior"):
        run_session(
            kuhn, uni, OPP_B, 2, np.random.default_rng(0), equilibrium_value_seat0=V0, reference=ref
        )
    d = m3.to_dict()
    assert len(d["kl_policy"]) == H and d["n_pure_hands"] == 0


def test_kl_rows_and_policy_kl_validate_inputs():
    with pytest.raises(ValueError, match="shape"):
        kl_rows([[0.5, 0.5, 0.0]], [[0.5, 0.5]])
    with pytest.raises(ValueError, match="finite"):
        kl_rows([[0.5, 0.5, 0.0]], [[np.nan, 0.5, 0.5]])
    np.testing.assert_allclose(
        kl_rows([[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]), [BELIEF_FLOOR_CAP_NATS]
    )
    pi = np.array([[0.5, 0.5, 0.0]])
    with pytest.raises(ValueError, match="sum to 1"):
        policy_kl(pi, [1.0], [[0.6, 0.6, 0.0]])
    with pytest.raises(ValueError, match="weights sum"):
        policy_kl(pi, [0.5], [[0.5, 0.5, 0.0]])
    with pytest.raises(ValueError, match="expected 1 weights"):
        policy_kl(pi, [0.5, 0.5], [[0.5, 0.5, 0.0]])
    assert policy_kl(pi, [1.0], [[0.5, 0.5, 0.0]]) == (0.0, 0.0)
    w, u = policy_kl(pi, [1.0], [[0.25, 0.75, 0.0]])
    assert w == u == pytest.approx(0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75))


def _metrics(agent, opp_id, session, H, *, ev, kl=None, kl_policy=None) -> SessionMetrics:
    nan = np.full(H, np.nan)
    return SessionMetrics(
        agent=agent,
        opp_id=opp_id,
        archetype=0,
        session=session,
        seats=np.arange(H) % 2,
        ev=np.asarray(ev, dtype=np.float64) * np.ones(H),
        realized=nan.copy(),
        expl=nan.copy(),
        post_entropy=nan.copy(),
        agent_entropy=nan.copy(),
        kl=nan.copy() if kl is None else np.asarray(kl, dtype=np.float64) * np.ones(H),
        kl_policy=nan.copy()
        if kl_policy is None
        else np.asarray(kl_policy, dtype=np.float64) * np.ones(H),
    )


def test_e2_criteria_semantics_on_hand_built_sessions():
    H, n = 24, 6
    klp = np.full(H, 0.2)
    klp[16:] = 0.02  # below 0.05 from t = 16 on
    groups = {}
    for name, ev, kl, kp in (
        ("Thompson", 0.10, None, None),
        ("PluralityBR", 0.11, None, None),
        ("BayesBR", 0.13, None, None),
        ("Transformer(sample)", 0.095, None, klp),  # 0.005 under Thompson: within 0.01
        ("Transformer(argmax)", 0.09, None, None),  # 0.02 under PluralityBR: not within 0.01
    ):
        groups[name] = [_metrics(name, o, 0, H, ev=ev, kl=kl, kl_policy=kp) for o in range(n)]
    crit = check_e2_criteria(groups, H)
    a, b, c = (
        crit["E2-1a_kl_policy"],
        crit["E2-1b_sample_within_thompson"],
        crit["E2-1c_argmax_within_plurality"],
    )
    assert a["pass"] is True and a["t"] == 16 and a["max_from_t"] == pytest.approx(0.02)
    assert a["value_at_t"] == pytest.approx(0.02) and a["n_sessions_at_t"] == n
    assert b["pass"] is True and b["max_shortfall"] == pytest.approx(0.005) and b["n_paired"] == n
    assert c["pass"] is False and c["max_shortfall"] == pytest.approx(0.02)
    assert "E2-2" not in " ".join(crit) and crit["thresholds"]["kl_policy"] == 0.05
    # a stricter threshold flips E2-1b; being better than the reference never fails (one-sided)
    assert (
        check_e2_criteria(groups, H, thresholds=E2Thresholds(match_gap=0.001))[
            "E2-1b_sample_within_thompson"
        ]["pass"]
        is False
    )
    groups["Transformer(sample)"] = [
        _metrics("Transformer(sample)", o, 0, H, ev=0.5, kl_policy=klp) for o in range(n)
    ]
    assert check_e2_criteria(groups, H)["E2-1b_sample_within_thompson"]["pass"] is True
    # kl_policy above the threshold at one late hand fails pointwise
    bad = klp.copy()
    bad[20] = 0.06
    groups["Transformer(sample)"] = [
        _metrics("Transformer(sample)", o, 0, H, ev=0.1, kl_policy=bad) for o in range(n)
    ]
    a = check_e2_criteria(groups, H)["E2-1a_kl_policy"]
    assert a["pass"] is False and a["t_of_max"] == 20
    # pure agents have no kl_policy: the check says so instead of passing vacuously
    groups["Transformer(sample)"] = [
        _metrics("Transformer(sample)", o, 0, H, ev=0.1) for o in range(n)
    ]
    assert check_e2_criteria(groups, H)["E2-1a_kl_policy"]["pass"] is None
    assert "no kl_policy" in check_e2_criteria(groups, H)["E2-1a_kl_policy"]["note"]
    # thresholds beyond H are clipped and flagged
    short = {
        k: [_metrics(k, o, 0, 8, ev=0.1, kl_policy=0.01) for o in range(n)]
        for k in ("Thompson", "Transformer(sample)")
    }
    a = check_e2_criteria(short, 8)["E2-1a_kl_policy"]
    assert a["t"] == 7 and a["scaled_to_H"] is True and a["pass"] is True
    # unpaired sessions raise instead of silently dropping
    groups["Thompson"] = groups["Thompson"][:-1]
    with pytest.raises(ValueError, match="paired"):
        check_e2_criteria(groups, H)


def test_e2_criteria_across_conditions_with_pending_ones():
    H, n = 20, 5
    refs = {}
    for name, ev in (("Thompson", 0.10), ("PluralityBR", 0.11), ("BayesBR", 0.13)):
        refs[name] = [_metrics(name, o, 0, H, ev=ev) for o in range(n)]
    ident_a = np.linspace(2.0, 1.0, H)
    groups = dict(refs)

    def add(cond, ev_sample, ev_argmax, kl_identity, kl_policy=0.01):
        for mode, ev in (("sample", ev_sample), ("argmax", ev_argmax)):
            name = tag_condition(f"Transformer({mode})", cond)
            groups[name] = [
                _metrics(
                    name,
                    o,
                    0,
                    H,
                    ev=ev,
                    kl=kl_identity,
                    kl_policy=kl_policy if mode == "sample" else None,
                )
                for o in range(n)
            ]

    add("A", 0.095, 0.105, ident_a)
    add("B", 0.096, 0.106, ident_a - 0.1)  # identity KL lower at every hand
    add("C", 0.095 + 0.005, 0.105 - 0.005, ident_a)  # EV change 0.005 < 0.01
    add("E", 0.115, 0.125, ident_a)  # 0.015 under BayesBR: within 0.02
    conditions = {
        "A": "evaluated",
        "B": "evaluated",
        "C": "evaluated",
        "D": "pending",
        "E": "evaluated",
    }
    crit = check_e2_criteria(groups, H, conditions=conditions, roles=E2Roles())
    assert crit["E2-1a_kl_policy[A]"]["pass"] is True
    assert crit["E2-1b_sample_within_thompson[A]"]["pass"] is True
    assert crit["E2-1c_argmax_within_plurality[A]"]["pass"] is True
    assert crit["E2-2_sample_within_bayes[E]"]["pass"] is True
    assert crit["E2-2_sample_within_bayes[E]"]["max_shortfall"] == pytest.approx(0.015)
    assert crit["E2-2_argmax_within_bayes[E]"]["pass"] is True
    e3a = crit["E2-3a_identity_kl[B]<[A]"]
    assert (
        e3a["pass"] is True
        and e3a["min_margin"] == pytest.approx(0.1)
        and e3a["n_hands_compared"] == H
    )
    e3c = crit["E2-3b_ev_change[C]vs[A]"]
    assert e3c["pass"] is True and e3c["max_abs_change_sample"] == pytest.approx(0.005)
    assert e3c["max_abs_change_argmax"] == pytest.approx(0.005) and e3c[
        "mean_change_argmax"
    ] == pytest.approx(-0.005)
    e3d = crit["E2-3b_ev_change[D]vs[A]"]
    assert (
        e3d["pass"] is None and "pending" in e3d["note"] and "Transformer(sample)[D]" in e3d["note"]
    )
    assert crit["conditions"] == conditions and crit["roles"]["baseline"] == "A"
    # B not lower at one hand -> fails, naming the hand; C changing by 0.02 -> fails
    kl_b = ident_a - 0.1
    kl_b[5] = ident_a[5]
    add("B", 0.096, 0.106, kl_b)
    add("C", 0.095 + 0.02, 0.105, ident_a)
    crit = check_e2_criteria(groups, H, conditions=conditions)
    assert (
        crit["E2-3a_identity_kl[B]<[A]"]["pass"] is False
        and crit["E2-3a_identity_kl[B]<[A]"]["t_of_min"] == 5
    )
    assert crit["E2-3a_identity_kl[B]<[A]"]["n_hands_failing"] == 1
    assert crit["E2-3b_ev_change[C]vs[A]"]["pass"] is False
    # a missing (never listed) condition is reported as absent, not pending
    without_e = {k: v for k, v in groups.items() if not k.endswith("[E]")}
    crit = check_e2_criteria(without_e, H, conditions={"A": "evaluated"})
    assert crit["E2-2_sample_within_bayes[E]"]["pass"] is None
    assert "absent" in crit["E2-2_sample_within_bayes[E]"]["note"]
    # grouping helper keeps first-appearance order
    flat = [m for g in groups.values() for m in g]
    assert list(group_by_agent(flat)) == list(groups)
    # renders (with descriptions) and serialises
    from exsolver.eval.plots import criteria_table

    text = criteria_table({"criteria_e2": crit}, "criteria_e2")
    assert "E2-1a_kl_policy[A]" in text and "PASS" in text and "n/a" in text
    json.dumps(crit)


def test_kl_policy_plot_and_summary_note(kuhn, tmp_path):
    ref = BRActionReference(kuhn, [OPP_A, OPP_B])
    probs = np.array([0.5, 0.5])
    metrics = []
    for opp_id, opp in enumerate((OPP_A, OPP_B)):
        for agent in (
            FixedStrategyAgent(uniform_strategy(kuhn), "Random"),
            FixedStrategyAgent(_pi_star_profile(ref, probs), "Transformer(sample)"),
            FixedStrategyAgent(
                merge_seats(best_response(kuhn, opp, 0)[0], best_response(kuhn, opp, 1)[0]),
                "OracleBR",
            ),
        ):
            metrics.append(
                run_session(
                    kuhn, agent, opp, 6, np.random.default_rng([opp_id]), posterior=FrozenPosterior(probs),
                    equilibrium_value_seat0=V0, reference=ref, opp_id=opp_id, session=0,
                )
            )  # fmt: skip
    agg = aggregate(metrics)
    blk = agg["per_agent"]
    assert all(v is None or math.isnan(v) for v in blk["OracleBR"]["metrics"]["kl_policy"]["mean"])
    assert blk["OracleBR"]["summary"]["pure_hands_share"] == 1.0
    ts = blk["Transformer(sample)"]["metrics"]["kl_policy"]["mean"]
    assert ts[0::2] == [0.0] * 3 and all(math.isnan(v) for v in ts[1::2])  # seat 1: pi* pure
    assert blk["Random"]["metrics"]["kl_policy"]["n"] == [2] * 6
    assert set(blk["Random"]["summary"]["kl_policy_at"]) == {"5"}  # 16 / 32 / 63 clipped to H - 1
    paths = write_all_plots(agg, tmp_path)
    kp = tmp_path / "kl_policy_vs_hand.png"
    assert kp in paths and kp.stat().st_size > 1000
    text = write_summary_md(agg, tmp_path / "s.md").read_text()
    assert "Policy KL" in text and "mean policy KL" in text and "pure" in text
    # figures also draw from strict JSON with nulls
    write_all_plots(json.loads(json.dumps(agg).replace("NaN", "null")), tmp_path / "again")
    assert (tmp_path / "again" / "kl_policy_vs_hand.png").exists()
