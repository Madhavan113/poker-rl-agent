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
