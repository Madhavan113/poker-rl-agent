import numpy as np
import pytest

from exsolver.games import CALL, FOLD, RAISE, KuhnPoker
from exsolver.population import (
    ARCHETYPE_NAMES,
    KUHN_PARAM_NAMES,
    N_PARAMS,
    KuhnPrior,
    Population,
    default_archetypes,
    nash_theta,
    profile_to_theta,
    sample_population,
    theta_to_profile,
)
from exsolver.population.kuhn_prior import PARAM_ACTIONS
from exsolver.solvers import exploitability
from exsolver.strategy import enumerate_infosets, strategy_to_array


def test_param_names_and_archetypes():
    assert KUHN_PARAM_NAMES == [
        "0:J|", "0:Q|", "0:K|", "0:J|cb", "0:Q|cb", "0:K|cb",
        "1:J|c", "1:Q|c", "1:K|c", "1:J|b", "1:Q|b", "1:K|b",
    ]  # fmt: skip
    assert N_PARAMS == 12
    assert ARCHETYPE_NAMES == ["equilibrium", "maniac", "station", "rock", "random"]
    prior = KuhnPrior()
    assert prior.names == ARCHETYPE_NAMES
    np.testing.assert_allclose(prior.weights, [0.30, 0.15, 0.20, 0.20, 0.15])
    assert [a.kappa for a in prior.archetypes] == [30.0, 10.0, 10.0, 10.0, None]
    assert set(KUHN_PARAM_NAMES) == set(enumerate_infosets(KuhnPoker()))


def test_default_archetypes_match_research_table():
    # RESEARCH.md section 3.1, typed out literally; columns in KUHN_PARAM_NAMES order:
    # 0:J| 0:Q| 0:K| P(bet) | 0:J|cb 0:Q|cb 0:K|cb P(call) | 1:J|c 1:Q|c 1:K|c P(bet) | 1:J|b 1:Q|b 1:K|b P(call)
    # fmt: off
    centres = {
        "maniac":  (0.70, 0.80, 0.95, 0.50, 0.85, 1.00, 0.70, 0.70, 0.95, 0.50, 0.90, 1.00),
        "station": (0.05, 0.15, 0.80, 0.60, 0.95, 1.00, 0.05, 0.10, 0.90, 0.60, 0.95, 1.00),
        "rock":    (0.02, 0.05, 0.90, 0.02, 0.20, 0.95, 0.02, 0.05, 0.95, 0.02, 0.15, 1.00),
    }
    weights = {"equilibrium": 0.30, "maniac": 0.15, "station": 0.20, "rock": 0.20, "random": 0.15}
    kappas = {"equilibrium": 30.0, "maniac": 10.0, "station": 10.0, "rock": 10.0, "random": None}
    # fmt: on
    archetypes = default_archetypes()
    assert [a.name for a in archetypes] == ["equilibrium", "maniac", "station", "rock", "random"]
    assert sum(weights.values()) == pytest.approx(1.0)
    for a in archetypes:
        assert a.weight == weights[a.name]
        assert a.kappa == kappas[a.name]
        if a.name in centres:
            assert a.centre == centres[a.name]
        else:
            assert a.centre is None  # equilibrium: Nash family per sample; random: Beta(1, 1)
    # equilibrium column: alpha, 0, 3alpha, 0, alpha + 1/3, 1 | 1/3, 0, 1, 0, 1/3, 1
    for alpha in (0.0, 0.1, 1 / 3):
        np.testing.assert_allclose(
            nash_theta(alpha),
            [alpha, 0.0, 3 * alpha, 0.0, alpha + 1 / 3, 1.0, 1 / 3, 0.0, 1.0, 0.0, 1 / 3, 1.0],
        )
    prior = KuhnPrior()
    assert prior.centre_clip == (0.02, 0.98) and prior.alpha_max == pytest.approx(1 / 3)


def test_theta_profile_round_trip():
    rng = np.random.default_rng(0)
    game = KuhnPoker()
    for _ in range(20):
        theta = rng.uniform(size=N_PARAMS)
        profile = theta_to_profile(theta)
        assert list(profile) == KUHN_PARAM_NAMES
        np.testing.assert_allclose(profile_to_theta(profile), theta)
        dense = strategy_to_array(game, profile)  # valid: sums to 1, zeros on illegal actions
        assert dense.shape == (12, 3)
        for name, a in zip(KUHN_PARAM_NAMES, PARAM_ACTIONS, strict=True):
            v = profile[name]
            assert v.sum() == pytest.approx(1.0)
            assert v[a] == theta[KUHN_PARAM_NAMES.index(name)]
            assert v[FOLD] == 0.0 if a == RAISE else v[RAISE] == 0.0
    nash = theta_to_profile(nash_theta(0.2))
    assert nash["0:K|"][RAISE] == pytest.approx(0.6) and nash["0:Q|cb"][CALL] == pytest.approx(
        0.2 + 1 / 3
    )
    assert exploitability(game, nash) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        theta_to_profile(np.full(N_PARAMS, 1.5))


def test_prior_samples_are_valid_and_deterministic():
    prior = KuhnPrior()
    rng = np.random.default_rng(42)
    for _ in range(200):
        profile, theta, k = prior.sample(rng)
        assert theta.shape == (N_PARAMS,)
        assert np.all(theta >= 0.0) and np.all(theta <= 1.0)
        assert 0 <= k < len(ARCHETYPE_NAMES)
        np.testing.assert_allclose(profile_to_theta(profile), theta)
        assert set(profile) == set(KUHN_PARAM_NAMES)
    a = prior.sample(np.random.default_rng(5))
    b = prior.sample(np.random.default_rng(5))
    np.testing.assert_array_equal(a[1], b[1])
    assert a[2] == b[2]


def test_archetype_frequencies_and_centres():
    prior = KuhnPrior()
    pop = sample_population(prior, 4000, np.random.default_rng(0))
    freq = np.bincount(pop.archetypes, minlength=5) / len(pop)
    np.testing.assert_allclose(freq, prior.weights, atol=0.03)
    means = {
        name: pop.thetas[pop.archetypes == k].mean(axis=0) for k, name in enumerate(ARCHETYPE_NAMES)
    }
    assert means["maniac"][0] > 0.6 and means["rock"][0] < 0.1 and means["station"][3] > 0.5
    assert abs(means["random"].mean() - 0.5) < 0.05
    # equilibrium samples are much less exploitable than maniacs
    game = KuhnPoker()
    expl = np.array([exploitability(game, p) for p in pop.profiles[:800]])
    arch = pop.archetypes[:800]
    assert expl[arch == 0].mean() < 0.5 * expl[arch == 1].mean()
    assert expl[arch == 0].mean() < 0.1


def test_population_shapes_save_load(tmp_path):
    prior = KuhnPrior()
    pop = sample_population(prior, 64, np.random.default_rng(1))
    assert len(pop) == 64 and len(pop.profiles) == 64
    assert pop.thetas.shape == (64, 12) and pop.archetypes.shape == (64,)
    assert pop.weights.shape == (64,) and pop.weights.sum() == pytest.approx(1.0)
    assert np.all(pop.weights == pop.weights[0])
    assert pop.archetype_names == ARCHETYPE_NAMES
    for profile, theta in zip(pop.profiles, pop.thetas, strict=True):
        np.testing.assert_allclose(profile_to_theta(profile), theta)
    path = tmp_path / "pop.npz"
    pop.save(path)
    loaded = Population.load(path)
    np.testing.assert_array_equal(loaded.thetas, pop.thetas)
    np.testing.assert_array_equal(loaded.archetypes, pop.archetypes)
    np.testing.assert_array_equal(loaded.weights, pop.weights)
    assert loaded.archetype_names == pop.archetype_names
    for p, q in zip(loaded.profiles, pop.profiles, strict=True):
        for k in KUHN_PARAM_NAMES:
            np.testing.assert_array_equal(p[k], q[k])
    again = sample_population(prior, 64, np.random.default_rng(1))
    np.testing.assert_array_equal(again.thetas, pop.thetas)
    with pytest.raises(ValueError):
        Population(
            profiles=pop.profiles,
            thetas=pop.thetas,
            archetypes=pop.archetypes,
            weights=pop.weights * 2,
        )
