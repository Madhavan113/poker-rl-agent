"""Kuhn ground truth: analytic values the engine and solvers must reproduce.

These tests are written against the interfaces in docs/experiments/e1-kuhn.md, independently of
the engine's own test-suite, and take their expected values from truth/kuhn.json only.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.ground_truth

kuhn_mod = pytest.importorskip("exsolver.games.kuhn")
br_mod = pytest.importorskip("exsolver.solvers.best_response")
cfr_mod = pytest.importorskip("exsolver.solvers.cfr")
prior_mod = pytest.importorskip("exsolver.population.kuhn_prior")


@pytest.fixture(scope="module")
def game():
    return kuhn_mod.KuhnPoker()


def test_param_order_matches_truth(truth):
    assert list(prior_mod.KUHN_PARAM_NAMES) == truth("kuhn")["param_order"]


def test_infoset_count(game, truth):
    from exsolver.strategy import enumerate_infosets

    assert len(enumerate_infosets(game)) == truth("kuhn")["n_infosets"]


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_nash_family_is_unexploitable_and_has_game_value(game, truth, idx):
    data = truth("kuhn")
    entry = data["nash_family"]["thetas"][idx]
    profile = prior_mod.theta_to_profile(np.asarray(entry["theta"], dtype=float))
    tol = data["tolerances"]
    assert br_mod.exploitability(game, profile) <= tol["nash_exploitability"]
    ev = br_mod.expected_value(game, profile, profile)
    assert abs(ev - data["game_value_seat0"]) <= tol["nash_value"]


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_hand_computed_best_responses(game, truth, idx):
    data = truth("kuhn")
    case = data["hand_computed_best_responses"][idx]
    opponent = prior_mod.theta_to_profile(np.asarray(case["opponent_theta"], dtype=float))
    value = br_mod.best_response_value(game, opponent, case["br_seat"])
    assert abs(value - case["value"]) <= data["tolerances"]["hand_computed_value"], case["name"]


def test_cfr_plus_reaches_game_value(game, truth):
    data = truth("kuhn")
    strategy = cfr_mod.cfr_plus(game, data["cfr_plus"]["iterations"])
    tol = data["tolerances"]
    assert (
        abs(br_mod.expected_value(game, strategy, strategy) - data["game_value_seat0"])
        <= tol["cfr_value"]
    )
    assert br_mod.exploitability(game, strategy) <= tol["cfr_exploitability"]
