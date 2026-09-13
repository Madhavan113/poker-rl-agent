"""Leduc ground truth from OpenSpiel's reference implementation (see truth/leduc.json)."""

import pytest

pytestmark = pytest.mark.ground_truth

leduc_mod = pytest.importorskip("exsolver.games.leduc")
br_mod = pytest.importorskip("exsolver.solvers.best_response")
cfr_mod = pytest.importorskip("exsolver.solvers.cfr")


@pytest.fixture(scope="module")
def game():
    return leduc_mod.LeducPoker()


def test_infoset_count(game, truth):
    from exsolver.strategy import enumerate_infosets

    assert len(enumerate_infosets(game)) == truth("leduc")["n_infosets"]


def test_max_result(game, truth):
    assert game.spec.max_result == truth("leduc")["max_result"]


def test_cfr_plus_reaches_reference_value(game, truth):
    data = truth("leduc")
    strategy = cfr_mod.cfr_plus(game, data["cfr_plus"]["iterations"])
    tol = data["tolerances"]
    assert (
        abs(br_mod.expected_value(game, strategy, strategy) - data["game_value_seat0"])
        <= tol["cfr_value"]
    )
    assert br_mod.exploitability(game, strategy) <= tol["cfr_exploitability"]
