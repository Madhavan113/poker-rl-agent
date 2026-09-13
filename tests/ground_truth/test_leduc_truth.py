"""Leduc ground truth from OpenSpiel 2.0.2's reference implementation (see truth/leduc.json)."""

import pytest

from tests.conftest import require_module

pytestmark = pytest.mark.ground_truth

leduc_mod = require_module("exsolver.games.leduc")
br_mod = require_module("exsolver.solvers.best_response")
cfr_mod = require_module("exsolver.solvers.cfr")
strategy_mod = require_module("exsolver.strategy")


@pytest.fixture(scope="module")
def game():
    return leduc_mod.LeducPoker()


def test_infoset_count(game, truth):
    assert len(strategy_mod.enumerate_infosets(game)) == truth("leduc")["n_infosets"]


def test_terminal_history_count(game, truth):
    """Walk the tree through the Game API only (not games/tree.py) and count terminal histories."""

    def count_terminals(state) -> int:
        if game.is_terminal(state):
            return 1
        if game.is_chance(state):
            return sum(count_terminals(child) for child, _prob in game.chance_outcomes(state))
        return sum(count_terminals(game.apply(state, a)) for a in game.legal_actions(state))

    assert count_terminals(game.root()) == truth("leduc")["n_terminals"]


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
