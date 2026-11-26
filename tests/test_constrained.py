import numpy as np
import pytest

from exsolver.games import CALL, FOLD, RAISE, KuhnPoker, LeducPoker
from exsolver.games.tree import compile_tree
from exsolver.solvers import (
    LockedCFR,
    best_response,
    best_response_value,
    ev_dense,
    restricted_value,
    rnr_objective,
)
from exsolver.strategy import strategy_to_array
from tests.engine_helpers import random_strategy

KUHN_VALUE = -1.0 / 18.0


@pytest.fixture(scope="module")
def kuhn() -> KuhnPoker:
    return KuhnPoker()


def _random_locks(game, n, seed, concentration=1.0):
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            strategy_to_array(game, random_strategy(game, rng, concentration=concentration))
            for _ in range(n)
        ]
    )


def _seat_value(seat: int) -> float:
    return KUHN_VALUE if seat == 0 else -KUHN_VALUE


@pytest.mark.parametrize("seat", [0, 1])
def test_no_lock_is_nash(kuhn, seat):
    n_inf = compile_tree(kuhn).n_infosets
    locks = _random_locks(kuhn, 3, seed=1)
    ours = LockedCFR(kuhn, seat, 3).solve(locks, 2000, conf=np.zeros((3, n_inf)))
    for b in range(3):
        worst = restricted_value(kuhn, seat, ours[b], locks[b], np.zeros(n_inf))
        assert worst == pytest.approx(_seat_value(seat), abs=2e-3)
    # RNR with p = 0 is the same game
    ours_r = LockedCFR(kuhn, seat, 3).solve(locks, 2000, rnr_p=0.0)
    for b in range(3):
        worst = restricted_value(kuhn, seat, ours_r[b], locks[b], np.zeros(n_inf))
        assert worst == pytest.approx(_seat_value(seat), abs=2e-3)


@pytest.mark.parametrize("seat", [0, 1])
def test_full_lock_and_rnr_one_are_best_response(kuhn, seat):
    tree = compile_tree(kuhn)
    locks = _random_locks(kuhn, 4, seed=2)
    full = LockedCFR(kuhn, seat, 4).solve(locks, 1000, conf=np.ones((4, tree.n_infosets)))
    rnr1 = LockedCFR(kuhn, seat, 4).solve(locks, 1000, rnr_p=1.0)
    for b in range(4):
        opp = tree.strategy_from_dense(locks[b], (1 - seat,))
        target = best_response_value(kuhn, opp, seat)
        assert ev_dense(kuhn, seat, full[b], locks[b]) == pytest.approx(target, abs=1e-4)
        assert ev_dense(kuhn, seat, rnr1[b], locks[b]) == pytest.approx(target, abs=1e-4)


def test_restricted_value_endpoints(kuhn):
    tree = compile_tree(kuhn)
    rng = np.random.default_rng(3)
    for seat in (0, 1):
        for _ in range(5):
            ours = strategy_to_array(kuhn, random_strategy(kuhn, rng))
            lock = strategy_to_array(kuhn, random_strategy(kuhn, rng))
            ours_tab = tree.strategy_from_dense(ours, (seat,))
            free = restricted_value(kuhn, seat, ours, lock, np.zeros(tree.n_infosets))
            assert free == pytest.approx(-best_response_value(kuhn, ours_tab, 1 - seat), abs=1e-12)
            locked = restricted_value(kuhn, seat, ours, lock, np.ones(tree.n_infosets))
            assert locked == pytest.approx(ev_dense(kuhn, seat, ours, lock), abs=1e-12)
            mid = restricted_value(kuhn, seat, ours, lock, np.full(tree.n_infosets, 0.5))
            assert free - 1e-12 <= mid <= locked + 1e-12


def test_restricted_value_hand_computed(kuhn):
    """Seat 0 always bets; only seat 1's response facing a bet matters.

    Free adversary: K calls (+2 vs fold -1), J folds (-2 vs -1), Q calls (0 vs -1). Over the six
    equally likely deals (ours, theirs) our results are (J,Q) -2, (J,K) -2, (Q,J) +1, (Q,K) -2,
    (K,J) +1, (K,Q) +2, so the value is -1/3. Locking 1:J|b to "always call" turns (Q,J) and
    (K,J) into +2 each: value 0.
    """
    tree = compile_tree(kuhn)
    ours = np.zeros((tree.n_infosets, 3))
    for key, i in tree.infoset_index.items():
        legal = tree.infoset_legal[i]
        if key.startswith("0:"):
            ours[i, RAISE if legal[RAISE] else CALL] = 1.0
    lock = strategy_to_array(kuhn, random_strategy(kuhn, np.random.default_rng(0)))
    lock[tree.infoset_index["1:J|b"]] = [0.0, 1.0, 0.0]  # J always calls
    conf = np.zeros(tree.n_infosets)
    conf[tree.infoset_index["1:J|b"]] = 1.0
    free = restricted_value(kuhn, 0, ours, lock, np.zeros(tree.n_infosets))
    assert free == pytest.approx(-1.0 / 3.0, abs=1e-12)
    assert restricted_value(kuhn, 0, ours, lock, conf) == pytest.approx(0.0, abs=1e-12)
    assert FOLD == 0  # the hand computation above relies on the action encoding


@pytest.mark.parametrize("seat", [0, 1])
def test_solution_dominates_candidates(kuhn, seat):
    """The solved strategy is (approximately) optimal in its restricted game."""
    tree = compile_tree(kuhn)
    rng = np.random.default_rng(10 + seat)
    locks = _random_locks(kuhn, 3, seed=20 + seat, concentration=0.5)
    conf = rng.uniform(size=(3, tree.n_infosets))
    conf[:, :4] = 1.0
    solver = LockedCFR(kuhn, seat, 3)
    soft = solver.solve(locks, 3000, conf=conf)
    rnr = LockedCFR(kuhn, seat, 3).solve(locks, 3000, rnr_p=0.6)
    for b in range(3):
        opp = tree.strategy_from_dense(locks[b], (1 - seat,))
        br, _ = best_response(kuhn, opp, seat)
        cands = [strategy_to_array(kuhn, random_strategy(kuhn, rng)) for _ in range(30)]
        cands.append(
            strategy_to_array(kuhn, {**br, **tree.strategy_from_dense(locks[b], (1 - seat,))})
        )
        v_soft = restricted_value(kuhn, seat, soft[b], locks[b], conf[b])
        v_rnr = rnr_objective(kuhn, seat, rnr[b], locks[b], 0.6)
        for cand in cands:
            assert v_soft >= restricted_value(kuhn, seat, cand, locks[b], conf[b]) - 2e-3
            assert v_rnr >= rnr_objective(kuhn, seat, cand, locks[b], 0.6) - 2e-3


def test_rnr_frontier_is_monotone(kuhn):
    tree = compile_tree(kuhn)
    locks = _random_locks(kuhn, 2, seed=5, concentration=0.5)
    evs, worsts = [], []
    for p in (0.0, 0.3, 0.6, 0.9, 1.0):
        ours = LockedCFR(kuhn, 0, 2).solve(locks, 3000, rnr_p=p)
        evs.append([ev_dense(kuhn, 0, ours[b], locks[b]) for b in range(2)])
        worsts.append(
            [
                restricted_value(kuhn, 0, ours[b], locks[b], np.zeros(tree.n_infosets))
                for b in range(2)
            ]
        )
    evs, worsts = np.array(evs), np.array(worsts)
    assert np.all(np.diff(evs, axis=0) >= -2e-3)  # more trust in the model: more EV against it
    assert np.all(np.diff(worsts, axis=0) <= 2e-3)  # ... and a worse worst case


def test_batch_matches_single_and_warm_start(kuhn):
    tree = compile_tree(kuhn)
    locks = _random_locks(kuhn, 3, seed=7)
    conf = np.random.default_rng(8).uniform(size=(3, tree.n_infosets))
    batched = LockedCFR(kuhn, 1, 3).solve(locks, 300, conf=conf)
    for b in range(3):
        single = LockedCFR(kuhn, 1, 1).solve(locks[b : b + 1], 300, conf=conf[b : b + 1])
        np.testing.assert_allclose(batched[b], single[0], atol=1e-12)
    solver = LockedCFR(kuhn, 1, 3)
    solver.solve(locks, 500, conf=conf)
    warm = solver.solve(locks, 200, conf=conf)
    cold = LockedCFR(kuhn, 1, 3).solve(locks, 200, conf=conf)
    for b in range(3):
        v_warm = restricted_value(kuhn, 1, warm[b], locks[b], conf[b])
        v_cold = restricted_value(kuhn, 1, cold[b], locks[b], conf[b])
        assert v_warm >= v_cold - 1e-3


def test_leduc_lock_endpoints():
    game = LeducPoker()
    tree = compile_tree(game)
    locks = _random_locks(game, 2, seed=9)
    full = LockedCFR(game, 0, 2).solve(locks, 300, conf=np.ones((2, tree.n_infosets)))
    nash = LockedCFR(game, 1, 2).solve(locks, 1500, conf=np.zeros((2, tree.n_infosets)))
    for b in range(2):
        opp = tree.strategy_from_dense(locks[b], (1,))
        assert ev_dense(game, 0, full[b], locks[b]) == pytest.approx(
            best_response_value(game, opp, 0), abs=1e-3
        )
        worst = restricted_value(game, 1, nash[b], locks[b], np.zeros(tree.n_infosets))
        assert worst == pytest.approx(0.0856, abs=0.01)  # Leduc seat-1 value ~ +0.0856


def test_input_validation(kuhn):
    tree = compile_tree(kuhn)
    locks = _random_locks(kuhn, 1, seed=0)
    solver = LockedCFR(kuhn, 0, 1)
    with pytest.raises(ValueError):
        solver.solve(locks, 10)
    with pytest.raises(ValueError):
        solver.solve(locks, 10, conf=np.zeros((1, tree.n_infosets)), rnr_p=0.5)
    with pytest.raises(ValueError):
        solver.solve(locks, 10, conf=np.full((1, tree.n_infosets), 1.5))
    with pytest.raises(ValueError):
        solver.solve(locks, 10, rnr_p=-0.1)
    bad = locks.copy()
    bad[0, tree.infoset_rows[1][0]] = [0.5, 0.0, 0.0]
    with pytest.raises(ValueError):
        solver.solve(bad, 10, rnr_p=0.5)
    with pytest.raises(ValueError):
        LockedCFR(kuhn, 2, 1)
