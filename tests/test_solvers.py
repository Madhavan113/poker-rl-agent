import numpy as np
import pytest
from engine_helpers import pure_strategy, random_strategy

from exsolver.games import CALL, FOLD, RAISE, KuhnPoker, LeducPoker
from exsolver.solvers import (
    CFRPlus,
    StrategyMixer,
    best_response,
    best_response_value,
    cfr_plus,
    expected_value,
    exploitability,
    mix_strategies,
    seat_exploitability,
)
from exsolver.strategy import (
    enumerate_infosets,
    merge_seats,
    seat_entries,
    seat_of_key,
    strategy_to_array,
    uniform_strategy,
)

KUHN_VALUE = -1.0 / 18.0


@pytest.fixture(scope="module")
def kuhn() -> KuhnPoker:
    return KuhnPoker()


@pytest.fixture(scope="module")
def kuhn_eq(kuhn):
    return cfr_plus(kuhn, 1000)


def test_kuhn_cfr_plus_value_and_exploitability(kuhn, kuhn_eq):
    assert expected_value(kuhn, kuhn_eq, kuhn_eq) == pytest.approx(KUHN_VALUE, abs=1e-3)
    assert exploitability(kuhn, kuhn_eq) < 1e-3
    assert set(kuhn_eq) == set(enumerate_infosets(kuhn))
    strategy_to_array(kuhn, kuhn_eq)  # valid distributions with zeros on illegal actions


def test_kuhn_equilibrium_structure(kuhn_eq):
    alpha = kuhn_eq["0:J|"][RAISE]
    assert 0.0 <= alpha <= 1 / 3 + 0.01
    assert kuhn_eq["0:Q|"][RAISE] == pytest.approx(0.0, abs=0.01)
    assert kuhn_eq["0:K|"][RAISE] == pytest.approx(3 * alpha, abs=0.01)
    assert kuhn_eq["0:J|cb"][CALL] == pytest.approx(0.0, abs=0.01)
    assert kuhn_eq["0:Q|cb"][CALL] == pytest.approx(alpha + 1 / 3, abs=0.01)
    assert kuhn_eq["0:K|cb"][CALL] == pytest.approx(1.0, abs=0.01)
    assert kuhn_eq["1:J|c"][RAISE] == pytest.approx(1 / 3, abs=0.01)
    assert kuhn_eq["1:Q|c"][RAISE] == pytest.approx(0.0, abs=0.01)
    assert kuhn_eq["1:K|c"][RAISE] == pytest.approx(1.0, abs=0.01)
    assert kuhn_eq["1:J|b"][CALL] == pytest.approx(0.0, abs=0.01)
    assert kuhn_eq["1:Q|b"][CALL] == pytest.approx(1 / 3, abs=0.01)
    assert kuhn_eq["1:K|b"][CALL] == pytest.approx(1.0, abs=0.01)


def test_expected_value_hand_computed(kuhn):
    always_bet = pure_strategy(kuhn, lambda k, legal: RAISE if RAISE in legal else CALL, (0,))
    always_call = pure_strategy(kuhn, lambda k, legal: CALL, (1,))
    always_fold = pure_strategy(kuhn, lambda k, legal: FOLD if FOLD in legal else CALL, (1,))
    check_fold = pure_strategy(kuhn, lambda k, legal: FOLD if FOLD in legal else CALL, (0,))
    assert expected_value(kuhn, always_bet, always_call) == pytest.approx(0.0)
    assert expected_value(kuhn, always_bet, always_fold) == pytest.approx(1.0)
    assert expected_value(kuhn, check_fold, always_call) == pytest.approx(0.0)
    # seat 1 bets after every check; seat 0 folds: seat 1 wins the ante every hand
    bet_after_check = pure_strategy(kuhn, lambda k, legal: RAISE if RAISE in legal else CALL, (1,))
    assert expected_value(kuhn, check_fold, bet_after_check) == pytest.approx(-1.0)
    with pytest.raises(KeyError):
        expected_value(kuhn, seat_entries(always_bet, 0), {})


def test_best_response_hand_computed_vs_station_seat1(kuhn):
    # seat-1 opponent always calls facing a bet and always bets after a check
    opp = pure_strategy(kuhn, lambda k, legal: CALL if FOLD in legal else RAISE, (1,))
    br, value = best_response(kuhn, opp, 0)
    # K: +2 either way (tie -> CALL); Q: bet 0 or check-call 0 (tie -> CALL); J: check-fold -1
    assert value == pytest.approx(1 / 3)
    assert br["0:K|"][CALL] == 1.0 and br["0:K|cb"][CALL] == 1.0
    assert br["0:Q|"][CALL] == 1.0 and br["0:Q|cb"][CALL] == 1.0
    assert br["0:J|"][CALL] == 1.0 and br["0:J|cb"][FOLD] == 1.0
    assert set(br) == {k for k in enumerate_infosets(kuhn) if seat_of_key(k) == 0}
    assert expected_value(kuhn, br, opp) == pytest.approx(value)
    assert best_response_value(kuhn, opp, 0) == pytest.approx(value)


def test_best_response_hand_computed_vs_maniac_seat0(kuhn):
    # seat-0 opponent always bets; seat 1's check-infosets have zero reach, so the BR plays CALL
    # there (the passive rule: CALL when legal, else lowest legal index)
    opp = pure_strategy(kuhn, lambda k, legal: RAISE if RAISE in legal else FOLD, (0,))
    br, value = best_response(kuhn, opp, 1)
    assert value == pytest.approx(1 / 3)  # K calls +2, Q calls 0, J folds -1
    assert br["1:K|b"][CALL] == 1.0 and br["1:Q|b"][CALL] == 1.0 and br["1:J|b"][FOLD] == 1.0
    assert all(br[f"1:{c}|c"][CALL] == 1.0 for c in "JQK")
    assert -expected_value(kuhn, opp, br) == pytest.approx(value)


def test_best_response_calls_at_unreachable_infosets(kuhn):
    # seat 0 never bets, so seat 1's facing-bet infosets are unreachable: BR checks/calls there
    opp = pure_strategy(kuhn, lambda k, legal: CALL, (0,))
    br, value = best_response(kuhn, opp, 1)
    for c in "JQK":
        assert br[f"1:{c}|b"][CALL] == 1.0
    assert value == pytest.approx(best_response_value(kuhn, opp, 1))
    # a genuine tie at a reachable infoset still resolves to the lowest legal index
    station = pure_strategy(kuhn, lambda k, legal: CALL if FOLD in legal else RAISE, (1,))
    br0, _ = best_response(kuhn, station, 0)
    assert br0["0:K|"][CALL] == 1.0  # bet and check-call both win 2 with a K


def test_strategy_validation_rejects_bad_rows(kuhn):
    good = uniform_strategy(kuhn)
    for row, what in [
        ([0.0, 1.5, -0.5], "negative"),
        ([0.0, np.nan, np.nan], "non-finite"),
        ([0.0, np.inf, 0.0], "non-finite"),
        ([0.5, 0.5, 0.0], "illegal"),
        ([0.0, 0.7, 0.7], "sum to 1"),
    ]:
        bad = {**good, "0:J|": np.array(row)}
        with pytest.raises(ValueError, match="0:J") as info:
            strategy_to_array(kuhn, bad)
        assert what in str(info.value)
        with pytest.raises(ValueError):
            expected_value(kuhn, bad, good)
    strategy_to_array(kuhn, good)  # the unmodified strategy is fine


def test_cfr_regrets_and_average_stay_zero_on_illegal_actions(kuhn):
    solver = CFRPlus(kuhn).run(50)
    illegal = ~solver.tree.infoset_legal
    assert np.all(solver.regrets[illegal] == 0.0)
    assert np.all(solver.cum_strategy[illegal] == 0.0)
    assert np.all(solver.regrets >= 0.0)


def test_best_response_is_pure_and_consistent(kuhn):
    rng = np.random.default_rng(1)
    for seat in (0, 1):
        for _ in range(5):
            opp = random_strategy(kuhn, rng)
            br, value = best_response(kuhn, opp, seat)
            for k, v in br.items():
                assert seat_of_key(k) == seat and v.sum() == 1.0 and v.max() == 1.0
            ev = expected_value(kuhn, br, opp) if seat == 0 else -expected_value(kuhn, opp, br)
            assert ev == pytest.approx(value, abs=1e-12)
            # no other strategy of `seat` does better
            for _ in range(5):
                alt = random_strategy(kuhn, rng, (seat,))
                ev_alt = (
                    expected_value(kuhn, alt, opp) if seat == 0 else -expected_value(kuhn, opp, alt)
                )
                assert ev_alt <= value + 1e-12


def test_exploitability_definitions(kuhn, kuhn_eq):
    uni = uniform_strategy(kuhn)
    br0 = best_response_value(kuhn, uni, 0)
    br1 = best_response_value(kuhn, uni, 1)
    assert exploitability(kuhn, uni) == pytest.approx(0.5 * (br0 + br1))
    assert exploitability(kuhn, uni) > 0.1
    v = expected_value(kuhn, kuhn_eq, kuhn_eq)
    assert seat_exploitability(kuhn, kuhn_eq, 0, v) == pytest.approx(0.0, abs=1e-3)
    assert seat_exploitability(kuhn, kuhn_eq, 1, v) == pytest.approx(0.0, abs=1e-3)
    assert seat_exploitability(kuhn, uni, 0, v) == pytest.approx(br1 + v)
    assert seat_exploitability(kuhn, uni, 1, v) == pytest.approx(br0 - v)
    assert seat_exploitability(kuhn, uni, 0, v) > 0 and seat_exploitability(kuhn, uni, 1, v) > 0
    # an equilibrium strategy paired with anything is never exploitable at its own seat
    mixed = merge_seats(kuhn_eq, uni)
    assert seat_exploitability(kuhn, mixed, 0, v) == pytest.approx(0.0, abs=1e-3)
    assert seat_exploitability(kuhn, mixed, 1, v) > 0.1


@pytest.mark.parametrize("game_cls,n_profiles,n_trials", [(KuhnPoker, 7, 6), (LeducPoker, 4, 2)])
def test_mixture_is_realization_equivalent(game_cls, n_profiles, n_trials):
    game = game_cls()
    rng = np.random.default_rng(7)
    for _ in range(n_trials):
        profiles = [random_strategy(game, rng, concentration=0.5) for _ in range(n_profiles)]
        w = rng.dirichlet(np.ones(n_profiles))
        sigma = random_strategy(game, rng)
        mix1 = mix_strategies(game, w, profiles, 1)
        target = sum(
            wi * expected_value(game, sigma, th) for wi, th in zip(w, profiles, strict=True)
        )
        assert expected_value(game, sigma, mix1) == pytest.approx(target, abs=1e-9)
        mix0 = mix_strategies(game, w, profiles, 0)
        target = sum(
            wi * expected_value(game, th, sigma) for wi, th in zip(w, profiles, strict=True)
        )
        assert expected_value(game, mix0, sigma) == pytest.approx(target, abs=1e-9)
        strategy_to_array(game, mix0, (0,))
        strategy_to_array(game, mix1, (1,))
        assert set(mix0) == {k for k in enumerate_infosets(game) if seat_of_key(k) == 0}


def test_mixture_handles_unreachable_infosets_and_zero_weights(kuhn):
    rng = np.random.default_rng(3)
    # profile A never checks as seat 0, so its "0:X|cb" infosets are unreachable
    a = pure_strategy(kuhn, lambda k, legal: RAISE if RAISE in legal else FOLD, (0,))
    b = random_strategy(kuhn, rng, (0,))
    c = random_strategy(kuhn, rng, (0,))
    sigma = random_strategy(kuhn, rng, (1,))
    for w in (np.array([1.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.0]), np.array([0.2, 0.3, 0.5])):
        mix = mix_strategies(kuhn, w, [a, b, c], 0)
        target = sum(
            wi * expected_value(kuhn, th, sigma) for wi, th in zip(w, [a, b, c], strict=True)
        )
        assert expected_value(kuhn, mix, sigma) == pytest.approx(target, abs=1e-9)
        for v in mix.values():
            assert v.sum() == pytest.approx(1.0) and np.all(v >= 0)
    only_a = mix_strategies(kuhn, np.array([1.0, 0.0, 0.0]), [a, b, c], 0)
    for k in ("0:J|", "0:Q|", "0:K|"):
        np.testing.assert_allclose(only_a[k], a[k])
    for k in ("0:J|cb", "0:Q|cb", "0:K|cb"):
        np.testing.assert_allclose(only_a[k], a[k])  # unreachable: plain weighted average = a


def test_mixer_matches_function_and_single_profile_identity(kuhn):
    rng = np.random.default_rng(11)
    profiles = [random_strategy(kuhn, rng) for _ in range(5)]
    mixer = StrategyMixer(kuhn, profiles, 1)
    w = rng.dirichlet(np.ones(5))
    ref = mix_strategies(kuhn, w, profiles, 1)
    out = mixer.mix(w)
    assert set(out) == set(ref)
    for k in ref:
        np.testing.assert_allclose(out[k], ref[k])
    single = mix_strategies(kuhn, np.array([2.0]), [profiles[0]], 1)  # weights get normalised
    for k in single:
        np.testing.assert_allclose(single[k], profiles[0][k], atol=1e-15)
    with pytest.raises(ValueError):
        mixer.mix(np.zeros(5))
