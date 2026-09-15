"""Data generator: BR labels, exact masks, showdown reveals, parallel == sequential, meta, train smoke,
and the ``bayes_br`` label variant (posterior-mixture best response)."""

import hashlib
import inspect

import numpy as np
import pytest

import exsolver.data.generate as generate_module
from exsolver.agents import BayesBRAgent
from exsolver.bayes import ExactPosterior
from exsolver.data.bayes_labels import BayesBRLabeller, bayes_br_label_table
from exsolver.data.generate import (
    COLLECTORS,
    DEFAULT_COLLECTION_MIX,
    DEFAULT_LABELS,
    LABEL_VARIANTS,
    check_labels,
    default_L,
    generate_dataset,
    labels_from_meta,
    main,
    max_hand_tokens,
    normalise_mix,
    population_fingerprint,
    session_rng,
)
from exsolver.data.records import ACT, FOLD, HandRecord, SessionContext
from exsolver.data.shards import SHARD_KEYS, Shard, load_all, read_meta, shard_paths
from exsolver.data.tokenizer import Tokenizer, TokenizerSpec
from exsolver.games import KuhnPoker, LeducPoker
from exsolver.games.tree import compile_tree
from exsolver.population import KUHN_PARAM_NAMES, KuhnPrior, Population, sample_population
from exsolver.solvers import best_response
from exsolver.solvers.mixture import mix_strategies
from exsolver.strategy import merge_seats, uniform_strategy
from exsolver.train import TrainConfig, tokenizer_spec_from_meta, train

N, H, M = 20, 8, 8

# sha256 over every array of ``load_all`` of the ``data`` fixture's sequential run (M = 8 opponents from
# ``sample_population(KuhnPrior(), 8, default_rng(0))``, N = 20, H = 8, seed 0, default labels), recorded
# on the commit *before* the label-variant change so that the default output is pinned byte for byte.
ORACLE_BR_FIXTURE_SHA256 = "83dacd92cd2aa0da20f5ea260d41b6903cc3ab15284cc39f030f0742c919a226"


@pytest.fixture(scope="module")
def game() -> KuhnPoker:
    return KuhnPoker()


@pytest.fixture(scope="module")
def pop():
    return sample_population(KuhnPrior(), M, np.random.default_rng(0))


@pytest.fixture(scope="module")
def data(game, pop, tmp_path_factory):
    """The same tiny run written sequentially and with two spawn workers."""
    tok = Tokenizer(game.spec)
    root = tmp_path_factory.mktemp("gen")
    pop_path = root / "pop.npz"
    pop.save(pop_path)
    common = dict(n_sessions=N, hands_per_session=H, seed=0, shard_size=7, chunk_size=6, population_path=pop_path, progress=False)  # fmt: skip
    meta_seq = generate_dataset(game, pop, tok, root / "seq", n_workers=1, **common)
    meta_par = generate_dataset(game, pop, tok, root / "par", n_workers=2, **common)
    return tok, root, meta_seq, meta_par


@pytest.fixture(scope="module")
def bayes_data(game, pop, data):
    """The ``data`` run again with ``labels="bayes_br"``, sequentially and with two workers."""
    tok, root, _, _ = data
    common = dict(n_sessions=N, hands_per_session=H, seed=0, shard_size=7, chunk_size=6, population_path=root / "pop.npz", progress=False, labels="bayes_br")  # fmt: skip
    meta_seq = generate_dataset(game, pop, tok, root / "bayes_seq", n_workers=1, **common)
    meta_par = generate_dataset(game, pop, tok, root / "bayes_par", n_workers=2, **common)
    return root, meta_seq, meta_par


def br_profile(game, profile):
    return merge_seats(best_response(game, profile, 0)[0], best_response(game, profile, 1)[0])


def shard_sha256(shard):
    h = hashlib.sha256()
    for key in SHARD_KEYS:
        a = np.ascontiguousarray(getattr(shard, key))
        h.update(key.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def label_table_of(game, strategy):
    """``[n_infosets]`` table of argmax actions of a (one-hot) strategy, -1 where it has no entry."""
    tree = compile_tree(game)
    table = np.full(tree.n_infosets, -1, dtype=np.int64)
    for key, row in strategy.items():
        table[tree.infoset_index[key]] = int(np.argmax(row))
    return table


def agent_decisions(game, hand):
    """``(infoset_key, legal mask, taken)`` for every agent action of a decoded hand, via the engine.

    The agent's infosets do not depend on the opponent's card, so any card consistent with the
    deal (the revealed one at a showdown) can stand in for it. Also returns the terminal state.
    """
    agent, opp = hand.seat, 1 - hand.seat
    for s, _ in game.chance_outcomes(game.root()):
        if tuple(game.private_cards(s, agent)) != tuple(hand.my_cards):
            continue
        if hand.opp_cards is None or tuple(game.private_cards(s, opp)) == tuple(hand.opp_cards):
            break
    else:
        raise AssertionError("no deal is consistent with the decoded hand")
    out = []
    for ev in hand.events:
        assert ev[0] == ACT  # no board in Kuhn
        _, actor, a = ev
        p = game.current_player(s)
        assert actor == (0 if p == agent else 1)
        if actor == 0:
            legal = np.zeros(3, dtype=bool)
            legal[game.legal_actions(s)] = True
            out.append((game.infoset_key(s, p), legal, a))
        s = game.apply(s, a)
    return out, s


def test_labels_are_best_response_actions_and_masks_are_exact(game, pop, data):
    tok, root, meta, _ = data
    shard = load_all(root / "seq")
    shard.validate(L=meta["L"], vocab_size=tok.vocab_size)
    assert shard.n == N and shard.L == meta["L"] == default_L(game, H) == 128 and shard.P == 12
    np.testing.assert_array_equal(shard.action_mask, tok.decision_mask(shard.tokens))
    np.testing.assert_array_equal(shard.belief_mask, tok.belief_mask(shard.tokens))
    assert not shard.legal[~shard.action_mask].any()
    brs = {}
    n_showdowns = n_decisions = 0
    for s in range(shard.n):
        opp_id = int(shard.opp_id[s])
        assert 0 <= opp_id < M and shard.archetype[s] == pop.archetypes[opp_id]
        np.testing.assert_allclose(shard.theta[s], pop.thetas[opp_id], atol=1e-6)
        br = brs.setdefault(opp_id, br_profile(game, pop.profiles[opp_id]))
        hands = tok.decode_session(shard.tokens[s])
        assert len(hands) == H and [h.seat for h in hands] == [t % 2 for t in range(H)]
        positions = np.flatnonzero(shard.action_mask[s])
        k = 0
        for hand in hands:
            decisions, terminal = agent_decisions(game, hand)
            assert game.is_terminal(terminal)
            folded = any(ev[2] == FOLD for ev in hand.events)
            assert (hand.opp_cards is None) == folded  # opponent cards only at showdowns
            assert hand.result == game.returns(terminal)[hand.seat]
            n_showdowns += hand.opp_cards is not None
            for key, legal, taken in decisions:
                p = positions[k]
                k += 1
                assert shard.action_target[s, p] == int(np.argmax(br[key])), (s, key)
                np.testing.assert_array_equal(shard.legal[s, p], legal)
                assert tok.action_of_token(shard.tokens[s, p + 1]) == taken and legal[taken]
        assert k == positions.size
        n_decisions += k
    assert 0 < n_showdowns < N * H
    assert meta["n_decisions"] == n_decisions == int(shard.action_mask.sum())


def test_parallel_equals_sequential(data):
    _, root, meta_seq, meta_par = data
    seq, par = load_all(root / "seq"), load_all(root / "par")
    for key in SHARD_KEYS:
        np.testing.assert_array_equal(getattr(seq, key), getattr(par, key), err_msg=key)
    names = ["shard_00000.npz", "shard_00001.npz", "shard_00002.npz"]
    assert (
        [p.name for p in shard_paths(root / "seq")]
        == [p.name for p in shard_paths(root / "par")]
        == names
    )
    assert [Shard.load(p).n for p in shard_paths(root / "par")] == [7, 7, 6]
    for key in ("n_decisions", "collector_counts", "L", "H", "seed", "n_shards", "shards"):
        assert meta_seq[key] == meta_par[key], key
    assert meta_seq["n_workers"] == 1 and meta_par["n_workers"] == 2


def test_meta_has_what_train_reads(data, pop):
    tok, root, meta, _ = data
    on_disk = read_meta(root / "seq")
    for key in ("game", "vocab_size", "L", "H", "population", "collection_mix", "seed", "n_cards", "max_result", "n_opp"):  # fmt: skip
        assert key in on_disk and on_disk[key] == meta[key], key
    assert on_disk["game"] == "kuhn" and on_disk["vocab_size"] == tok.vocab_size == 27
    assert (on_disk["n_cards"], on_disk["max_result"]) == (3, 2)
    assert tokenizer_spec_from_meta(on_disk) == TokenizerSpec(3, 2)
    assert (
        on_disk["n_opp"] == M and on_disk["H"] == H and on_disk["seed"] == 0 and on_disk["L"] == 128
    )
    assert on_disk["collection_mix"] == DEFAULT_COLLECTION_MIX
    population = on_disk["population"]
    assert population["kind"] == "discrete" and population["size"] == M
    assert population["path"].endswith("pop.npz") and len(population["sha256"]) == 64
    assert population["content_sha256"] == population_fingerprint(pop)
    assert on_disk["theta_names"] == KUHN_PARAM_NAMES and on_disk["theta_dim"] == 12
    assert (
        on_disk["n_sessions"] == N and on_disk["n_shards"] == 3 and on_disk["continuous"] is False
    )
    assert set(on_disk["collector_counts"]) == set(COLLECTORS)
    assert sum(on_disk["collector_counts"].values()) == N


def test_train_smoke_on_generated_data(data, tmp_path):
    _, root, meta, _ = data
    cfg = TrainConfig(
        data_dir=str(root / "seq"),
        out_dir=str(tmp_path / "run"),
        steps=5,
        batch_size=4,
        lr=1e-3,
        warmup=1,
        d_model=32,
        n_layers=1,
        n_heads=2,
        max_len=meta["L"],
        eval_every=0,
        log_every=5,
        eval_size=2,
        seed=0,
        device="cpu",
        progress=False,
    )
    summary = train(cfg)
    assert summary["steps"] == 5
    assert summary["model_config"]["n_opp"] == M and summary["model_config"]["vocab_size"] == 27
    assert summary["model_config"]["theta_dim"] is None  # discrete population -> opponent head
    assert np.isfinite(summary["eval"]["loss"]) and summary["eval"]["n_action"] > 0


def test_continuous_prior_variant(game, tmp_path):
    tok = Tokenizer(game.spec)
    prior = KuhnPrior()
    meta = generate_dataset(game, None, tok, tmp_path / "cont", 6, 4, seed=11, n_workers=1, continuous_prior=prior, progress=False)  # fmt: skip
    shard = load_all(tmp_path / "cont")
    shard.validate(L=meta["L"], vocab_size=tok.vocab_size)
    assert (shard.opp_id == -1).all() and meta["n_opp"] == 0 and meta["continuous"] is True
    assert (
        meta["population"]["kind"] == "continuous"
        and meta["population"]["prior"]["class"] == "KuhnPrior"
    )
    for s in range(shard.n):
        # theta is the first draw of the session's own generator; the labels are BR(theta)
        profile, theta, archetype = prior.sample(session_rng(11, s))
        np.testing.assert_allclose(shard.theta[s], theta, atol=1e-6)
        assert shard.archetype[s] == archetype
        br = br_profile(game, profile)
        positions = np.flatnonzero(shard.action_mask[s])
        k = 0
        for hand in tok.decode_session(shard.tokens[s]):
            for key, legal, _ in agent_decisions(game, hand)[0]:
                assert shard.action_target[s, positions[k]] == int(np.argmax(br[key]))
                np.testing.assert_array_equal(shard.legal[s, positions[k]], legal)
                k += 1
        assert k == positions.size
    with pytest.raises(ValueError, match="exactly one"):
        generate_dataset(game, None, tok, tmp_path / "x", 2, 2, seed=0, n_workers=1, progress=False)
    with pytest.raises(ValueError, match="exactly one"):
        pop = sample_population(prior, 2, np.random.default_rng(0))
        generate_dataset(game, pop, tok, tmp_path / "y", 2, 2, seed=0, n_workers=1, continuous_prior=prior, progress=False)  # fmt: skip


def test_helpers_validation_and_cli(game, tmp_path, capsys):
    assert max_hand_tokens(game) == 9 and default_L(game, 64) == 640 and default_L(game, 8) == 128
    assert max_hand_tokens(LeducPoker()) == 2 + 1 + 9 + 1 + 2  # cbbc / board / cbbc
    assert normalise_mix(None) == DEFAULT_COLLECTION_MIX
    assert normalise_mix({"oracle": 1.0}) == {"equilibrium": 0.0, "random": 0.0, "oracle": 1.0}
    with pytest.raises(ValueError, match="sum to 1"):
        normalise_mix({"equilibrium": 0.5})
    with pytest.raises(ValueError, match="unknown"):
        normalise_mix({"foo": 1.0})
    # one independent, reproducible stream per session
    assert session_rng(0, 1).random() == session_rng(0, 1).random()
    assert session_rng(0, 1).random() != session_rng(0, 2).random() != session_rng(1, 2).random()

    tok = Tokenizer(game.spec)
    pop = sample_population(KuhnPrior(), 2, np.random.default_rng(0))
    generate_dataset(game, pop, tok, tmp_path / "d", 2, 2, seed=0, n_workers=1, progress=False)
    with pytest.raises(FileExistsError):  # never silently mix two datasets
        generate_dataset(game, pop, tok, tmp_path / "d", 2, 2, seed=0, n_workers=1, progress=False)
    generate_dataset(game, pop, tok, tmp_path / "d", 3, 2, seed=1, n_workers=1, progress=False, overwrite=True)  # fmt: skip
    assert load_all(tmp_path / "d").n == 3 and len(shard_paths(tmp_path / "d")) == 1
    with pytest.raises(ValueError, match="cannot hold"):
        generate_dataset(game, pop, tok, tmp_path / "e", 2, 2, seed=0, n_workers=1, L=10, progress=False)  # fmt: skip
    with pytest.raises(ValueError):
        generate_dataset(game, pop, tok, tmp_path / "f", 0, 2, seed=0, n_workers=1, progress=False)

    # CLI: samples and saves a missing population of --m opponents, then generates
    pop_path = tmp_path / "cli" / "pop.npz"
    meta = main(
        ["--population", str(pop_path), "--out", str(tmp_path / "cli" / "data"), "--n", "3", "--hands", "2",
         "--seed", "0", "--workers", "1", "--m", "4", "--pop-seed", "5", "--no-progress"]
    )  # fmt: skip
    saved = Population.load(pop_path)
    assert len(saved) == 4
    np.testing.assert_array_equal(
        saved.thetas, sample_population(KuhnPrior(), 4, np.random.default_rng(5)).thetas
    )
    assert meta["n_sessions"] == 3 and meta["population"]["path"] == str(pop_path)
    assert meta["population"]["content_sha256"] == population_fingerprint(saved)
    assert load_all(tmp_path / "cli" / "data").n == 3
    printed = capsys.readouterr().out
    assert '"n_sessions": 3' in printed


def test_collector_is_fixed_within_a_session(game, monkeypatch, tmp_path):
    """One collection policy per session: every hand of a session is played by the same collector."""
    tok = Tokenizer(game.spec)
    pop = sample_population(KuhnPrior(), 4, np.random.default_rng(2))
    n, h = 30, 6
    kinds_per_call: list[str] = []
    real = generate_module.play_hand

    def spy(game_, seat, agent_strategy, opp_profile, rng, **kw):
        sampler = generate_module._SAMPLER
        if agent_strategy is sampler.equilibrium:
            kind = "equilibrium"
        elif agent_strategy is sampler.random:
            kind = "random"
        else:
            kind = "oracle"
            want = br_profile(game_, opp_profile)
            assert agent_strategy.keys() == want.keys()
            assert all(np.array_equal(agent_strategy[k], want[k]) for k in want)
        kinds_per_call.append(kind)
        return real(game_, seat, agent_strategy, opp_profile, rng, **kw)

    monkeypatch.setattr(generate_module, "play_hand", spy)
    meta = generate_dataset(
        game, pop, tok, tmp_path / "c", n, h, seed=3, n_workers=1, progress=False
    )
    assert len(kinds_per_call) == n * h
    per_session = [set(kinds_per_call[s * h : (s + 1) * h]) for s in range(n)]
    assert all(len(kinds) == 1 for kinds in per_session)
    kinds = [next(iter(k)) for k in per_session]
    assert set(kinds) == set(COLLECTORS)
    assert {c: kinds.count(c) for c in COLLECTORS} == meta["collector_counts"]
    # shard-level cross-check: oracle sessions took the labelled action at every decision
    shard = load_all(tmp_path / "c")
    for s, kind in enumerate(kinds):
        pos = np.flatnonzero(shard.action_mask[s])
        taken = np.array([tok.action_of_token(shard.tokens[s, p + 1]) for p in pos])
        assert shard.legal[s, pos, taken].all()
        if kind == "oracle":
            np.testing.assert_array_equal(taken, shard.action_target[s, pos])
        else:
            assert kind in ("equilibrium", "random")


# ---------------------------------------------------------------------- labels="bayes_br"
def test_default_variant_output_is_unchanged(data):
    """(a) the default ``oracle_br`` output is byte-identical to the pre-variant generator."""
    _, root, meta, _ = data
    assert meta["labels"] == DEFAULT_LABELS == "oracle_br"
    assert shard_sha256(load_all(root / "seq")) == ORACLE_BR_FIXTURE_SHA256


def test_bayes_br_labels_are_best_responses_to_the_posterior_mixture(game, pop, data, bayes_data):
    """(b) hand 0: BR to the prior mixture; every hand t: BR to the mixture under the posterior
    given hands 0..t-1 only. Both recomputed here from ``mix_strategies`` + ``best_response``."""
    tok, root, _, _ = data
    _, meta, _ = bayes_data
    oracle, bayes = load_all(root / "seq"), load_all(root / "bayes_seq")
    # same seed => same sessions: only the labels may differ between the two variants
    for key in SHARD_KEYS:
        if key != "action_target":
            np.testing.assert_array_equal(getattr(oracle, key), getattr(bayes, key), err_msg=key)
    assert (oracle.action_target != bayes.action_target).any()
    assert meta["n_decisions"] == int(bayes.action_mask.sum())

    prior = np.full(M, 1.0 / M)
    np.testing.assert_array_equal(pop.weights, prior)
    prior_br = {seat: best_response(game, mix_strategies(game, prior, pop.profiles, 1 - seat), seat)[0] for seat in (0, 1)}  # fmt: skip
    n_hand0 = n_later = n_changed = 0
    for s in range(bayes.n):
        hands = tok.decode_session(bayes.tokens[s])
        positions = np.flatnonzero(bayes.action_mask[s])
        posterior = ExactPosterior(game, pop)  # the evaluator's object, prior = population weights
        k = 0
        for t, hand in enumerate(hands):
            seat = t % 2
            assert hand.seat == seat
            mix = mix_strategies(game, posterior.probs, pop.profiles, 1 - seat)
            br = best_response(game, mix, seat)[0]
            for key, _, _ in agent_decisions(game, hand)[0]:
                label = int(bayes.action_target[s, positions[k]])
                assert label == int(np.argmax(br[key])), (s, t, key)
                if t == 0:
                    assert label == int(np.argmax(prior_br[seat][key]))
                    n_hand0 += 1
                else:
                    n_later += 1
                    n_changed += label != int(np.argmax(prior_br[seat][key]))
                k += 1
            posterior.update(hand)  # the completed hand is observed only after its decisions
        assert k == positions.size
    assert n_hand0 > 0 and n_later > 0 and n_changed > 0  # the posterior actually moved the labels


def test_bayes_br_labels_match_the_bayes_br_agent(game, pop, data, bayes_data):
    """The label at hand t is the action ``BayesBRAgent`` takes with the same completed hands."""
    tok, root, _, _ = data
    bayes = load_all(root / "bayes_seq")
    agent = BayesBRAgent(game, pop)
    for s in range(0, bayes.n, 3):
        hands = tok.decode_session(bayes.tokens[s])
        positions = np.flatnonzero(bayes.action_mask[s])
        agent.reset(np.random.default_rng(0))
        k = 0
        for t, hand in enumerate(hands):
            strat = agent.strategy_for_hand(SessionContext(list(hands[:t])), t % 2)
            for key, _, _ in agent_decisions(game, hand)[0]:
                assert bayes.action_target[s, positions[k]] == int(np.argmax(strat[key]))
                k += 1
        assert k == positions.size


def _two_opponent_population():
    """Opponent A never bets a Jack, opponent B always does; everything else is identical."""
    base = np.array([0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
    never, always = base.copy(), base.copy()
    always[KUHN_PARAM_NAMES.index("0:J|")] = 1.0
    always[KUHN_PARAM_NAMES.index("1:J|c")] = 1.0
    return Population.from_thetas(np.stack([never, always]), np.array([0, 1]))


def test_bayes_br_labels_converge_to_the_true_opponents_oracle_br(game, tmp_path):
    """(c) once the showdowns make the posterior a point mass, the labels are that opponent's BR."""
    pop2 = _two_opponent_population()
    oracle = [label_table_of(game, br_profile(game, p)) for p in pop2.profiles]
    assert (oracle[0] != oracle[1]).any()  # the two best responses differ, so the test can fail
    prior_tables = [label_table_of(game, best_response(game, mix_strategies(game, pop2.weights, pop2.profiles, 1 - seat), seat)[0]) for seat in (0, 1)]  # fmt: skip
    tree = compile_tree(game)
    random_agent = uniform_strategy(game)
    for true in (0, 1):
        labeller = BayesBRLabeller(game, pop2)
        rng = np.random.default_rng(7 + true)
        np.testing.assert_array_equal(labeller.probs, [0.5, 0.5])
        converged_at = None
        hands: list[HandRecord] = []
        for t in range(48):
            seat = t % 2
            rows = tree.infoset_rows[seat]
            table = labeller.label_table(seat)
            assert (table[tree.infoset_rows[1 - seat]] == -1).all()
            if t == 0:
                np.testing.assert_array_equal(table[rows], prior_tables[seat][rows])
            if converged_at is not None:
                np.testing.assert_array_equal(table[rows], oracle[true][rows])
            hand, _ = generate_module.play_hand(game, seat, random_agent, pop2.profiles[true], rng)
            labeller.observe(hand)
            hands.append(hand)
            if converged_at is None and labeller.probs[true] == 1.0:
                converged_at = t + 1
        assert converged_at is not None and converged_at < 40  # a Jack showdown rules the other out
        np.testing.assert_array_equal(labeller.probs, np.eye(2)[true])
        for seat in (0, 1):
            rows = tree.infoset_rows[seat]
            np.testing.assert_array_equal(labeller.label_table(seat)[rows], oracle[true][rows])
            np.testing.assert_array_equal(
                bayes_br_label_table(game, pop2, hands, seat)[rows], oracle[true][rows]
            )
        # reset() forgets the session: back to the prior labels
        labeller.reset()
        assert labeller.n_hands == 0
        np.testing.assert_array_equal(labeller.label_table(0), prior_tables[0])

    # the same through the generator: after the posterior hits a point mass on the true opponent,
    # every later label equals that opponent's oracle label
    tok = Tokenizer(game.spec)
    generate_dataset(game, pop2, tok, tmp_path / "two", 12, 32, seed=5, n_workers=1, progress=False, labels="bayes_br")  # fmt: skip
    shard = load_all(tmp_path / "two")
    n_converged_sessions = n_checked = 0
    for s in range(shard.n):
        true = int(shard.opp_id[s])
        hands = tok.decode_session(shard.tokens[s])
        positions = np.flatnonzero(shard.action_mask[s])
        posterior = ExactPosterior(game, pop2)
        k = 0
        converged = False
        for hand in hands:
            point_mass = posterior.probs[true] == 1.0
            converged = converged or point_mass
            for key, _, _ in agent_decisions(game, hand)[0]:
                if point_mass:
                    assert (
                        shard.action_target[s, positions[k]]
                        == oracle[true][tree.infoset_index[key]]
                    )
                    n_checked += 1
                k += 1
            posterior.update(hand)
        assert k == positions.size
        n_converged_sessions += converged
    assert n_converged_sessions >= shard.n // 2 and n_checked > 0


def test_bayes_br_labeller_sees_only_population_and_observed_hands(
    game, pop, monkeypatch, tmp_path
):
    """(d) structural: the label function is built from the population and fed the completed hands
    exactly as the token stream records them; no theta / opp_id flows into it."""
    params = list(inspect.signature(BayesBRLabeller.__init__).parameters)
    assert params == ["self", "game", "population", "prior_weights"]
    assert list(inspect.signature(BayesBRLabeller.observe).parameters) == ["self", "hand"]
    assert list(inspect.signature(BayesBRLabeller.label_table).parameters) == ["self", "seat"]
    assert list(inspect.signature(BayesBRLabeller.label).parameters) == [
        "self",
        "seat",
        "infoset_key",
    ]
    assert list(inspect.signature(bayes_br_label_table).parameters) == ["game", "population", "hands", "seat", "prior_weights"]  # fmt: skip
    labeller = BayesBRLabeller(game, pop)
    assert not any(isinstance(v, np.ndarray) and v.shape == pop.thetas.shape for v in vars(labeller).values())  # fmt: skip

    calls: list[tuple[str, object]] = []

    class Spy(BayesBRLabeller):
        def reset(self):
            calls.append(("reset", None))
            super().reset()

        def observe(self, hand):
            calls.append(("observe", hand))
            super().observe(hand)

        def label_table(self, seat):
            calls.append(("label_table", seat))
            return super().label_table(seat)

    monkeypatch.setattr(generate_module, "BayesBRLabeller", Spy)
    n, h = 6, 5
    tok = Tokenizer(game.spec)
    generate_dataset(game, pop, tok, tmp_path / "spy", n, h, seed=2, n_workers=1, progress=False, labels="bayes_br")  # fmt: skip
    shard = load_all(tmp_path / "spy")
    assert len(calls) == n * (1 + 2 * h)
    for s in range(n):
        session = calls[s * (1 + 2 * h) : (s + 1) * (1 + 2 * h)]
        assert session[0] == ("reset", None)
        kinds = [c[0] for c in session[1:]]
        assert kinds == ["label_table", "observe"] * h  # labels before the hand, observe after
        assert [c[1] for c in session[1::2]] == [t % 2 for t in range(h)]
        observed = [c[1] for c in session[2::2]]
        assert observed == tok.decode_session(shard.tokens[s])  # exactly what the model sees


def test_bayes_br_parallel_equals_sequential(bayes_data):
    """(e)"""
    root, meta_seq, meta_par = bayes_data
    seq, par = load_all(root / "bayes_seq"), load_all(root / "bayes_par")
    for key in SHARD_KEYS:
        np.testing.assert_array_equal(getattr(seq, key), getattr(par, key), err_msg=key)
    assert [Shard.load(p).n for p in shard_paths(root / "bayes_par")] == [7, 7, 6]
    for key in (
        "n_decisions",
        "collector_counts",
        "labels",
        "L",
        "H",
        "seed",
        "n_shards",
        "shards",
    ):
        assert meta_seq[key] == meta_par[key], key


def test_meta_records_the_label_variant(data, bayes_data):
    """(f)"""
    _, root, meta_oracle, _ = data
    _, meta_bayes, _ = bayes_data
    assert meta_oracle["labels"] == "oracle_br" and meta_bayes["labels"] == "bayes_br"
    assert read_meta(root / "seq")["labels"] == "oracle_br"
    assert read_meta(root / "bayes_seq")["labels"] == "bayes_br"
    assert labels_from_meta(read_meta(root / "bayes_seq")) == "bayes_br"
    assert labels_from_meta({}) == "oracle_br"  # datasets written before the key existed
    assert labels_from_meta({"labels": "oracle_br"}) == "oracle_br"
    assert LABEL_VARIANTS == ("oracle_br", "bayes_br") and check_labels("bayes_br") == "bayes_br"
    with pytest.raises(ValueError, match="unknown label variant"):
        labels_from_meta({"labels": "thompson"})
    with pytest.raises(ValueError, match="unknown label variant"):
        check_labels("oracle")


def test_bayes_br_refuses_a_continuous_prior_and_is_a_cli_option(game, tmp_path, capsys):
    """(g) plus the CLI flag."""
    tok = Tokenizer(game.spec)
    with pytest.raises(ValueError, match="bayes_br.*discrete population"):
        generate_dataset(game, None, tok, tmp_path / "cont", 2, 2, seed=0, n_workers=1, continuous_prior=KuhnPrior(), progress=False, labels="bayes_br")  # fmt: skip
    assert not (tmp_path / "cont").exists()  # refused before anything was written
    with pytest.raises(ValueError, match="unknown label variant"):
        pop = sample_population(KuhnPrior(), 2, np.random.default_rng(0))
        generate_dataset(game, pop, tok, tmp_path / "bad", 2, 2, seed=0, n_workers=1, progress=False, labels="oracle")  # fmt: skip

    pop_path = tmp_path / "cli" / "pop.npz"
    common = ["--population", str(pop_path), "--n", "3", "--hands", "2", "--seed", "0", "--workers", "1", "--m", "3", "--no-progress"]  # fmt: skip
    meta = main(["--out", str(tmp_path / "cli" / "bayes"), "--labels", "bayes_br", *common])
    assert meta["labels"] == "bayes_br" and load_all(tmp_path / "cli" / "bayes").n == 3
    assert '"labels": "bayes_br"' in capsys.readouterr().out
    meta = main(["--out", str(tmp_path / "cli" / "default"), *common])
    assert meta["labels"] == "oracle_br"
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path / "cli" / "x"), "--labels", "thompson", *common])
    with pytest.raises(ValueError, match="bayes_br.*discrete population"):
        main(
            ["--out", str(tmp_path / "cli" / "y"), "--continuous", "--labels", "bayes_br", *common]
        )
