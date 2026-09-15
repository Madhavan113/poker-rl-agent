"""DPT session generator: sessions against opponents from a population (or a continuous prior).

CLI::

    uv run python -m exsolver.data.generate --population runs/e1/population.npz --out data/e1a \\
        --n 100000 --hands 64 --seed 0 --workers 8
    uv run python -m exsolver.data.generate --continuous --out data/e1b --n 100000 --hands 64

Per session ``s`` (docs/experiments/e1-kuhn.md, "Data") the private generator
``session_rng(seed, s)`` draws, in this order: the opponent (``opp_id ~ weights`` from the
population, or a fresh theta from ``continuous_prior`` with ``opp_id = -1``); one collection
policy for the whole session from ``collection_mix`` (``equilibrium`` = CFR+ average strategy,
``random`` = uniform over legal actions, ``oracle`` = BR(theta)); then ``hands_per_session``
hands with alternating seats (agent at seat ``t % 2``) through ``exsolver.play.play_hand``. At
every agent decision the label is the best-response action against the true theta at that
infoset (ties to the lowest index, as the solver resolves them) together with the legal mask;
the *taken* action only enters the token stream. Because every session owns its generator the
output does not depend on the worker count: the sequential path and the multiprocessing path
write byte-identical shards (``exsolver.data.shards`` format) plus ``meta.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from exsolver.data.records import ACT, HandRecord
from exsolver.data.shards import (
    META_FILENAME,
    SHARD_GLOB,
    Shard,
    file_hash,
    make_meta,
    write_meta,
)
from exsolver.data.tokenizer import Tokenizer
from exsolver.games.base import Game
from exsolver.games.kuhn import KuhnPoker
from exsolver.games.tree import compile_tree
from exsolver.play import play_hand
from exsolver.population.kuhn_prior import KUHN_PARAM_NAMES, N_PARAMS, KuhnPrior
from exsolver.population.population import Population, sample_population
from exsolver.solvers.best_response import best_response
from exsolver.solvers.cfr import cfr_plus
from exsolver.strategy import TabularStrategy, merge_seats, uniform_strategy

COLLECTORS: tuple[str, ...] = ("equilibrium", "random", "oracle")
DEFAULT_COLLECTION_MIX: dict[str, float] = {"equilibrium": 0.4, "random": 0.3, "oracle": 0.3}
EQUILIBRIUM_ITERATIONS = 2000
DEFAULT_POPULATION_PATH = Path("runs/e1/population.npz")
_L_MULTIPLE = 64


# ---------------------------------------------------------------------- deterministic streams
def session_rng(seed: int, index: int) -> np.random.Generator:
    """Generator owned by session ``index`` of a run seeded with ``seed`` (independent streams)."""
    return np.random.default_rng(np.random.SeedSequence(int(seed), spawn_key=(int(index),)))


# ---------------------------------------------------------------------- sequence length
def max_hand_tokens(game: Game) -> int:
    """Largest number of tokens ``Tokenizer.encode_hand`` can emit for one hand of ``game``.

    ``HAND POS`` + private cards + the longest event sequence (actions and board reveals) +
    ``SHOW_*`` per opponent card (or ``NO_SHOW``) + ``RESULT END_HAND``.
    """
    root = game.root()
    if not game.is_chance(root):
        raise ValueError("expected the root of the game to be the deal chance node")
    dealt = game.chance_outcomes(root)[0][0]
    n_private = max(len(game.private_cards(dealt, 0)), len(game.private_cards(dealt, 1)))
    longest = 0
    stack: list[tuple[Any, int]] = [(dealt, 0)]
    while stack:
        s, n = stack.pop()
        if game.is_terminal(s):
            longest = max(longest, n)
        elif game.is_chance(s):
            before = len(game.board_cards(s))
            for s2, _ in game.chance_outcomes(s):
                stack.append((s2, n + len(game.board_cards(s2)) - before))
        else:
            for a in game.legal_actions(s):
                stack.append((game.apply(s, a), n + 1))
    return 2 + n_private + longest + max(n_private, 1) + 2


def default_L(game: Game, hands_per_session: int) -> int:
    """``1 + max_hand_tokens * H`` rounded up to a multiple of 64 (Kuhn, H = 64: 577 -> 640)."""
    n = 1 + max_hand_tokens(game) * int(hands_per_session)
    return -(-n // _L_MULTIPLE) * _L_MULTIPLE


# ---------------------------------------------------------------------- configuration
def normalise_mix(mix: Mapping[str, float] | None) -> dict[str, float]:
    """Validate a collection mix (known names, non-negative, sums to one) in ``COLLECTORS`` order."""
    mix = dict(DEFAULT_COLLECTION_MIX if mix is None else mix)
    unknown = set(mix) - set(COLLECTORS)
    if unknown:
        raise ValueError(f"unknown collection policies {sorted(unknown)}; choose from {COLLECTORS}")
    probs = np.array([float(mix.get(name, 0.0)) for name in COLLECTORS])
    if np.any(probs < 0.0) or not np.isfinite(probs).all():
        raise ValueError("collection mix weights must be finite and non-negative")
    if abs(float(probs.sum()) - 1.0) > 1e-9:
        raise ValueError(f"collection mix must sum to 1, got {probs.sum()!r}")
    return {name: float(p) for name, p in zip(COLLECTORS, probs, strict=True)}


@dataclass(frozen=True)
class GenConfig:
    """Everything a worker needs to reproduce any session of the run."""

    game: Game
    tokenizer: Tokenizer
    population: Population | None
    continuous_prior: KuhnPrior | None
    hands_per_session: int
    L: int
    seed: int
    collection_mix: dict[str, float]

    def __post_init__(self) -> None:
        if (self.population is None) == (self.continuous_prior is None):
            raise ValueError("pass exactly one of population / continuous_prior")
        if self.population is not None and len(self.population) < 1:
            raise ValueError("population is empty")
        if self.hands_per_session < 1:
            raise ValueError("hands_per_session must be >= 1")
        if self.L < 1 + max_hand_tokens(self.game) * self.hands_per_session:
            raise ValueError(
                f"L={self.L} cannot hold {self.hands_per_session} hands of {self.game.spec.name}"
            )

    @property
    def theta_dim(self) -> int:
        if self.population is not None:
            return int(self.population.thetas.shape[1])
        return N_PARAMS

    @property
    def collector_probs(self) -> np.ndarray:
        return np.array([self.collection_mix[name] for name in COLLECTORS])


# ---------------------------------------------------------------------- per-worker sampler
class SessionSampler:
    """Plays and labels sessions; one instance per worker (caches equilibrium and BR tables)."""

    def __init__(self, cfg: GenConfig) -> None:
        self.cfg = cfg
        self.tree = compile_tree(cfg.game)
        self.index = self.tree.infoset_index
        self.equilibrium = cfr_plus(cfg.game, EQUILIBRIUM_ITERATIONS)
        self.random = uniform_strategy(cfg.game)
        self._br_cache: dict[int, tuple[TabularStrategy, np.ndarray]] = {}

    def oracle_and_labels(
        self, opp_id: int, profile: TabularStrategy
    ) -> tuple[TabularStrategy, np.ndarray]:
        """BR(theta) for both seats and the ``[n_infosets]`` table of BR actions (cached per id)."""
        hit = self._br_cache.get(opp_id) if opp_id >= 0 else None
        if hit is not None:
            return hit
        br0, _ = best_response(self.cfg.game, profile, 0)
        br1, _ = best_response(self.cfg.game, profile, 1)
        oracle = merge_seats(br0, br1)
        table = np.full(self.tree.n_infosets, -1, dtype=np.int8)
        for key, row in oracle.items():
            table[self.index[key]] = int(np.argmax(row))  # one-hot rows: argmax is the BR action
        if opp_id >= 0:
            self._br_cache[opp_id] = (oracle, table)
        return oracle, table

    def draw_opponent(
        self, rng: np.random.Generator
    ) -> tuple[int, TabularStrategy, np.ndarray, int]:
        """``(opp_id, profile, theta, archetype)``; ``opp_id = -1`` under a continuous prior."""
        cfg = self.cfg
        if cfg.population is not None:
            pop = cfg.population
            opp_id = int(rng.choice(len(pop), p=pop.weights))
            return opp_id, pop.profiles[opp_id], pop.thetas[opp_id], int(pop.archetypes[opp_id])
        assert cfg.continuous_prior is not None
        profile, theta, archetype = cfg.continuous_prior.sample(rng)
        return -1, profile, theta, int(archetype)

    def play_session(self, index: int) -> dict[str, Any]:
        """Session ``index``: token stream, decision positions, labels, legal masks, opponent info."""
        cfg = self.cfg
        rng = session_rng(cfg.seed, index)
        opp_id, profile, theta, archetype = self.draw_opponent(rng)
        oracle, label_table = self.oracle_and_labels(opp_id, profile)
        collector = int(rng.choice(len(COLLECTORS), p=cfg.collector_probs))
        agent_strategy = (self.equilibrium, self.random, oracle)[collector]

        hands: list[HandRecord] = []
        labels: list[int] = []
        legal: list[np.ndarray] = []
        for t in range(cfg.hands_per_session):
            hand, decisions = play_hand(cfg.game, t % 2, agent_strategy, profile, rng)
            n_agent_actions = sum(1 for ev in hand.events if ev[0] == ACT and ev[1] == 0)
            if n_agent_actions != len(decisions):
                raise RuntimeError(
                    f"hand {t}: {len(decisions)} decisions but {n_agent_actions} agent actions"
                )
            for d in decisions:  # k-th decision of the hand <-> k-th ME_* position of the hand
                labels.append(int(label_table[self.index[d.infoset_key]]))
                legal.append(np.asarray(d.legal, dtype=bool))
            hands.append(hand)

        tokens = cfg.tokenizer.encode_session(hands, cfg.L)
        positions = cfg.tokenizer.decision_positions(tokens)
        if positions.size != len(labels):
            raise RuntimeError(f"{positions.size} decision positions but {len(labels)} labels")
        return {
            "tokens": tokens,
            "positions": positions,
            "labels": np.asarray(labels, dtype=np.int8),
            "legal": np.asarray(legal, dtype=bool).reshape(len(legal), 3),
            "belief_mask": cfg.tokenizer.belief_mask(tokens),
            "opp_id": opp_id,
            "theta": np.asarray(theta, dtype=np.float32),
            "archetype": archetype,
            "collector": collector,
        }

    def fill(self, shard: Shard, row: int, index: int) -> int:
        """Write session ``index`` into ``shard`` row ``row``; returns the collector id."""
        s = self.play_session(index)
        pos = s["positions"]
        shard.tokens[row] = s["tokens"]
        shard.action_mask[row, pos] = True
        shard.action_target[row, pos] = s["labels"]
        shard.legal[row, pos] = s["legal"]
        shard.belief_mask[row] = s["belief_mask"]
        shard.opp_id[row] = s["opp_id"]
        shard.theta[row] = s["theta"]
        shard.archetype[row] = s["archetype"]
        return int(s["collector"])


# ---------------------------------------------------------------------- chunked generation
_SAMPLER: SessionSampler | None = None


def _init_worker(cfg: GenConfig) -> None:
    global _SAMPLER
    _SAMPLER = SessionSampler(cfg)


def _generate_chunk(bounds: tuple[int, int]) -> tuple[Shard, np.ndarray]:
    """Sessions ``[start, stop)`` as one ``Shard`` plus the per-collector counts."""
    if _SAMPLER is None:
        raise RuntimeError("worker not initialised")
    start, stop = bounds
    cfg = _SAMPLER.cfg
    shard = Shard.empty(stop - start, cfg.L, cfg.theta_dim)
    counts = np.zeros(len(COLLECTORS), dtype=np.int64)
    for row, index in enumerate(range(start, stop)):
        counts[_SAMPLER.fill(shard, row, index)] += 1
    return shard, counts


def _chunks(n: int, size: int) -> list[tuple[int, int]]:
    return [(a, min(a + size, n)) for a in range(0, n, size)]


def _iter_results(
    cfg: GenConfig, chunks: Sequence[tuple[int, int]], n_workers: int
) -> Iterator[tuple[Shard, np.ndarray]]:
    """Chunk results in order, from the calling process or a spawn pool (same output either way)."""
    if n_workers <= 1:
        _init_worker(cfg)
        yield from map(_generate_chunk, chunks)
        return
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    if main_file is not None and main_file.startswith("<"):
        # a Pool whose workers die in their initializer respawns them forever, i.e. hangs
        raise RuntimeError(
            f"multiprocessing (spawn) cannot re-import a __main__ read from {main_file}; "
            "run from a file / module or pass n_workers=1"
        )
    ctx = mp.get_context("spawn")
    with ctx.Pool(int(n_workers), initializer=_init_worker, initargs=(cfg,)) as pool:
        yield from pool.imap(_generate_chunk, chunks)


class _ShardWriter:
    """Accumulate session rows in order and write ``shard_{i:05d}.npz`` files of ``shard_size``."""

    def __init__(self, out_dir: Path, shard_size: int, L: int, vocab_size: int) -> None:
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.L = L
        self.vocab_size = vocab_size
        self.paths: list[Path] = []
        self._buffer: list[Shard] = []
        self._buffered = 0

    def add(self, shard: Shard) -> None:
        self._buffer.append(shard)
        self._buffered += shard.n
        while self._buffered >= self.shard_size:
            big = Shard.concat(self._buffer)
            self._write(big.take(slice(0, self.shard_size)))
            rest = big.take(slice(self.shard_size, None))
            self._buffer = [rest] if rest.n else []
            self._buffered = rest.n

    def close(self) -> None:
        if self._buffer:
            self._write(Shard.concat(self._buffer))
            self._buffer, self._buffered = [], 0

    def _write(self, shard: Shard) -> None:
        shard.validate(L=self.L, vocab_size=self.vocab_size)
        path = self.out_dir / f"shard_{len(self.paths):05d}.npz"
        shard.save(path)
        self.paths.append(path)


def _prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob(SHARD_GLOB))
    meta = out_dir / META_FILENAME
    if existing or meta.exists():
        if not overwrite:
            raise FileExistsError(
                f"{out_dir} already holds a dataset; pass overwrite=True to replace it"
            )
        for p in existing:
            p.unlink()
        if meta.exists():
            meta.unlink()


def population_fingerprint(pop: Population) -> str:
    """sha256 of the population's thetas and weights (identifies it independently of any file)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(pop.thetas, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(pop.weights, dtype=np.float64).tobytes())
    return h.hexdigest()


def _population_meta(cfg: GenConfig, population_path: str | Path | None) -> dict[str, Any]:
    if cfg.population is not None:
        path = None if population_path is None else Path(population_path)
        return {
            "kind": "discrete",
            "size": len(cfg.population),
            "path": None if path is None else str(path),
            "sha256": None if path is None or not path.exists() else file_hash(path),
            "content_sha256": population_fingerprint(cfg.population),
            "archetype_names": list(cfg.population.archetype_names),
        }
    prior = cfg.continuous_prior
    assert prior is not None
    return {
        "kind": "continuous",
        "prior": {
            "class": type(prior).__name__,
            "archetypes": [
                {"name": a.name, "weight": a.weight, "kappa": a.kappa, "centre": a.centre}
                for a in prior.archetypes
            ],
            "centre_clip": list(prior.centre_clip),
            "alpha_max": prior.alpha_max,
        },
        "archetype_names": list(prior.names),
    }


def generate_dataset(
    game: Game,
    population: Population | None,
    tokenizer: Tokenizer,
    out_dir: str | Path,
    n_sessions: int,
    hands_per_session: int,
    seed: int,
    collection_mix: Mapping[str, float] | None = None,
    shard_size: int = 5000,
    n_workers: int = 8,
    continuous_prior: KuhnPrior | None = None,
    *,
    L: int | None = None,
    population_path: str | Path | None = None,
    chunk_size: int = 250,
    overwrite: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Generate ``n_sessions`` labelled sessions into ``out_dir`` (shards + ``meta.json``).

    Exactly one of ``population`` / ``continuous_prior`` must be given. ``L`` defaults to
    ``default_L(game, hands_per_session)``. Work is split into chunks of ``chunk_size`` sessions
    processed in index order, so the result is identical for every ``n_workers``. Returns the
    meta dictionary that was written.
    """
    if n_sessions < 1:
        raise ValueError("n_sessions must be >= 1")
    if shard_size < 1 or chunk_size < 1:
        raise ValueError("shard_size and chunk_size must be >= 1")
    if collection_mix is not None and not isinstance(collection_mix, Mapping):
        raise TypeError("collection_mix must be a mapping name -> probability")
    cfg = GenConfig(
        game=game,
        tokenizer=tokenizer,
        population=population,
        continuous_prior=continuous_prior,
        hands_per_session=int(hands_per_session),
        L=int(L) if L is not None else default_L(game, hands_per_session),
        seed=int(seed),
        collection_mix=normalise_mix(collection_mix),
    )
    out = Path(out_dir)
    _prepare_out_dir(out, overwrite)
    chunks = _chunks(int(n_sessions), min(int(chunk_size), int(shard_size)))
    writer = _ShardWriter(out, int(shard_size), cfg.L, tokenizer.vocab_size)
    counts = np.zeros(len(COLLECTORS), dtype=np.int64)
    n_decisions = 0
    t0 = time.perf_counter()
    bar = tqdm(
        total=n_sessions, disable=not progress, dynamic_ncols=True, file=sys.stderr, unit="sess"
    )
    for shard, chunk_counts in _iter_results(cfg, chunks, n_workers):
        counts += chunk_counts
        n_decisions += int(shard.action_mask.sum())
        writer.add(shard)
        bar.update(shard.n)
    writer.close()
    bar.close()
    elapsed = time.perf_counter() - t0

    meta = make_meta(
        game=game.spec.name,
        vocab_size=tokenizer.vocab_size,
        L=cfg.L,
        H=cfg.hands_per_session,
        population=_population_meta(cfg, population_path),
        collection_mix=cfg.collection_mix,
        seed=cfg.seed,
        n_cards=game.spec.n_cards,
        max_result=game.spec.max_result,
        n_opp=len(population) if population is not None else 0,
        theta_dim=cfg.theta_dim,
        theta_names=list(KUHN_PARAM_NAMES) if cfg.theta_dim == N_PARAMS else None,
        continuous=population is None,
        n_sessions=int(n_sessions),
        hands_per_session=cfg.hands_per_session,
        max_hand_tokens=max_hand_tokens(game),
        shard_size=int(shard_size),
        n_shards=len(writer.paths),
        shards=[p.name for p in writer.paths],
        n_decisions=n_decisions,
        collector_counts={name: int(c) for name, c in zip(COLLECTORS, counts, strict=True)},
        equilibrium_iterations=EQUILIBRIUM_ITERATIONS,
        n_workers=int(n_workers),
        chunk_size=int(chunk_size),
        elapsed_s=elapsed,
        sessions_per_s=n_sessions / max(elapsed, 1e-9),
        generator="exsolver.data.generate",
    )
    write_meta(out, meta)
    return meta


# ---------------------------------------------------------------------- CLI
def load_or_sample_population(path: Path, m: int, pop_seed: int) -> Population:
    """Load ``path`` or, if it does not exist, sample ``m`` opponents from ``KuhnPrior`` and save."""
    if path.exists():
        return Population.load(path)
    pop = sample_population(KuhnPrior(), int(m), np.random.default_rng(int(pop_seed)))
    path.parent.mkdir(parents=True, exist_ok=True)
    pop.save(path)
    print(
        f"sampled a population of {m} from KuhnPrior (seed {pop_seed}) and saved it to {path}",
        file=sys.stderr,
    )
    return pop


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate DPT training sessions for Kuhn poker.")
    p.add_argument(
        "--population",
        type=Path,
        default=DEFAULT_POPULATION_PATH,
        help="population .npz (sampled and saved if missing)",
    )
    p.add_argument(
        "--out", type=Path, required=True, help="output directory for shards + meta.json"
    )
    p.add_argument("--n", type=int, default=100_000, help="number of sessions")
    p.add_argument("--hands", type=int, default=64, help="hands per session")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--shard-size", type=int, default=5000)
    p.add_argument("--chunk-size", type=int, default=250, help="sessions per worker task")
    p.add_argument("--L", type=int, default=None, help="sequence length (default: default_L)")
    p.add_argument(
        "--continuous",
        action="store_true",
        help="fresh theta per session from KuhnPrior (E1b); --population is not used",
    )
    p.add_argument(
        "--m", type=int, default=256, help="population size when --population must be sampled"
    )
    p.add_argument(
        "--pop-seed", type=int, default=0, help="seed used when sampling a missing population"
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    game = KuhnPoker()
    tokenizer = Tokenizer(game.spec)
    if args.continuous:
        population, prior, pop_path = None, KuhnPrior(), None
    else:
        population, prior, pop_path = (
            load_or_sample_population(args.population, args.m, args.pop_seed),
            None,
            args.population,
        )
    meta = generate_dataset(
        game,
        population,
        tokenizer,
        args.out,
        n_sessions=args.n,
        hands_per_session=args.hands,
        seed=args.seed,
        shard_size=args.shard_size,
        n_workers=args.workers,
        continuous_prior=prior,
        L=args.L,
        population_path=pop_path,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
        progress=not args.no_progress,
    )
    summary = {
        k: meta[k]
        for k in (
            "n_sessions",
            "H",
            "L",
            "n_shards",
            "n_decisions",
            "collector_counts",
            "elapsed_s",
            "sessions_per_s",
        )
    }
    summary["out"] = str(args.out)
    print(json.dumps(summary, indent=2))
    return meta


__all__: Iterable[str] = (
    "COLLECTORS",
    "DEFAULT_COLLECTION_MIX",
    "GenConfig",
    "SessionSampler",
    "default_L",
    "generate_dataset",
    "max_hand_tokens",
    "normalise_mix",
    "population_fingerprint",
    "session_rng",
)

if __name__ == "__main__":
    main()
