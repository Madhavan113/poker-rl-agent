"""Engine-free synthetic Kuhn-shaped sessions for tests and smoke training.

This is *not* Kuhn poker and must never be used for real experiments: the opponent's behaviour
ignores its card (one raise rate and one call rate per opponent id), and the "best-response"
labels are an arbitrary per-opponent lookup table over the agent's 12 infosets.  Predicting a
label therefore requires inferring the opponent id from the observed action frequencies in
context -- exactly the capability the training machinery must deliver.  The token stream,
the hand tree (histories, legal actions, showdown/fold payoffs) and the shard format are the
real ones, so shards produced here are drop-in replacements for generated data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from exsolver.data.records import ACT, CALL, FOLD, RAISE, Event, HandRecord
from exsolver.data.shards import Shard, make_meta, write_meta, write_shards
from exsolver.data.tokenizer import Tokenizer, TokenizerSpec

KUHN_N_CARDS = 3
KUHN_MAX_RESULT = 2
KUHN_MAX_HAND_TOKENS = 9  # HAND POS CARD + 3 actions + SHOW + RESULT + END_HAND
KUHN_THETA_DIM = 12

# Same fixed order as the population module's KUHN_PARAM_NAMES (spec); duplicated to avoid
# importing the game side.
SYNTHETIC_THETA_NAMES: tuple[str, ...] = (
    "0:J|", "0:Q|", "0:K|", "0:J|cb", "0:Q|cb", "0:K|cb",
    "1:J|c", "1:Q|c", "1:K|c", "1:J|b", "1:Q|b", "1:K|b",
)  # fmt: skip

_HIST_CHAR = {FOLD: "f", CALL: "c", RAISE: "b"}
# decision node history -> (acting seat, legal actions); legal[1] is the "continue" action.
_NODES: dict[str, tuple[int, tuple[int, int]]] = {
    "": (0, (CALL, RAISE)),
    "c": (1, (CALL, RAISE)),
    "b": (1, (FOLD, CALL)),
    "cb": (0, (FOLD, CALL)),
}
# terminal history -> (showdown?, pot share won, winning seat when folded)
_TERMINALS: dict[str, tuple[bool, int, int | None]] = {
    "cc": (True, 1, None),
    "bc": (True, 2, None),
    "cbc": (True, 2, None),
    "bf": (False, 1, 0),
    "cbf": (False, 1, 1),
}
_DEALS: tuple[tuple[int, int], ...] = tuple(
    (a, b) for a in range(KUHN_N_CARDS) for b in range(KUHN_N_CARDS) if a != b
)
_LEGAL_ROW = {
    legal: np.array([a in legal for a in range(3)], dtype=bool) for _, legal in _NODES.values()
}
# Agent infosets in KUHN_PARAM_NAMES order: (seat, card, history).
AGENT_INFOSETS: tuple[tuple[int, int, str], ...] = tuple(
    (seat, card, hist) for hist, (seat, _) in _NODES.items() for card in range(KUHN_N_CARDS)
)
_INFOSET_INDEX = {key: i for i, key in enumerate(AGENT_INFOSETS)}

Decision = tuple[int, tuple[int, int]]  # (label action, legal actions)


@dataclass(frozen=True)
class SyntheticConfig:
    """Knobs of the synthetic population and session shape."""

    n_opponents: int = 4
    hands_per_session: int = 64
    L: int | None = None  # default 1 + 9 * hands_per_session
    p_lo: float = 0.15  # least aggressive raise/call rate on the trait grid
    p_hi: float = 0.85  # most aggressive

    @property
    def seq_len(self) -> int:
        return self.L if self.L is not None else 1 + KUHN_MAX_HAND_TOKENS * self.hands_per_session


class SyntheticKuhnSessions:
    """Fixed synthetic population (``seed``) and a sampler of sessions against it.

    Opponent ``i`` has traits ``(p_raise, p_call)`` on a grid and a label table
    ``labels[i, infoset]`` drawn uniformly over the legal actions of that infoset.
    """

    def __init__(self, cfg: SyntheticConfig | None = None, seed: int = 0) -> None:
        self.cfg = cfg or SyntheticConfig()
        self.seed = int(seed)
        if self.cfg.n_opponents < 1:
            raise ValueError("need at least one opponent")
        if self.cfg.seq_len < 1 + KUHN_MAX_HAND_TOKENS * self.cfg.hands_per_session:
            raise ValueError("L too small for hands_per_session Kuhn hands")
        self.tokenizer = Tokenizer(TokenizerSpec(KUHN_N_CARDS, KUHN_MAX_RESULT))
        rng = np.random.default_rng(self.seed)
        m = self.cfg.n_opponents
        n_levels = max(1, math.ceil(math.sqrt(m)))
        levels = (
            np.array([(self.cfg.p_lo + self.cfg.p_hi) / 2])
            if n_levels == 1
            else np.linspace(self.cfg.p_lo, self.cfg.p_hi, n_levels)
        )
        ids = np.arange(m)
        self.p_raise: np.ndarray = levels[ids % n_levels].astype(np.float64)
        self.p_call: np.ndarray = levels[ids // n_levels].astype(np.float64)
        self.archetypes: np.ndarray = (ids // n_levels).astype(np.int32)
        # Balanced random split per infoset: half the opponents get legal[0], half legal[1], so
        # no label is predictable from the infoset alone (no-context accuracy ~ 1/2).
        self.labels: np.ndarray = np.empty((m, len(AGENT_INFOSETS)), dtype=np.int8)
        for j, (_, _, hist) in enumerate(AGENT_INFOSETS):
            legal = _NODES[hist][1]
            order = rng.permutation(m)
            self.labels[order[: m // 2], j] = legal[0]
            self.labels[order[m // 2 :], j] = legal[1]
        # theta in KUHN_PARAM_NAMES order: P(RAISE) x3, P(CALL) x3, P(RAISE) x3, P(CALL) x3.
        self.thetas: np.ndarray = np.stack(
            [
                np.repeat([pr, pc, pr, pc], 3)
                for pr, pc in zip(self.p_raise, self.p_call, strict=True)
            ]
        ).astype(np.float32)

    # ------------------------------------------------------------------ sampling
    @property
    def n_opponents(self) -> int:
        return self.cfg.n_opponents

    def play_hand(
        self, u: np.ndarray, deal: int, opp_id: int, seat: int
    ) -> tuple[HandRecord, list[Decision]]:
        """Play one hand from three uniforms ``u`` and a deal index; return record + agent decisions."""
        c_agent, c_opp = _DEALS[deal]
        hist = ""
        events: list[Event] = []
        decisions: list[Decision] = []
        k = 0
        while hist not in _TERMINALS:
            actor, legal = _NODES[hist]
            if actor == seat:
                label = int(self.labels[opp_id, _INFOSET_INDEX[(seat, c_agent, hist)]])
                decisions.append((label, legal))
                a = legal[int(u[k] * len(legal))]  # collection policy: uniform over legal
                events.append((ACT, 0, a))
            else:
                p = self.p_raise[opp_id] if legal[1] == RAISE else self.p_call[opp_id]
                a = legal[1] if u[k] < p else legal[0]
                events.append((ACT, 1, a))
            hist += _HIST_CHAR[a]
            k += 1
        showdown, share, winner = _TERMINALS[hist]
        if showdown:
            won = c_agent > c_opp
            opp_cards: tuple[int, ...] | None = (c_opp,)
        else:
            won = winner == seat
            opp_cards = None
        return HandRecord(seat, (c_agent,), events, opp_cards, share if won else -share), decisions

    def sample_session(
        self, rng: np.random.Generator, opp_id: int | None = None
    ) -> tuple[list[HandRecord], list[Decision], int]:
        """Sample ``hands_per_session`` hands (agent seat ``t % 2``) against ``opp_id``."""
        if opp_id is None:
            opp_id = int(rng.integers(self.n_opponents))
        h = self.cfg.hands_per_session
        u = rng.random((h, 3))
        deals = rng.integers(0, len(_DEALS), size=h)
        hands: list[HandRecord] = []
        decisions: list[Decision] = []
        for t in range(h):
            hand, dec = self.play_hand(u[t], int(deals[t]), opp_id, t % 2)
            hands.append(hand)
            decisions.extend(dec)
        return hands, decisions, opp_id

    def generate(self, n_sessions: int, rng: np.random.Generator) -> Shard:
        """Sample ``n_sessions`` sessions into one validated ``Shard``."""
        tok = self.tokenizer
        L = self.cfg.seq_len
        shard = Shard.empty(n_sessions, L, KUHN_THETA_DIM)
        opp_ids = rng.integers(0, self.n_opponents, size=n_sessions)
        for s in range(n_sessions):
            hands, decisions, opp_id = self.sample_session(rng, int(opp_ids[s]))
            tokens = tok.encode_session(hands, L)
            pos = tok.decision_positions(tokens)
            if len(pos) != len(decisions):
                raise RuntimeError("decision positions and labels disagree")
            shard.tokens[s] = tokens
            shard.belief_mask[s] = tok.belief_mask(tokens)
            shard.action_mask[s, pos] = True
            shard.action_target[s, pos] = [label for label, _ in decisions]
            shard.legal[s, pos] = np.stack([_LEGAL_ROW[legal] for _, legal in decisions])
            shard.opp_id[s] = opp_id
            shard.theta[s] = self.thetas[opp_id]
            shard.archetype[s] = self.archetypes[opp_id]
        shard.validate(L=L, tokenizer=tok)
        return shard

    # ------------------------------------------------------------------ io
    def meta(self, n_sessions: int, seed: int, **extra: Any) -> dict[str, Any]:
        return make_meta(
            game="kuhn-synthetic",
            vocab_size=self.tokenizer.vocab_size,
            L=self.cfg.seq_len,
            H=self.cfg.hands_per_session,
            population={"kind": "synthetic-grid", "seed": self.seed, **asdict(self.cfg)},
            collection_mix={"uniform": 1.0},
            seed=seed,
            n_cards=KUHN_N_CARDS,
            max_result=KUHN_MAX_RESULT,
            n_opp=self.n_opponents,
            theta_dim=KUHN_THETA_DIM,
            theta_names=list(SYNTHETIC_THETA_NAMES),
            n_sessions=int(n_sessions),
            **extra,
        )

    def write(
        self,
        out_dir: str | Path,
        n_sessions: int,
        seed: int = 1,
        shard_size: int | None = None,
        compress: bool = False,
    ) -> list[Path]:
        """Write ``n_sessions`` sessions as shards plus ``meta.json`` into ``out_dir``."""
        rng = np.random.default_rng(seed)
        chunk = shard_size or n_sessions
        paths: list[Path] = []
        done = 0
        while done < n_sessions:
            n = min(chunk, n_sessions - done)
            paths += write_shards(
                out_dir, self.generate(n, rng), compress=compress, start_index=len(paths)
            )
            done += n
        write_meta(out_dir, self.meta(n_sessions, seed))
        return paths


# ---------------------------------------------------------------------- no-context baseline
def decision_keys(tokenizer: Tokenizer, tokens: np.ndarray) -> list[tuple[Any, ...]]:
    """Hand-local information at each agent decision of one session, in stream order.

    A key is ``(seat, my_cards, events_so_far)``: everything a predictor without access to the
    previous hands could condition on.
    """
    keys: list[tuple[Any, ...]] = []
    for hand in tokenizer.decode_session(tokens):
        for i, ev in enumerate(hand.events):
            if ev[0] == ACT and ev[1] == 0:
                keys.append((hand.seat, hand.my_cards, tuple(hand.events[:i])))
    return keys


def no_context_baseline_accuracy(tokenizer: Tokenizer, train: Shard, eval_: Shard) -> float:
    """Accuracy on ``eval_`` of the best predictor that ignores previous hands.

    Fits the majority label per hand-local key on ``train`` (falling back to the global majority
    for unseen keys) -- the Bayes-optimal no-context predictor for this data.
    """
    counts: dict[tuple[Any, ...], np.ndarray] = {}
    total = np.zeros(3, dtype=np.int64)
    for s in range(train.n):
        labels = train.action_target[s][train.action_mask[s]]
        for key, label in zip(decision_keys(tokenizer, train.tokens[s]), labels, strict=True):
            counts.setdefault(key, np.zeros(3, dtype=np.int64))[label] += 1
            total[label] += 1
    fallback = int(total.argmax())
    correct = n = 0
    for s in range(eval_.n):
        labels = eval_.action_target[s][eval_.action_mask[s]]
        for key, label in zip(decision_keys(tokenizer, eval_.tokens[s]), labels, strict=True):
            c = counts.get(key)
            pred = fallback if c is None else int(c.argmax())
            correct += int(pred == label)
            n += 1
    return correct / max(n, 1)


def per_hand_accuracy(shard: Shard, pred: np.ndarray, tokenizer: Tokenizer) -> np.ndarray:
    """Mean accuracy of ``pred`` (``[N, L]`` predicted actions) grouped by hand index."""
    hand_idx = np.cumsum(shard.tokens == tokenizer.HAND, axis=1) - 1  # hand index per position
    ok = (pred == shard.action_target) & shard.action_mask
    n_hands = int(hand_idx.max()) + 1
    acc = np.zeros(n_hands)
    for h in range(n_hands):
        sel = shard.action_mask & (hand_idx == h)
        acc[h] = ok[sel].mean() if sel.any() else np.nan
    return acc


def default_generator(seed: int = 0, **cfg_kwargs: Any) -> SyntheticKuhnSessions:
    """Convenience constructor used by tests and the smoke CLI."""
    return SyntheticKuhnSessions(SyntheticConfig(**cfg_kwargs), seed=seed)


__all__: Sequence[str] = (
    "AGENT_INFOSETS",
    "KUHN_MAX_HAND_TOKENS",
    "KUHN_THETA_DIM",
    "SYNTHETIC_THETA_NAMES",
    "SyntheticConfig",
    "SyntheticKuhnSessions",
    "decision_keys",
    "default_generator",
    "no_context_baseline_accuracy",
    "per_hand_accuracy",
)
