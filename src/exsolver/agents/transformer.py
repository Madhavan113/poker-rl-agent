"""Agent driven by a trained ``ExploitTransformer`` through ``exsolver.model.inference.Policy``.

For the upcoming hand at ``seat`` the agent builds one counterfactual token prefix per infoset of
that seat -- ``ctx + HAND POS_s CARD_c [events...]``, i.e. ``tokenizer.encode_prefix`` -- and runs
the policy on all of them, plus the bare session stream (the shared head of those prefixes, whose
last token is a belief read-out position), in a **single** forward pass via ``Policy.all_at``.
The action rows give the strategy; the extra row gives the belief, which is cached for the same
context so that ``belief(ctx)`` right after ``strategy_for_hand(ctx, seat)`` costs nothing. The
agent never sees theta, the opponent id or labels: its only inputs are the completed hands in
``ctx``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from exsolver.agents.base import check_seat
from exsolver.data.records import HandRecord, SessionContext
from exsolver.data.tokenizer import Tokenizer
from exsolver.games.base import N_ACTIONS, Game
from exsolver.play import seat_infosets_with_prefixes
from exsolver.strategy import TabularStrategy, enumerate_infosets

if TYPE_CHECKING:  # keep exsolver.agents importable without torch
    from exsolver.model.inference import Policy

MODES = ("sample", "argmax")


class TransformerAgent:
    """``mode="sample"`` returns the action-head distribution (the runner samples from it);
    ``mode="argmax"`` puts all mass on its argmax (ties to the lowest index)."""

    def __init__(
        self,
        game: Game,
        policy: Policy,
        tokenizer: Tokenizer,
        mode: str = "sample",
        name: str | None = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.game = game
        self.policy = policy
        self.tokenizer = tokenizer
        self.mode = mode
        self.name = name if name is not None else f"transformer_{mode}"
        legal_of = enumerate_infosets(game)
        self._views = tuple(seat_infosets_with_prefixes(game, s) for s in (0, 1))
        self._legal = tuple(
            np.array(
                [[a in legal_of[key] for a in range(N_ACTIONS)] for key, _, _ in views], dtype=bool
            )
            for views in self._views
        )
        self.has_opponent_head = getattr(policy.model, "opp_head", None) is not None
        # (encoded session stream, belief) of the most recent context seen; the belief is a
        # function of the token stream only, so equal streams share it and no collision is possible
        self._belief_cache: tuple[tuple[int, ...], np.ndarray] | None = None

    def reset(self, rng: np.random.Generator) -> None:
        self._belief_cache = None

    def observe(self, hand: HandRecord) -> None:
        pass

    def _rows(self, ctx: SessionContext, seat: int) -> tuple[list[list[int]], list[int]]:
        """Counterfactual prefixes of ``seat`` (enumeration order) and their shared head.

        Each prefix equals ``tokenizer.encode_prefix(ctx.hands, seat, my_cards, events)``; the
        head is the unpadded session stream (``BOS`` + completed hands), encoded once.
        """
        tok = self.tokenizer
        head = tok.encode_hands(ctx.hands)
        prefixes = [
            head + tok.encode_hand_prefix(seat, cards, events)
            for _, cards, events in self._views[seat]
        ]
        return prefixes, head

    def prefixes_for_hand(self, ctx: SessionContext, seat: int) -> list[list[int]]:
        """Counterfactual prefixes, one per infoset of ``seat`` (enumeration order)."""
        return self._rows(ctx, check_seat(seat))[0]

    def strategy_for_hand(self, ctx: SessionContext, seat: int) -> TabularStrategy:
        seat = check_seat(seat)
        views, legal = self._views[seat], self._legal[seat]
        prefixes, head = self._rows(ctx, seat)
        rows = [*prefixes, head]  # the head's last token is BOS / END_HAND: a belief position
        legal_ext = np.concatenate([legal, np.ones((1, N_ACTIONS), dtype=bool)])  # dummy row
        out = self.policy.all_at(rows, legal_ext)
        probs = np.asarray(out["policy"], dtype=np.float64)
        if probs.shape != legal_ext.shape:
            raise RuntimeError(f"policy returned shape {probs.shape}, expected {legal_ext.shape}")
        probs = probs[: len(views)]
        belief = out["belief"]
        if belief is not None:
            self._belief_cache = (tuple(head), np.asarray(belief[len(views)], dtype=np.float64))
        if self.mode == "argmax":
            strat = np.zeros_like(probs)
            strat[np.arange(len(views)), np.argmax(probs, axis=1)] = 1.0
        else:
            strat = np.where(legal, probs, 0.0)
            strat /= strat.sum(axis=1, keepdims=True)  # float64 renormalisation of float32 rows
        return {key: strat[i] for i, (key, _, _) in enumerate(views)}

    def belief(self, ctx: SessionContext) -> np.ndarray | None:
        if not self.has_opponent_head:
            return None
        head = self.tokenizer.encode_hands(ctx.hands)  # unpadded: ends at BOS / last END_HAND
        key = tuple(head)
        cached = self._belief_cache
        if cached is not None and cached[0] == key:
            return cached[1].copy()
        b = np.asarray(self.policy.belief_at([head])[0], dtype=np.float64)
        self._belief_cache = (key, b)
        return b.copy()
