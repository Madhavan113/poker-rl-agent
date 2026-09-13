"""Game-parametrised tokenizer for session token streams.

Vocabulary order (spec, docs/experiments/e1-kuhn.md "Tokenizer")::

    PAD, BOS, HAND, END_HAND, POS_0, POS_1, CARD_0..CARD_{n-1}, BOARD_0..BOARD_{n-1},
    ME_FOLD, ME_CALL, ME_RAISE, OPP_FOLD, OPP_CALL, OPP_RAISE, SHOW_0..SHOW_{n-1}, NO_SHOW,
    RESULT_{-max_result}..RESULT_{+max_result}

A session is ``BOS`` followed, per hand, by
``HAND POS_s CARD_c.. [events..] (SHOW_c.. | NO_SHOW) RESULT_r END_HAND`` and right-padded with
``PAD``.  Multi-card games emit one ``CARD`` token per private card and one ``SHOW`` token per
revealed opponent card; Kuhn and Leduc use exactly one of each.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from exsolver.data.records import (
    ACT,
    ACTION_NAMES,
    BOARD,
    N_ACTIONS,
    Event,
    HandRecord,
)

# Number of tokens that do not scale with the deck or the stakes: PAD BOS HAND END_HAND POS_0
# POS_1 (6) + ME_* (3) + OPP_* (3) + NO_SHOW (1) + RESULT_0 (1).
_FIXED_TOKENS = 14


@dataclass(frozen=True)
class TokenizerSpec:
    """The two GameSpec fields the tokenizer depends on."""

    n_cards: int
    max_result: int

    @classmethod
    def from_game_spec(cls, spec: Any) -> TokenizerSpec:
        """Duck-typed constructor: anything with ``n_cards`` and ``max_result`` attributes."""
        if isinstance(spec, TokenizerSpec):
            return spec
        return cls(n_cards=int(spec.n_cards), max_result=int(spec.max_result))

    @property
    def vocab_size(self) -> int:
        return 3 * self.n_cards + 2 * self.max_result + _FIXED_TOKENS


def build_vocab(spec: TokenizerSpec) -> list[str]:
    """Token names in vocabulary order."""
    n, m = spec.n_cards, spec.max_result
    names = ["PAD", "BOS", "HAND", "END_HAND", "POS_0", "POS_1"]
    names += [f"CARD_{c}" for c in range(n)]
    names += [f"BOARD_{c}" for c in range(n)]
    names += [f"ME_{a}" for a in ACTION_NAMES]
    names += [f"OPP_{a}" for a in ACTION_NAMES]
    names += [f"SHOW_{c}" for c in range(n)]
    names += ["NO_SHOW"]
    names += [f"RESULT_{r}" for r in range(-m, m + 1)]
    return names


class Tokenizer:
    """Encode/decode ``HandRecord`` sessions to/from integer token streams.

    ``spec`` may be a ``TokenizerSpec`` or any object exposing ``n_cards`` / ``max_result``
    (e.g. the game engine's ``GameSpec``).
    """

    def __init__(self, spec: TokenizerSpec | Any) -> None:
        self.spec = TokenizerSpec.from_game_spec(spec)
        if self.spec.n_cards < 1 or self.spec.max_result < 0:
            raise ValueError(f"invalid tokenizer spec {self.spec}")
        n = self.spec.n_cards
        self.names: list[str] = build_vocab(self.spec)
        self._stoi: dict[str, int] = {name: i for i, name in enumerate(self.names)}
        self.vocab_size: int = len(self.names)

        # Fixed ids / block offsets (see module docstring for the layout).
        self.PAD, self.BOS, self.HAND, self.END_HAND = 0, 1, 2, 3
        self.POS_0, self.POS_1 = 4, 5
        self._card0 = 6
        self._board0 = self._card0 + n
        self._me0 = self._board0 + n
        self._opp0 = self._me0 + N_ACTIONS
        self._show0 = self._opp0 + N_ACTIONS
        self.NO_SHOW = self._show0 + n
        self._result0 = self.NO_SHOW + 1  # id of RESULT_{-max_result}
        assert (
            self.vocab_size == self.spec.vocab_size == self._result0 + 2 * self.spec.max_result + 1
        )

    # ------------------------------------------------------------------ names <-> ids
    def id(self, name: str) -> int:
        """Token id of ``name`` (KeyError if unknown)."""
        return self._stoi[name]

    def name(self, token_id: int) -> str:
        """Token name of ``token_id``."""
        return self.names[int(token_id)]

    # ------------------------------------------------------------------ typed token builders
    def pos_token(self, seat: int) -> int:
        if seat not in (0, 1):
            raise ValueError(f"seat must be 0 or 1, got {seat}")
        return self.POS_0 + seat

    def card_token(self, card: int) -> int:
        return self._card0 + self._check_card(card)

    def board_token(self, card: int) -> int:
        return self._board0 + self._check_card(card)

    def show_token(self, card: int) -> int:
        return self._show0 + self._check_card(card)

    def result_token(self, result: int) -> int:
        m = self.spec.max_result
        if not -m <= result <= m:
            raise ValueError(f"result {result} outside [-{m}, {m}]")
        return self._result0 + int(result) + m

    def me_action_token(self, action: int) -> int:
        return self._me0 + self._check_action(action)

    def opp_action_token(self, action: int) -> int:
        return self._opp0 + self._check_action(action)

    def action_token(self, actor: int, action: int) -> int:
        """Action token for ``actor`` (0 = agent, 1 = opponent)."""
        if actor == 0:
            return self.me_action_token(action)
        if actor == 1:
            return self.opp_action_token(action)
        raise ValueError(f"actor must be 0 or 1, got {actor}")

    # ------------------------------------------------------------------ token classification
    def is_me_action(self, token_id: int) -> bool:
        return self._me0 <= token_id < self._me0 + N_ACTIONS

    def is_opp_action(self, token_id: int) -> bool:
        return self._opp0 <= token_id < self._opp0 + N_ACTIONS

    def is_board(self, token_id: int) -> bool:
        return self._board0 <= token_id < self._board0 + self.spec.n_cards

    def is_card(self, token_id: int) -> bool:
        return self._card0 <= token_id < self._card0 + self.spec.n_cards

    def is_show(self, token_id: int) -> bool:
        return self._show0 <= token_id < self._show0 + self.spec.n_cards

    def is_result(self, token_id: int) -> bool:
        return self._result0 <= token_id < self.vocab_size

    def action_of_token(self, token_id: int) -> int:
        """Action (FOLD/CALL/RAISE) encoded by a ``ME_*`` or ``OPP_*`` token."""
        token_id = int(token_id)
        if self.is_me_action(token_id):
            return token_id - self._me0
        if self.is_opp_action(token_id):
            return token_id - self._opp0
        raise ValueError(f"token {token_id} ({self.name(token_id)}) is not an action token")

    def card_of_token(self, token_id: int) -> int:
        """Card index encoded by a ``CARD_*``, ``BOARD_*`` or ``SHOW_*`` token."""
        token_id = int(token_id)
        if self.is_card(token_id):
            return token_id - self._card0
        if self.is_board(token_id):
            return token_id - self._board0
        if self.is_show(token_id):
            return token_id - self._show0
        raise ValueError(f"token {token_id} ({self.name(token_id)}) is not a card token")

    def result_of_token(self, token_id: int) -> int:
        token_id = int(token_id)
        if not self.is_result(token_id):
            raise ValueError(f"token {token_id} ({self.name(token_id)}) is not a result token")
        return token_id - self._result0 - self.spec.max_result

    # ------------------------------------------------------------------ encoding
    def encode_event(self, event: Event) -> int:
        kind = event[0]
        if kind == ACT:
            _, actor, action = event
            return self.action_token(actor, action)
        if kind == BOARD:
            return self.board_token(event[1])
        raise ValueError(f"unknown event {event!r}")

    def encode_hand_prefix(
        self, seat: int, my_cards: Iterable[int] | int, events: Iterable[Event]
    ) -> list[int]:
        """``HAND POS_s CARD_c.. [events..]`` -- the hand-local part of a counterfactual query."""
        if isinstance(my_cards, int | np.integer):
            my_cards = (int(my_cards),)
        out = [self.HAND, self.pos_token(seat)]
        out.extend(self.card_token(c) for c in my_cards)
        if len(out) == 2:
            raise ValueError("a hand needs at least one private card")
        out.extend(self.encode_event(e) for e in events)
        return out

    def encode_hand(self, hand: HandRecord) -> list[int]:
        """Full hand: ``HAND POS_s CARD_c.. [events..] (SHOW_c.. | NO_SHOW) RESULT_r END_HAND``."""
        out = self.encode_hand_prefix(hand.seat, hand.my_cards, hand.events)
        if hand.opp_cards is None:
            out.append(self.NO_SHOW)
        else:
            shows = [self.show_token(c) for c in hand.opp_cards]
            if not shows:
                raise ValueError("opp_cards must be None or a non-empty tuple")
            out.extend(shows)
        out.append(self.result_token(hand.result))
        out.append(self.END_HAND)
        return out

    def encode_hands(self, hands: Iterable[HandRecord] | Any) -> list[int]:
        """``BOS`` followed by every completed hand (accepts a ``SessionContext`` too)."""
        hands = getattr(hands, "hands", hands)
        out = [self.BOS]
        for hand in hands:
            out.extend(self.encode_hand(hand))
        return out

    def encode_session(self, hands: Iterable[HandRecord] | Any, L: int | None = None) -> np.ndarray:
        """Encode a completed session as ``int16``; right-pad with ``PAD`` to ``L`` if given."""
        ids = self.encode_hands(hands)
        if L is None:
            return np.asarray(ids, dtype=np.int16)
        if len(ids) > L:
            raise ValueError(f"session has {len(ids)} tokens, longer than L={L}")
        out = np.full(L, self.PAD, dtype=np.int16)
        out[: len(ids)] = ids
        return out

    def encode_prefix(
        self,
        hands: Iterable[HandRecord] | Any,
        seat: int,
        my_cards: Iterable[int] | int,
        events: Iterable[Event] = (),
    ) -> list[int]:
        """Counterfactual query prefix: completed hands + ``HAND POS_s CARD_c.. [events..]``.

        This is exactly a prefix of ``encode_session`` for the hand that would complete it.
        """
        out = self.encode_hands(hands)
        out.extend(self.encode_hand_prefix(seat, my_cards, events))
        return out

    # ------------------------------------------------------------------ decoding
    def decode_hand(self, tokens: Sequence[int], start: int) -> tuple[HandRecord, int]:
        """Parse one hand starting at ``tokens[start]`` (which must be ``HAND``).

        Returns the record and the index just after its ``END_HAND``.
        """
        n_tok = len(tokens)
        i = start

        def peek() -> int:
            if i >= n_tok:
                raise ValueError(f"truncated hand starting at {start}")
            return int(tokens[i])

        if peek() != self.HAND:
            raise ValueError(f"expected HAND at {i}, got {self.name(peek())}")
        i += 1
        pos = peek()
        if pos not in (self.POS_0, self.POS_1):
            raise ValueError(f"expected POS_* at {i}, got {self.name(pos)}")
        seat = pos - self.POS_0
        i += 1
        my_cards: list[int] = []
        while self.is_card(peek()):
            my_cards.append(peek() - self._card0)
            i += 1
        if not my_cards:
            raise ValueError(f"expected CARD_* at {i}, got {self.name(peek())}")
        events: list[Event] = []
        while True:
            t = peek()
            if self.is_me_action(t):
                events.append((ACT, 0, t - self._me0))
            elif self.is_opp_action(t):
                events.append((ACT, 1, t - self._opp0))
            elif self.is_board(t):
                events.append((BOARD, t - self._board0))
            else:
                break
            i += 1
        t = peek()
        opp_cards: tuple[int, ...] | None
        if t == self.NO_SHOW:
            opp_cards = None
            i += 1
        elif self.is_show(t):
            shown: list[int] = []
            while self.is_show(peek()):
                shown.append(peek() - self._show0)
                i += 1
            opp_cards = tuple(shown)
        else:
            raise ValueError(f"expected SHOW_*/NO_SHOW at {i}, got {self.name(t)}")
        result = self.result_of_token(peek())
        i += 1
        if peek() != self.END_HAND:
            raise ValueError(f"expected END_HAND at {i}, got {self.name(peek())}")
        i += 1
        return HandRecord(seat, tuple(my_cards), events, opp_cards, result), i

    def decode_session(self, tokens: Sequence[int] | np.ndarray) -> list[HandRecord]:
        """Inverse of ``encode_session`` for a 1-D token stream; stops at the first ``PAD``."""
        toks = np.asarray(tokens)
        if toks.ndim != 1:
            raise ValueError(f"expected a 1-D token stream, got shape {toks.shape}")
        toks_l: list[int] = toks.tolist()
        if not toks_l or toks_l[0] != self.BOS:
            raise ValueError("session must start with BOS")
        hands: list[HandRecord] = []
        i = 1
        while i < len(toks_l) and toks_l[i] != self.PAD:
            hand, i = self.decode_hand(toks_l, i)
            hands.append(hand)
        return hands

    # ------------------------------------------------------------------ position masks
    def decision_mask(self, tokens: np.ndarray) -> np.ndarray:
        """Bool mask of positions whose *next* token is ``ME_*`` (any leading batch dims)."""
        t = np.asarray(tokens)
        is_me = (t >= self._me0) & (t < self._me0 + N_ACTIONS)
        mask = np.zeros(t.shape, dtype=bool)
        mask[..., :-1] = is_me[..., 1:]
        return mask

    def decision_positions(self, tokens: np.ndarray) -> np.ndarray:
        """Indices of decision positions in a 1-D token stream."""
        t = np.asarray(tokens)
        if t.ndim != 1:
            raise ValueError("decision_positions expects a 1-D stream; use decision_mask")
        return np.flatnonzero(self.decision_mask(t))

    def belief_mask(self, tokens: np.ndarray) -> np.ndarray:
        """Bool mask of belief read-out positions: ``BOS`` and every ``END_HAND``."""
        t = np.asarray(tokens)
        return (t == self.BOS) | (t == self.END_HAND)

    def belief_positions(self, tokens: np.ndarray) -> np.ndarray:
        """Indices of belief read-out positions in a 1-D token stream."""
        t = np.asarray(tokens)
        if t.ndim != 1:
            raise ValueError("belief_positions expects a 1-D stream; use belief_mask")
        return np.flatnonzero(self.belief_mask(t))

    # ------------------------------------------------------------------ helpers
    def _check_card(self, card: int) -> int:
        card = int(card)
        if not 0 <= card < self.spec.n_cards:
            raise ValueError(f"card {card} outside [0, {self.spec.n_cards})")
        return card

    @staticmethod
    def _check_action(action: int) -> int:
        action = int(action)
        if not 0 <= action < N_ACTIONS:
            raise ValueError(f"action {action} outside [0, {N_ACTIONS})")
        return action

    def __repr__(self) -> str:
        return f"Tokenizer(n_cards={self.spec.n_cards}, max_result={self.spec.max_result}, vocab_size={self.vocab_size})"
