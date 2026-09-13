"""Leduc hold'em with OpenSpiel's default parameters.

Six-card deck: ranks J, Q, K x suits s, h; card index ``rank * 2 + suit`` (Js=0, Jh=1, Qs=2,
Qh=3, Ks=4, Kh=5). Each player antes 1 and receives one private card. Two betting rounds, seat 0
acting first in both; bet size 2 in round 1 and 4 in round 2, at most two raises per round. One
public board card is revealed by a chance node between the rounds. At showdown a pair (private
card matching the board rank) beats a high card, otherwise the higher rank wins; equal ranks
split the pot. The largest possible swing is 1 + 2*2 + 2*4 = 13.

State layout: ``(card0, card1, board, round1, round2, pending_board)`` where cards are ``-1``
before the deal, ``board`` is ``-1`` until revealed, ``round1``/``round2`` are the per-round
action strings (``c`` check/call, ``b`` bet/raise, ``f`` fold) and ``pending_board`` is a card
pinned by ``state_from_deal(..., board=(card,))`` (``-1`` otherwise): when it is set the board
chance node has exactly one outcome, so hands can be replayed deterministically.

Infoset keys: ``"{seat}:{card}|{round1}"`` before the board and
``"{seat}:{card}|{round1}/{board}|{round2}"`` after it, e.g. ``"1:Qs|cbc/Kh|c"``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from exsolver.games.base import ACTION_CHARS, CALL, FOLD, RAISE, Action, GameSpec

LeducState = tuple[int, int, int, str, str, int]

RANK_NAMES: tuple[str, ...] = ("J", "Q", "K")
SUIT_NAMES: tuple[str, ...] = ("s", "h")
N_CARDS = 6
CARD_NAMES: tuple[str, ...] = tuple(RANK_NAMES[c // 2] + SUIT_NAMES[c % 2] for c in range(N_CARDS))
ANTE = 1
BET_SIZES: tuple[int, int] = (2, 4)
MAX_RAISES = 2

# within-round action strings that still need a decision -> (seat to act, legal actions)
_DECISION: dict[str, tuple[int, tuple[Action, ...]]] = {
    "": (0, (CALL, RAISE)),
    "c": (1, (CALL, RAISE)),
    "b": (1, (FOLD, CALL, RAISE)),
    "cb": (0, (FOLD, CALL, RAISE)),
    "bb": (0, (FOLD, CALL)),
    "cbb": (1, (FOLD, CALL)),
}
_CLOSED = frozenset({"cc", "bc", "cbc", "bbc", "cbbc"})  # round complete, hand continues
_FOLDED: dict[str, int] = {"bf": 1, "cbf": 0, "bbf": 0, "cbbf": 1}  # -> seat that folded
_DONE = _CLOSED | frozenset(_FOLDED)
# chips a player put into a *closed* round: every raise was matched
_ROUND_RAISES: dict[str, int] = {h: h.count("b") for h in _CLOSED}
# raises made by the folding player (seat 0 acts at even positions within a round)
_FOLDER_RAISES: dict[str, int] = {
    h: sum(1 for i, ch in enumerate(h) if ch == "b" and i % 2 == folder)
    for h, folder in _FOLDED.items()
}
ROUND_HISTORIES: tuple[str, ...] = tuple(_DECISION) + tuple(sorted(_CLOSED)) + tuple(_FOLDED)


class LeducPoker:
    """Leduc hold'em engine implementing the ``Game`` protocol."""

    spec = GameSpec(name="leduc", n_cards=N_CARDS, n_rounds=2, max_result=13)

    def __init__(self) -> None:
        n_deals = N_CARDS * (N_CARDS - 1)
        self._deals: tuple[tuple[LeducState, float], ...] = tuple(
            ((c0, c1, -1, "", "", -1), 1.0 / n_deals)
            for c0 in range(N_CARDS)
            for c1 in range(N_CARDS)
            if c0 != c1
        )

    # games hash by (type, spec) so compiled trees are shared between instances
    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and other.spec == self.spec

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.spec))

    # -- tree structure -------------------------------------------------------------------
    def root(self) -> LeducState:
        return (-1, -1, -1, "", "", -1)

    def is_chance(self, s: LeducState) -> bool:
        return s[0] < 0 or (s[2] < 0 and s[3] in _CLOSED)

    def chance_outcomes(self, s: LeducState) -> list[tuple[LeducState, float]]:
        if s[0] < 0:
            return list(self._deals)
        if s[2] >= 0 or s[3] not in _CLOSED:
            raise ValueError(f"not a chance node: {s}")
        c0, c1, _, r1, _, pending = s
        if pending >= 0:
            return [((c0, c1, pending, r1, "", -1), 1.0)]
        cards = [c for c in range(N_CARDS) if c != c0 and c != c1]
        p = 1.0 / len(cards)
        return [((c0, c1, b, r1, "", -1), p) for b in cards]

    def is_terminal(self, s: LeducState) -> bool:
        return s[3] in _FOLDED or (s[2] >= 0 and s[4] in _DONE)

    def returns(self, s: LeducState) -> tuple[float, float]:
        """Integer-valued chips (as floats) won by each seat.

        A fold hands the winner the folder's ante plus the folder's matched bets; a showdown moves
        the loser's whole stake. Both stakes are equal at showdown, so a split pot nets 0 to each
        player and no half-chips can occur.
        """
        c0, c1, board, r1, r2, _ = s
        folder = _FOLDED.get(r1)
        if folder is not None:
            won = float(ANTE + _FOLDER_RAISES[r1] * BET_SIZES[0])
            return (won, -won) if folder == 1 else (-won, won)
        if board < 0:
            raise ValueError(f"not a terminal node: {s}")
        pot1 = _ROUND_RAISES[r1] * BET_SIZES[0]
        folder = _FOLDED.get(r2)
        if folder is not None:
            won = float(ANTE + pot1 + _FOLDER_RAISES[r2] * BET_SIZES[1])
            return (won, -won) if folder == 1 else (-won, won)
        if r2 not in _CLOSED:
            raise ValueError(f"not a terminal node: {s}")
        won = float(ANTE + pot1 + _ROUND_RAISES[r2] * BET_SIZES[1])
        w = self.showdown_winner(c0, c1, board)
        return (w * won, -w * won)

    @staticmethod
    def showdown_winner(c0: int, c1: int, board: int) -> int:
        """+1 if seat 0 wins, -1 if seat 1 wins, 0 on a split pot."""
        r0, r1, rb = c0 // 2, c1 // 2, board // 2
        if r0 == rb:
            return 1
        if r1 == rb:
            return -1
        return (r0 > r1) - (r0 < r1)

    def current_player(self, s: LeducState) -> int:
        return _DECISION[s[3] if s[2] < 0 else s[4]][0]

    def legal_actions(self, s: LeducState) -> list[Action]:
        return list(_DECISION[s[3] if s[2] < 0 else s[4]][1])

    def apply(self, s: LeducState, a: Action) -> LeducState:
        c0, c1, board, r1, r2, pending = s
        if board < 0:
            if r1 not in _DECISION or a not in _DECISION[r1][1]:
                raise ValueError(f"illegal action {a} at {s}")
            return (c0, c1, board, r1 + ACTION_CHARS[a], r2, pending)
        if r2 not in _DECISION or a not in _DECISION[r2][1]:
            raise ValueError(f"illegal action {a} at {s}")
        return (c0, c1, board, r1, r2 + ACTION_CHARS[a], pending)

    # -- observations ---------------------------------------------------------------------
    def infoset_key(self, s: LeducState, player: int) -> str:
        card = CARD_NAMES[s[player]]
        if s[2] < 0:
            return f"{player}:{card}|{s[3]}"
        return f"{player}:{card}|{s[3]}/{CARD_NAMES[s[2]]}|{s[4]}"

    def private_cards(self, s: LeducState, player: int) -> tuple[int, ...]:
        return (s[player],)

    def board_cards(self, s: LeducState) -> tuple[int, ...]:
        return (s[2],) if s[2] >= 0 else ()

    def history(self, s: LeducState) -> str:
        """Public action history, rounds separated by ``/`` with the board card after it."""
        if s[2] < 0:
            return s[3]
        return f"{s[3]}/{CARD_NAMES[s[2]]}|{s[4]}"

    def card_name(self, c: int) -> str:
        return CARD_NAMES[c]

    # -- dealing --------------------------------------------------------------------------
    def deal(
        self, rng: np.random.Generator, seat_cards: Mapping[int, Sequence[int]] | None = None
    ) -> LeducState:
        fixed = {seat: int(cards[0]) for seat, cards in (seat_cards or {}).items()}
        c0, c1 = fixed.get(0, -1), fixed.get(1, -1)
        if c0 < 0 and c1 < 0:
            return self._deals[int(rng.integers(len(self._deals)))][0]
        if c0 >= 0 and c1 >= 0:
            return self.state_from_deal((c0,), (c1,))
        known = c0 if c0 >= 0 else c1
        rest = [c for c in range(N_CARDS) if c != known]
        other = rest[int(rng.integers(len(rest)))]
        return (c0, other, -1, "", "", -1) if c0 >= 0 else (other, c1, -1, "", "", -1)

    def state_from_deal(
        self, cards0: Sequence[int], cards1: Sequence[int], board: Sequence[int] = ()
    ) -> LeducState:
        c0, c1 = int(cards0[0]), int(cards1[0])
        if c0 == c1 or not (0 <= c0 < N_CARDS and 0 <= c1 < N_CARDS):
            raise ValueError(f"invalid deal {c0}, {c1}")
        pending = -1
        if board:
            pending = int(board[0])
            if pending in (c0, c1) or not 0 <= pending < N_CARDS:
                raise ValueError(f"invalid board card {pending} for deal {c0}, {c1}")
        return (c0, c1, -1, "", "", pending)

    def sample_chance(self, s: LeducState, rng: np.random.Generator) -> LeducState:
        """Resolve the chance node ``s`` (deal or board reveal) with ``rng``."""
        outcomes = self.chance_outcomes(s)  # all outcomes of a node are equiprobable
        return outcomes[int(rng.integers(len(outcomes)))][0]
