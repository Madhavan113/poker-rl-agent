"""Kuhn poker.

Cards J=0, Q=1, K=2; each player antes 1; one card each, no board. Seat 0 acts first:
check (CALL) or bet 1 (RAISE). After a check seat 1 checks (showdown, winner +1) or bets; after
check-bet seat 0 calls (showdown, winner +2) or folds (seat 1 wins +1). After a bet seat 1 calls
(showdown, winner +2) or folds (seat 0 wins +1).

State layout: ``(card0, card1, history)`` with cards ``-1`` before the deal and ``history`` the
string of action characters (``c`` check/call, ``b`` bet, ``f`` fold). Everything is a table
lookup so the hot paths (``apply``, ``infoset_key``, ``state_from_deal``) allocate almost nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from exsolver.games.base import ACTION_CHARS, CALL, FOLD, RAISE, Action, GameSpec

KuhnState = tuple[int, int, str]

CARD_NAMES: tuple[str, ...] = ("J", "Q", "K")
N_CARDS = 3

# decision histories -> (seat to act, legal actions)
_DECISION: dict[str, tuple[int, tuple[Action, ...]]] = {
    "": (0, (CALL, RAISE)),
    "c": (1, (CALL, RAISE)),
    "b": (1, (FOLD, CALL)),
    "cb": (0, (FOLD, CALL)),
}
# showdown histories -> returns indexed by ``card0 > card1``
_SHOWDOWN: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "cc": ((-1.0, 1.0), (1.0, -1.0)),
    "bc": ((-2.0, 2.0), (2.0, -2.0)),
    "cbc": ((-2.0, 2.0), (2.0, -2.0)),
}
_FOLD: dict[str, tuple[float, float]] = {"bf": (1.0, -1.0), "cbf": (-1.0, 1.0)}
_TERMINAL = frozenset(_SHOWDOWN) | frozenset(_FOLD)
HISTORIES: tuple[str, ...] = tuple(_DECISION) + tuple(_SHOWDOWN) + tuple(_FOLD)


class KuhnPoker:
    """Kuhn poker engine implementing the ``Game`` protocol."""

    spec = GameSpec(name="kuhn", n_cards=N_CARDS, n_rounds=1, max_result=2)

    def __init__(self) -> None:
        self._deals: tuple[tuple[KuhnState, float], ...] = tuple(
            ((c0, c1, ""), 1.0 / 6.0) for c0 in range(N_CARDS) for c1 in range(N_CARDS) if c0 != c1
        )
        self._keys: dict[tuple[int, int, str], str] = {
            (p, c, h): f"{p}:{CARD_NAMES[c]}|{h}"
            for p in (0, 1)
            for c in range(N_CARDS)
            for h in HISTORIES
        }

    # games hash by (type, spec) so compiled trees are shared between instances
    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and other.spec == self.spec

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.spec))

    # -- tree structure -------------------------------------------------------------------
    def root(self) -> KuhnState:
        return (-1, -1, "")

    def is_chance(self, s: KuhnState) -> bool:
        return s[0] < 0

    def chance_outcomes(self, s: KuhnState) -> list[tuple[KuhnState, float]]:
        if s[0] >= 0:
            raise ValueError(f"not a chance node: {s}")
        return list(self._deals)

    def is_terminal(self, s: KuhnState) -> bool:
        return s[2] in _TERMINAL

    def returns(self, s: KuhnState) -> tuple[float, float]:
        """Chips won by each seat as integer-valued floats.

        1 after a check-down or a fold, 2 after a called bet.
        """
        h = s[2]
        r = _FOLD.get(h)
        if r is not None:
            return r
        return _SHOWDOWN[h][s[0] > s[1]]

    def current_player(self, s: KuhnState) -> int:
        return _DECISION[s[2]][0]

    def legal_actions(self, s: KuhnState) -> list[Action]:
        return list(_DECISION[s[2]][1])

    def apply(self, s: KuhnState, a: Action) -> KuhnState:
        h = s[2]
        if a not in _DECISION[h][1]:
            raise ValueError(f"illegal action {a} at history {h!r}")
        return (s[0], s[1], h + ACTION_CHARS[a])

    # -- observations ---------------------------------------------------------------------
    def infoset_key(self, s: KuhnState, player: int) -> str:
        return self._keys[(player, s[player], s[2])]

    def private_cards(self, s: KuhnState, player: int) -> tuple[int, ...]:
        return (s[player],)

    def board_cards(self, s: KuhnState) -> tuple[int, ...]:
        return ()

    def history(self, s: KuhnState) -> str:
        return s[2]

    def card_name(self, c: int) -> str:
        return CARD_NAMES[c]

    # -- dealing --------------------------------------------------------------------------
    def deal(
        self, rng: np.random.Generator, seat_cards: Mapping[int, Sequence[int]] | None = None
    ) -> KuhnState:
        if not seat_cards:
            return self._deals[int(rng.integers(6))][0]
        fixed = {seat: int(cards[0]) for seat, cards in seat_cards.items()}
        c0, c1 = fixed.get(0, -1), fixed.get(1, -1)
        if c0 < 0 and c1 < 0:
            return self._deals[int(rng.integers(6))][0]
        if c0 >= 0 and c1 >= 0:
            return self.state_from_deal((c0,), (c1,))
        known = c0 if c0 >= 0 else c1
        rest = [c for c in range(N_CARDS) if c != known]
        other = rest[int(rng.integers(len(rest)))]
        return (c0, other, "") if c0 >= 0 else (other, c1, "")

    def state_from_deal(
        self, cards0: Sequence[int], cards1: Sequence[int], board: Sequence[int] = ()
    ) -> KuhnState:
        if board:
            raise ValueError("Kuhn poker has no board")
        c0, c1 = int(cards0[0]), int(cards1[0])
        if c0 == c1 or not (0 <= c0 < N_CARDS and 0 <= c1 < N_CARDS):
            raise ValueError(f"invalid deal {c0}, {c1}")
        return (c0, c1, "")

    def sample_chance(self, s: KuhnState, rng: np.random.Generator) -> KuhnState:
        """Resolve the chance node ``s`` with ``rng`` (the deal is the only chance node)."""
        if s[0] >= 0:
            raise ValueError(f"not a chance node: {s}")
        return self._deals[int(rng.integers(6))][0]
