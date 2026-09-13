"""Shared game interface: action constants, ``GameSpec`` and the ``Game`` protocol.

Conventions shared by every engine in this package:

* actions are ints ``FOLD=0, CALL=1, RAISE=2``; CALL doubles as *check* and RAISE as *bet*;
* strategy vectors always have length 3 with zeros on illegal actions;
* ``State`` is an immutable plain tuple (each game documents its own layout) so states are
  cheap to create, hash and compare;
* infoset keys are ``"{seat}:{cards}|{history}"`` where the history uses ``c`` for CALL,
  ``b`` for RAISE and ``f`` for FOLD; rounds are separated by ``/`` and, in Leduc, the board
  card is appended after the separator (``"0:K|"``, ``"0:Q|cb"``, ``"1:Qs|cbc/Kh|c"``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

Action = int
FOLD: Action = 0
CALL: Action = 1
RAISE: Action = 2
N_ACTIONS = 3
ACTION_NAMES: tuple[str, ...] = ("FOLD", "CALL", "RAISE")
ACTION_CHARS: tuple[str, ...] = ("f", "c", "b")  # history characters, indexed by action

State = tuple[Any, ...]


@dataclass(frozen=True)
class GameSpec:
    """Static description of a game; also parametrises the tokenizer."""

    name: str  # "kuhn" | "leduc"
    n_cards: int  # deck size; card indices 0..n_cards-1
    n_rounds: int
    max_result: int  # largest |chips| one player can win in a hand (Kuhn 2, Leduc 13)
    ante: int = 1


class Game(Protocol):
    """Two-player zero-sum extensive-form game with chance nodes.

    All methods are pure functions of the (immutable) state. ``current_player``,
    ``legal_actions``, ``apply`` and ``infoset_key`` are only defined at decision nodes,
    ``returns`` only at terminals and ``chance_outcomes`` only at chance nodes.
    """

    spec: GameSpec

    def root(self) -> State:
        """Initial chance node, nothing dealt."""
        ...

    def is_chance(self, s: State) -> bool: ...

    def chance_outcomes(self, s: State) -> list[tuple[State, float]]:
        """(successor, probability) pairs of a chance node."""
        ...

    def sample_chance(self, s: State, rng: np.random.Generator) -> State:
        """Draw one successor of the chance node ``s`` with ``rng``.

        Equivalent to sampling from ``chance_outcomes(s)``; raises ``ValueError`` when ``s`` is
        not a chance node.
        """
        ...

    def is_terminal(self, s: State) -> bool: ...

    def returns(self, s: State) -> tuple[float, float]:
        """Chips won by seat 0 and seat 1 at a terminal; zero-sum and integer-valued.

        Kept as floats for the solvers. No half-chips arise: at showdown both players have
        matched every bet, so a split pot hands each player their own stake back (net 0).
        """
        ...

    def current_player(self, s: State) -> int:
        """Seat to act (0 or 1) at a decision node."""
        ...

    def legal_actions(self, s: State) -> list[Action]:
        """Legal actions at a decision node, ascending."""
        ...

    def apply(self, s: State, a: Action) -> State: ...

    def infoset_key(self, s: State, player: int) -> str:
        """``"{seat}:{cards}|{history}"`` as seen by ``player`` (see module docstring)."""
        ...

    def private_cards(self, s: State, player: int) -> tuple[int, ...]: ...

    def board_cards(self, s: State) -> tuple[int, ...]:
        """Public cards revealed so far."""
        ...

    def card_name(self, c: int) -> str: ...

    def deal(
        self, rng: np.random.Generator, seat_cards: Mapping[int, Sequence[int]] | None = None
    ) -> State:
        """Resolve the initial deal with ``rng``; ``seat_cards`` pins the cards of given seats."""
        ...

    def state_from_deal(
        self, cards0: Sequence[int], cards1: Sequence[int], board: Sequence[int] = ()
    ) -> State:
        """Deterministic deal for posterior replay (a pinned board makes its chance node trivial)."""
        ...
