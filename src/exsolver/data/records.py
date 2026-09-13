"""Data-level contract shared by the game engine, the session generator and the tokenizer.

These definitions mirror the ``exsolver/data/records.py`` block of
``docs/experiments/e1-kuhn.md`` and deliberately depend on nothing else in the package.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Action encoding shared with ``exsolver.games.base`` (duplicated here so the data layer never
# has to import the game engine).
FOLD, CALL, RAISE = 0, 1, 2
ACTIONS: tuple[int, int, int] = (FOLD, CALL, RAISE)
ACTION_NAMES: tuple[str, str, str] = ("FOLD", "CALL", "RAISE")
N_ACTIONS = 3

# ("act", actor, action) with actor 0 = agent, 1 = opponent  |  ("board", card)
Event = tuple[str, int, int] | tuple[str, int]
ACT = "act"
BOARD = "board"


def act_event(actor: int, action: int) -> Event:
    """Build an ``("act", actor, action)`` event."""
    return (ACT, int(actor), int(action))


def board_event(card: int) -> Event:
    """Build a ``("board", card)`` event."""
    return (BOARD, int(card))


@dataclass
class HandRecord:
    """One completed hand from the agent's point of view."""

    seat: int  # agent's seat this hand
    my_cards: tuple[int, ...]
    events: list[Event]
    opp_cards: tuple[int, ...] | None  # revealed at showdown, else None
    result: int  # chips won by the agent (negative when losing)


@dataclass
class SessionContext:
    """Completed hands of the current session, oldest first."""

    hands: list[HandRecord] = field(default_factory=list)
