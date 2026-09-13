"""Play one hand of a ``Game`` between an agent strategy and an opponent profile.

Everything the rest of the pipeline knows about a hand comes through here:

* ``play_hand`` deals with ``game.deal(rng)``, resolves later chance nodes (Leduc's board) with
  ``game.sample_chance``, samples the agent's actions from ``agent_strategy`` and the opponent's
  from ``opp_profile`` and returns a ``HandRecord`` built from the **agent's observations only**
  (its own cards, both players' actions, the board, the opponent's cards only at a showdown, the
  result) plus the list of the agent's ``Decision``s (its infoset, legal mask and action taken).
* ``seat_infosets_with_prefixes`` maps every infoset of a seat to the hand-local observation
  prefix the agent would have seen on the way there; the transformer agent turns these into
  counterfactual token prefixes and the evaluator uses them to query a strategy at every infoset.

Strategy rows are validated, never renormalised: mass on an illegal action, a row that does not
sum to one (tolerance 1e-6) or a missing infoset raises.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from exsolver.data.records import Event, HandRecord, act_event, board_event
from exsolver.games.base import FOLD, N_ACTIONS, Action, Game, State
from exsolver.strategy import TabularStrategy, seat_infosets

ROW_TOL = 1e-6


@dataclass(frozen=True)
class Decision:
    """One decision of the agent: where it was, what was legal, what it did."""

    infoset_key: str
    legal: np.ndarray  # bool[3] over (FOLD, CALL, RAISE)
    taken: int


def legal_mask(legal: Sequence[Action]) -> np.ndarray:
    """bool[3] mask from a list of legal actions."""
    mask = np.zeros(N_ACTIONS, dtype=bool)
    mask[list(legal)] = True
    return mask


def check_row(probs: np.ndarray, legal: Sequence[Action], key: str) -> np.ndarray:
    """Validate a strategy row for infoset ``key`` and return it as a float64 array.

    Raises ``ValueError`` on a wrong shape, negative entries, mass on an illegal action or a
    sum that differs from one by more than ``ROW_TOL``.
    """
    row = np.asarray(probs, dtype=np.float64)
    if row.shape != (N_ACTIONS,):
        raise ValueError(f"strategy row at {key!r} has shape {row.shape}, expected ({N_ACTIONS},)")
    if not np.all(np.isfinite(row)) or np.any(row < 0.0):
        raise ValueError(f"strategy row at {key!r} has negative or non-finite entries: {row}")
    illegal = np.ones(N_ACTIONS, dtype=bool)
    illegal[list(legal)] = False
    if np.any(row[illegal] != 0.0):
        raise ValueError(f"strategy row at {key!r} puts mass on an illegal action: {row}")
    if abs(float(row.sum()) - 1.0) > ROW_TOL:
        raise ValueError(f"strategy row at {key!r} sums to {row.sum()!r}, not 1")
    return row


def sample_action(row: np.ndarray, rng: np.random.Generator) -> int:
    """Draw an action from a validated row with one uniform variate."""
    cum = np.cumsum(row)
    a = int(np.searchsorted(cum, rng.random() * cum[-1], side="right"))
    # guard the (measure-zero) round-off case u * sum >= cum[-1]
    while a >= N_ACTIONS or row[a] == 0.0:
        a -= 1
    return a


def lookup_row(
    strategy: TabularStrategy, key: str, legal: Sequence[Action], who: str
) -> np.ndarray:
    """``check_row(strategy[key])`` with a clear error when the infoset is missing."""
    try:
        probs = strategy[key]
    except KeyError:
        raise KeyError(f"{who} strategy has no entry for infoset {key!r}") from None
    return check_row(probs, legal, key)


def _as_int_result(x: float) -> int:
    r = int(round(x))
    if abs(x - r) > 1e-9:
        raise ValueError(f"non-integral hand result {x}")
    return r


def play_hand(
    game: Game,
    agent_seat: int,
    agent_strategy: TabularStrategy,
    opp_profile: TabularStrategy,
    rng: np.random.Generator,
    *,
    agent_rng: np.random.Generator | None = None,
    opp_rng: np.random.Generator | None = None,
) -> tuple[HandRecord, list[Decision]]:
    """Play one hand; return the agent's ``HandRecord`` and its ``Decision`` list.

    ``rng`` drives the deal and any later chance node; the agent's and the opponent's action
    draws come from ``agent_rng`` / ``opp_rng`` when given (so that different agents facing the
    same seeds see the same cards) and from ``rng`` otherwise.
    """
    if agent_seat not in (0, 1):
        raise ValueError(f"agent_seat must be 0 or 1, got {agent_seat}")
    a_rng = rng if agent_rng is None else agent_rng
    o_rng = rng if opp_rng is None else opp_rng
    opp_seat = 1 - agent_seat

    s: State = game.deal(rng)
    events: list[Event] = []
    decisions: list[Decision] = []
    folded = False
    while not game.is_terminal(s):
        if game.is_chance(s):
            before = len(game.board_cards(s))
            s = game.sample_chance(s, rng)  # type: ignore[attr-defined]
            events.extend(board_event(c) for c in game.board_cards(s)[before:])
            continue
        p = game.current_player(s)
        legal = game.legal_actions(s)
        key = game.infoset_key(s, p)
        if p == agent_seat:
            row = lookup_row(agent_strategy, key, legal, "agent")
            a = sample_action(row, a_rng)
            decisions.append(Decision(key, legal_mask(legal), a))
            events.append(act_event(0, a))
        else:
            row = lookup_row(opp_profile, key, legal, "opponent")
            a = sample_action(row, o_rng)
            events.append(act_event(1, a))
        folded = folded or a == FOLD
        s = game.apply(s, a)

    returns = game.returns(s)
    record = HandRecord(
        seat=agent_seat,
        my_cards=tuple(int(c) for c in game.private_cards(s, agent_seat)),
        events=events,
        opp_cards=None if folded else tuple(int(c) for c in game.private_cards(s, opp_seat)),
        result=_as_int_result(float(returns[agent_seat])),
    )
    return record, decisions


# ---------------------------------------------------------------------- infoset prefixes
@lru_cache(maxsize=16)
def _prefix_table(
    game: Game, seat: int
) -> tuple[tuple[str, tuple[int, ...], tuple[Event, ...]], ...]:
    """Walk the tree once and record, per infoset of ``seat``, the agent-side observation prefix."""
    found: dict[str, tuple[tuple[int, ...], tuple[Event, ...]]] = {}
    stack: list[tuple[State, tuple[Event, ...]]] = [(game.root(), ())]
    while stack:
        s, events = stack.pop()
        if game.is_terminal(s):
            continue
        if game.is_chance(s):
            before = len(game.board_cards(s))
            for s2, _ in game.chance_outcomes(s):
                new = tuple(board_event(c) for c in game.board_cards(s2)[before:])
                stack.append((s2, events + new))
            continue
        p = game.current_player(s)
        if p == seat:
            key = game.infoset_key(s, p)
            cards = tuple(int(c) for c in game.private_cards(s, seat))
            prev = found.get(key)
            if prev is None:
                found[key] = (cards, events)
            elif prev != (cards, events):
                raise ValueError(f"infoset {key!r} is reached with different observations")
        actor = 0 if p == seat else 1
        for a in game.legal_actions(s):
            stack.append((game.apply(s, a), events + (act_event(actor, a),)))
    keys = seat_infosets(game, seat)
    if set(keys) != set(found):
        raise RuntimeError("tree walk and enumerate_infosets disagree on the seat's infosets")
    return tuple((k, found[k][0], found[k][1]) for k in keys)


def seat_infosets_with_prefixes(
    game: Game, seat: int
) -> list[tuple[str, tuple[int, ...], list[Event]]]:
    """Every infoset of ``seat`` as ``(infoset_key, my_cards, events)`` in enumeration order.

    ``events`` is the hand-local observation prefix exactly as ``play_hand`` would have recorded
    it on the way to that infoset (``("act", 0, a)`` for the agent, ``("act", 1, a)`` for the
    opponent, ``("board", card)`` for revealed board cards), so
    ``tokenizer.encode_prefix(ctx, seat, my_cards, events)`` is the counterfactual query for it.
    """
    if seat not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {seat}")
    return [(k, cards, list(events)) for k, cards, events in _prefix_table(game, seat)]
