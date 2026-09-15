"""Baseline and learned agents sharing the ``Agent`` protocol (see ``base.py``).

Only ``OracleBRAgent`` is handed the opponent's profile; every other agent sees nothing but the
``HandRecord``s of the hands it played.
"""

from exsolver.agents.base import Agent, check_seat
from exsolver.agents.bayes import (
    BayesBRAgent,
    BestResponseTable,
    PluralityBRAgent,
    ThompsonAgent,
    br_action_posterior,
)
from exsolver.agents.fixed import EquilibriumAgent, FixedStrategyAgent, OracleBRAgent, RandomAgent
from exsolver.agents.transformer import TransformerAgent

__all__ = [
    "Agent",
    "BayesBRAgent",
    "BestResponseTable",
    "EquilibriumAgent",
    "FixedStrategyAgent",
    "OracleBRAgent",
    "PluralityBRAgent",
    "RandomAgent",
    "ThompsonAgent",
    "TransformerAgent",
    "br_action_posterior",
    "check_seat",
]
