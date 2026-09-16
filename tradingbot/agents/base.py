"""The agent contract.

An agent is a *stateless opinion generator*: it precomputes indicators over a
full history in :meth:`Agent.fit`, then answers "what do I think about each
symbol as of this session?" via :meth:`Agent.signals_on`.

Two invariants matter:

* **Bounded output.** ``score`` in [-1, 1], ``confidence`` in [0, 1]. An agent
  that violates this cannot be compared against its peers.
* **Causal indicators.** ``fit`` may only use rolling/backward-looking math.
  ``tests/test_no_lookahead.py`` proves it by re-fitting on truncated data and
  asserting the answer at date ``t`` is bit-identical.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

import pandas as pd

from tradingbot.data.schema import History

Role = Literal["directional", "gate"]


@dataclass(frozen=True)
class Signal:
    """One agent's opinion about one symbol on one session.

    ``role="directional"`` votes on *what* to do. ``role="gate"`` votes only on
    *how much conviction* the system should have; it multiplies into consensus
    confidence and never moves the direction. That separation is what lets a
    risk agent veto an idea without pretending to have a directional view.
    """

    symbol: str
    score: float
    confidence: float
    agent: str
    role: Role = "directional"
    reason: str = ""
    parts: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(f"{self.agent}: score {self.score} outside [-1, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"{self.agent}: confidence {self.confidence} outside [0, 1]")
        if self.role not in ("directional", "gate"):
            raise ValueError(f"{self.agent}: unknown role {self.role!r}")


def clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def safe_z(value: float, scale: float) -> float:
    """Normalise ``value`` by ``scale`` into [-1, 1] without dividing by zero."""
    if not pd.notna(value) or not pd.notna(scale) or scale <= 1e-12:
        return 0.0
    return clip(value / scale)


class Agent(ABC):
    """Base class for all opinion generators."""

    #: Display name; also the key used in consensus breakdowns.
    name: str = "agent"
    #: Relative voting weight in the consensus layer.
    weight: float = 1.0
    #: Sessions of history required before the agent will speak.
    min_history: int = 30

    def fit(self, history: History) -> None:
        """Precompute indicators. Must be causal (no future information)."""

    @abstractmethod
    def signals_on(self, date: pd.Timestamp) -> Sequence[Signal]:
        """Opinions for every symbol it covers as of ``date`` (inclusive)."""

    def evaluate(self, history: History, date: pd.Timestamp) -> Sequence[Signal]:
        self.fit(history)
        return self.signals_on(date)
