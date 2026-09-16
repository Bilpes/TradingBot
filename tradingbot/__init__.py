"""Multi-agent, sector-aware equity strategy engine.

Design contract
---------------
1. Every agent emits a *bounded* opinion: a directional score in [-1, 1] and a
   confidence in [0, 1]. Agents never see each other's output.
2. A consensus layer aggregates those opinions into a single score plus a
   confidence, and it explicitly *penalises disagreement* rather than averaging
   it away.
3. Confidence is never treated as a probability of profit. It is a *sizing*
   input: it scales position size and gates entries. Risk limits are applied
   after it and cannot be overridden by it.
4. Execution is always one bar behind the decision. No agent may see a bar that
   has not closed at decision time (enforced by tests/test_no_lookahead.py).
"""

from tradingbot.agents.base import Agent, Signal
from tradingbot.core.confidence import Consensus, aggregate
from tradingbot.core.engine import Engine, EngineConfig
from tradingbot.core.risk import RiskConfig

__all__ = [
    "Agent",
    "Signal",
    "Consensus",
    "aggregate",
    "Engine",
    "EngineConfig",
    "RiskConfig",
    "__version__",
]

__version__ = "0.1.0"
