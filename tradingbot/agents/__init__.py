from tradingbot.agents.base import Agent, Signal
from tradingbot.agents.meanrev import MeanReversionAgent
from tradingbot.agents.momentum import CrossSectionalMomentumAgent
from tradingbot.agents.sector import SectorRotationAgent
from tradingbot.agents.trend import TrendAgent
from tradingbot.agents.volatility import VolatilityRegimeAgent

DEFAULT_AGENTS = (
    CrossSectionalMomentumAgent,
    TrendAgent,
    MeanReversionAgent,
    SectorRotationAgent,
    VolatilityRegimeAgent,
)


def build_default_agents() -> list[Agent]:
    return [cls() for cls in DEFAULT_AGENTS]


__all__ = [
    "Agent",
    "Signal",
    "CrossSectionalMomentumAgent",
    "TrendAgent",
    "MeanReversionAgent",
    "SectorRotationAgent",
    "VolatilityRegimeAgent",
    "DEFAULT_AGENTS",
    "build_default_agents",
]
