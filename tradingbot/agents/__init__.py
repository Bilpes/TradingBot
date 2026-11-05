"""Agent registry — import all nine agents here.

run_all_parallel() executes them concurrently per scan cycle.
"""

from .base import BaseAgent, MarketContext, run_all_parallel, AGENT_CLASSES
from .market_structure import MarketStructureAgent
from .nifty_levels import NiftyLevelsAgent
from .technical import TechnicalAgent
from .ml_prediction import MLPredictionAgent
from .timeseries import TimeSeriesAgent
from .sentiment import SentimentAgent
from .altdata import AltDataAgent
from .macro import MacroAgent
from .risk import RiskAgent, compute_sizing

__all__ = [
    "BaseAgent", "MarketContext", "run_all_parallel", "AGENT_CLASSES",
    "MarketStructureAgent", "NiftyLevelsAgent", "TechnicalAgent",
    "MLPredictionAgent", "TimeSeriesAgent", "SentimentAgent",
    "AltDataAgent", "MacroAgent", "RiskAgent", "compute_sizing",
]
