"""Agent base class, shared MarketContext, and the parallel registry.

Every agent implements `run(ctx) -> AgentReport` and is defensive: an agent
must NEVER crash the engine — errors become ERROR reports with zero score.
Agents run concurrently via ThreadPoolExecutor (they are I/O and CPU bound
in roughly equal measure: news feeds, model inference, pandas work).
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from ..logger import get_logger
from ..models import AgentReport, Position

log = get_logger("agents")


# ----------------------------------------------------------------------------
@dataclass
class MarketContext:
    """Everything an agent may look at for ONE symbol in ONE cycle."""
    symbol: str
    style_hint: str                      # "intraday" | "swing" | "both"

    daily: pd.DataFrame                  # symbol daily OHLCV (≥200 rows)
    intraday: pd.DataFrame               # symbol 5-min bars for today
    index_daily: pd.DataFrame            # NIFTY daily
    index_intraday: pd.DataFrame         # NIFTY 5-min today
    vix: float = 14.0
    vix_chg: float = 0.0                 # VIX change vs prev session (%)

    sentiment: dict = field(default_factory=dict)    # symbol_news() output
    market_sentiment: dict = field(default_factory=dict)
    fii_dii: dict = field(default_factory=dict)
    macro: dict = field(default_factory=dict)        # usdinr, crude, spx cue...
    breadth: float = 50.0                            # % of universe above EMA50
    prev_index_day: Optional[pd.Series] = None       # yesterday's NIFTY OHLC row
    positions: List[Position] = field(default_factory=list)
    portfolio: dict = field(default_factory=dict)    # equity, day_pnl, heat...
    features_intraday: Optional[pd.DataFrame] = None  # engineered features (h=1)
    features_swing: Optional[pd.DataFrame] = None     # engineered features (h=5)


class BaseAgent:
    name = "Base"
    description = ""

    def __init__(self):
        self.log = get_logger(f"agent.{self.name}")

    def run(self, ctx: MarketContext) -> AgentReport:  # pragma: no cover
        raise NotImplementedError

    def guarded_run(self, ctx: MarketContext) -> AgentReport:
        t0 = time.time()
        try:
            rep = self.run(ctx)
            rep.elapsed_ms = int((time.time() - t0) * 1000)
            return rep
        except Exception as e:
            self.log.exception("agent %s failed on %s: %s", self.name, ctx.symbol, e)
            return AgentReport(agent=self.name, status="ERROR", score=0.0,
                               note=f"error: {e.__class__.__name__}",
                               elapsed_ms=int((time.time() - t0) * 1000))


# ----------------------------------------------------------------------------
AGENT_CLASSES: List[type] = []          # populated by agents/__init__.py


def run_all_parallel(agents: List[BaseAgent], ctx: MarketContext,
                     max_workers: int = 9) -> List[AgentReport]:
    """Fan out. Exceptions inside guarded_run are already normalised."""
    reports: List[AgentReport] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(a.guarded_run, ctx): a for a in agents}
        for fut in as_completed(futs):
            try:
                reports.append(fut.result())
            except Exception as e:  # belt & braces
                reports.append(AgentReport(agent=futs[fut].name, status="ERROR",
                                           note=str(e)))
    return sorted(reports, key=lambda r: r.agent)
