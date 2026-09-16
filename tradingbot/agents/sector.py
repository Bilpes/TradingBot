"""Sector rotation: the "multiple agents tracking sectors" piece.

This agent does not look at stocks as isolated time series. It builds an
equal-weight index for each GICS sector, ranks the sectors against each other,
and then passes that sector-level view down to every constituent. A name in the
top-ranked sector inherits a positive bias even before its own chart is
considered -- that is the information the single-stock agents structurally
cannot see.

Two ingredients per sector:

* **Relative strength** -- cross-sectionally ranked sector return.
* **Breadth** -- the share of constituents trading above their own moving
  average. A sector up on one mega-cap is a worse rotation target than a sector
  where nearly everything is participating.

Confidence comes from sector *dispersion*. When every sector is doing the same
thing, rotation has nothing to say and this agent correctly reports low
conviction instead of inventing a view.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd

from tradingbot.agents.base import Agent, Signal, clip
from tradingbot.data.schema import History, closes, row_asof
from tradingbot.data.universe import sector_of


class SectorRotationAgent(Agent):
    name = "sector_rotation"
    weight = 1.1
    min_history = 60

    def __init__(
        self,
        lookback: int = 63,
        breadth_ma: int = 50,
        strength_weight: float = 0.65,
        sector_map: Callable[[str], str] = sector_of,
    ) -> None:
        if not 0.0 <= strength_weight <= 1.0:
            raise ValueError("strength_weight must be in [0, 1]")
        self.lookback = lookback
        self.breadth_ma = breadth_ma
        self.strength_weight = strength_weight
        self.sector_map = sector_map
        self.min_history = max(self.min_history, lookback, breadth_ma)
        self._score: pd.DataFrame | None = None
        self._dispersion: pd.DataFrame | None = None
        self._sector_rank: pd.DataFrame | None = None

    def fit(self, history: History) -> None:
        px = closes(history)

        by_sector: Dict[str, list[str]] = defaultdict(list)
        for symbol in px.columns:
            by_sector[self.sector_map(symbol)].append(symbol)

        members = {s: px[cols] for s, cols in by_sector.items()}

        # Equal-weight sector index, then its trailing return.
        sector_px = pd.DataFrame({s: df.mean(axis=1) for s, df in members.items()})
        sector_ret = sector_px / sector_px.shift(self.lookback) - 1.0

        mu = sector_ret.mean(axis=1)
        sd = sector_ret.std(axis=1).replace(0.0, np.nan)
        sector_z = sector_ret.sub(mu, axis=0).div(sd, axis=0)
        strength = np.tanh(sector_z / 1.0)
        self._sector_rank = strength
        self._dispersion = sector_ret.std(axis=1).to_frame("spread")

        # Breadth: fraction of each sector's members above their own MA.
        breadth = pd.DataFrame(
            {s: (df > df.rolling(self.breadth_ma, min_periods=self.breadth_ma).mean()).mean(axis=1)
             for s, df in members.items()}
        )
        breadth_score = (2.0 * breadth - 1.0).fillna(0.0)

        w = self.strength_weight
        sector_score = (w * strength.fillna(0.0) + (1.0 - w) * breadth_score).clip(-1.0, 1.0)

        # Fan the sector-level score back out to each constituent.
        columns = {}
        for sector, cols in by_sector.items():
            vals = sector_score[sector].to_numpy(dtype="float64")
            for symbol in cols:
                columns[symbol] = vals
        self._score = pd.DataFrame(columns, index=px.index)[list(px.columns)].astype("float64")

    def signals_on(self, date: pd.Timestamp) -> Sequence[Signal]:
        if self._score is None:
            raise RuntimeError("fit() must be called before signals_on()")

        srow = row_asof(self._score, date)
        drow = row_asof(self._dispersion, date)
        if srow is None:
            return []

        spread = float(drow["spread"]) if drow is not None else 0.0
        if not np.isfinite(spread):
            # A cross-section of one sector has no dispersion to report. Staying
            # silent is correct; inventing a confidence would not be.
            return []
        confidence = float(np.clip(spread / 0.12, 0.05, 0.85))

        out = []
        for symbol in srow.index:
            score = float(srow[symbol])
            if not np.isfinite(score):
                continue
            out.append(
                Signal(
                    symbol=symbol,
                    score=clip(score),
                    confidence=confidence,
                    agent=self.name,
                    reason=f"sector {self.sector_map(symbol)} spread {spread:.3f}",
                    parts={"sector_spread": spread},
                )
            )
        return out
