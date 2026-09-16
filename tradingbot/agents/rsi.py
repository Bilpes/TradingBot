"""RSI agent: contrarian in ranges, trend-following in trends.

A naive RSI agent shorts anything above 70 and gets run over in a strong
uptrend. This one reads the regime first via the Kaufman efficiency ratio:

* **Ranging** (low ER) -- fade extremes. RSI 75 is a sell.
* **Trending** (high ER) -- RSI extremes are *confirmations*, not warnings.
  An RSI of 75 in a persistent uptrend is strength, not exhaustion.

Confidence comes from how extreme RSI is *and* how clearly the regime is
identified. A mid-range RSI in an ambiguous regime produces almost no vote, so
the agent adds noise to the consensus as rarely as possible.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from tradingbot.agents.base import Agent, Signal, clip
from tradingbot.data.schema import History, row_asof
from tradingbot.indicators import efficiency_ratio, rsi


class RSIAgent(Agent):
    name = "rsi"
    weight = 0.9
    min_history = 30

    def __init__(
        self,
        window: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
        er_window: int = 20,
        trend_er: float = 0.45,
    ) -> None:
        if not 0 < oversold < overbought < 100:
            raise ValueError("need 0 < oversold < overbought < 100")
        self.window = window
        self.overbought = overbought
        self.oversold = oversold
        self.er_window = er_window
        self.trend_er = trend_er
        self.min_history = max(self.min_history, window + er_window)
        self._score: pd.DataFrame | None = None
        self._conf: pd.DataFrame | None = None
        self._rsi: pd.DataFrame | None = None

    def fit(self, history: History) -> None:
        r = pd.DataFrame({s: rsi(df["close"], self.window) for s, df in history.items()}).sort_index()
        er = pd.DataFrame(
            {s: efficiency_ratio(df["close"], self.er_window) for s, df in history.items()}
        ).sort_index()
        self._rsi = r

        mid = (self.overbought + self.oversold) / 2.0
        half_band = (self.overbought - self.oversold) / 2.0
        # Distance outside the neutral zone, signed and scaled to ~[-1, 1].
        stretch = (r - mid) / half_band

        # Regime blend: er=0 -> fully contrarian, er>=trend_er -> fully trend-following.
        trend_mix = (er / self.trend_er).clip(0.0, 1.0)
        contrarian = -stretch
        trending = stretch.clip(-1.5, 1.5) / 1.5
        self._score = ((1 - trend_mix) * contrarian + trend_mix * trending).clip(-1.0, 1.0)

        # Conviction: |stretch| beyond the band, scaled by how clear the regime is.
        extremity = ((r - mid).abs() / half_band - 1.0).clip(lower=0.0)
        regime_clarity = (er - 0.25).abs() * 2.0  # confident at either extreme of ER
        self._conf = (
            (0.35 + 0.5 * extremity.clip(upper=1.0)) * regime_clarity.clip(0.0, 1.0)
        ).clip(0.0, 0.85)

    def signals_on(self, date: pd.Timestamp) -> Sequence[Signal]:
        if self._score is None:
            raise RuntimeError("fit() must be called before signals_on()")
        srow = row_asof(self._score, date)
        crow = row_asof(self._conf, date)
        rrow = row_asof(self._rsi, date)
        if srow is None or crow is None:
            return []

        out = []
        for symbol in srow.index:
            score, conf = float(srow[symbol]), float(crow[symbol])
            if not np.isfinite(score) or not np.isfinite(conf):
                continue
            if conf < 0.05:
                continue
            r_val = float(rrow[symbol]) if rrow is not None else float("nan")
            out.append(
                Signal(
                    symbol=symbol,
                    score=clip(score),
                    confidence=conf,
                    agent=self.name,
                    reason=f"RSI({self.window})={r_val:.1f}, "
                           f"{'overbought' if r_val > self.overbought else 'oversold' if r_val < self.oversold else 'neutral'}",
                    parts={"rsi": r_val},
                )
            )
        return out
