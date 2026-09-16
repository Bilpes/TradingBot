"""Moving-average agent: ribbon alignment plus crossover state.

Uses three timeframes as a ribbon (default 20/50/200). The score blends:

* **Ribbon alignment** -- fast above medium above slow is a clean uptrend; a
  tangled ribbon is no trend at all and scores near zero.
* **Price position** -- close relative to each MA.
* **Crossover recency** -- a golden cross that just happened carries more
  conviction than one from six months ago.

Confidence is driven by ribbon *separation* normalised by price. When the three
averages are stacked on top of each other the trend is ambiguous and the agent
reports low conviction rather than guessing.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from tradingbot.agents.base import Agent, Signal, clip
from tradingbot.data.schema import History, closes, row_asof


class MovingAverageAgent(Agent):
    name = "moving_average"
    weight = 1.0
    min_history = 60

    def __init__(self, fast: int = 20, medium: int = 50, slow: int = 200, cross_recency: int = 10) -> None:
        if not fast < medium < slow:
            raise ValueError("need fast < medium < slow")
        self.fast = fast
        self.medium = medium
        self.slow = slow
        self.cross_recency = cross_recency
        self.min_history = max(self.min_history, slow)
        self._score: pd.DataFrame | None = None
        self._conf: pd.DataFrame | None = None

    def fit(self, history: History) -> None:
        px = closes(history)
        f = px.rolling(self.fast, min_periods=self.fast).mean()
        m = px.rolling(self.medium, min_periods=self.medium).mean()
        s = px.rolling(self.slow, min_periods=self.slow).mean()

        # Each pairwise relationship contributes +1 / -1 when ordered.
        align = (np.sign(f - m) + np.sign(m - s) + np.sign(px - m)) / 3.0

        spread = ((f - m) / m + (m - s) / s) / 2.0
        spread_score = np.tanh(spread / 0.03)

        # Recency-weighted crossover: +1 for `cross_recency` bars after a golden
        # cross, decaying linearly to 0.
        golden = ((f > m) & (f.shift(1) <= m.shift(1))).astype("float64")
        death = ((f < m) & (f.shift(1) >= m.shift(1))).astype("float64")
        weights = pd.Series(
            np.arange(self.cross_recency, 0, -1) / self.cross_recency,
            index=range(self.cross_recency),
        )
        recent = (
            golden.rolling(self.cross_recency, min_periods=1).apply(
                lambda w: float(np.dot(w, weights.to_numpy()[: len(w)])), raw=True
            )
            - death.rolling(self.cross_recency, min_periods=1).apply(
                lambda w: float(np.dot(w, weights.to_numpy()[: len(w)])), raw=True
            )
        )

        self._score = (0.45 * align + 0.35 * spread_score + 0.20 * recent).clip(-1.0, 1.0)

        # Separation between the averages, in price units.
        sep = ((f - m).abs() / m + (m - s).abs() / s) / 2.0
        self._conf = (sep / 0.04).clip(0.0, 0.85)

    def signals_on(self, date: pd.Timestamp) -> Sequence[Signal]:
        if self._score is None:
            raise RuntimeError("fit() must be called before signals_on()")
        srow = row_asof(self._score, date)
        crow = row_asof(self._conf, date)
        if srow is None or crow is None:
            return []

        out = []
        for symbol in srow.index:
            score, conf = float(srow[symbol]), float(crow[symbol])
            if not np.isfinite(score) or not np.isfinite(conf):
                continue
            if conf < 0.05:
                continue
            out.append(
                Signal(
                    symbol=symbol,
                    score=clip(score),
                    confidence=conf,
                    agent=self.name,
                    reason=f"MA{self.fast}/{self.medium}/{self.slow} ribbon",
                )
            )
        return out
