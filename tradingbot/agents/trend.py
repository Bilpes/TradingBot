"""Trend / breakout agent: EMA spread plus Donchian channel position.

Two self-contained components, averaged:

* **EMA spread** -- normalised distance between a fast and slow EMA. Positive
  and widening means the trend has room; a spread near zero means no edge.
* **Channel position** -- where the close sits inside its own ``channel``-day
  high/low range. Near the top is a breakout, near the bottom a breakdown.

Confidence is driven by the *consistency* of the trend (an efficiency ratio),
not by its size. A violent but choppy move is a worse trend than a boring
persistent one, and this agent says so.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from tradingbot.agents.base import Agent, Signal, clip
from tradingbot.data.schema import History, closes, row_asof


class TrendAgent(Agent):
    name = "trend"
    weight = 1.0
    min_history = 40

    def __init__(self, fast: int = 10, slow: int = 50, channel: int = 60, er_window: int = 30) -> None:
        if fast >= slow:
            raise ValueError("fast EMA must be shorter than slow EMA")
        self.fast = fast
        self.slow = slow
        self.channel = channel
        self.er_window = er_window
        self.min_history = max(self.min_history, slow, channel, er_window)
        self._score: pd.DataFrame | None = None
        self._er: pd.DataFrame | None = None

    def fit(self, history: History) -> None:
        px = closes(history)

        ema_f = px.ewm(span=self.fast, adjust=False).mean()
        ema_s = px.ewm(span=self.slow, adjust=False).mean()
        # Normalise by the slow EMA so the spread is comparable across prices.
        spread = (ema_f - ema_s) / ema_s
        spread_score = np.tanh(spread / 0.05)

        hi = px.rolling(self.channel, min_periods=self.channel).max()
        lo = px.rolling(self.channel, min_periods=self.channel).min()
        rng = (hi - lo).replace(0.0, np.nan)
        pos = (px - lo) / rng
        # Map [0, 1] channel position to [-1, 1]; flat channels -> 0, not NaN.
        chan_score = (2.0 * pos - 1.0).fillna(0.0)

        self._score = (0.6 * spread_score + 0.4 * chan_score).clip(-1.0, 1.0)

        # Kaufman efficiency ratio: net move over the sum of daily moves.
        delta = px.diff()
        net = delta.rolling(self.er_window, min_periods=self.er_window).sum().abs()
        path = delta.abs().rolling(self.er_window, min_periods=self.er_window).sum()
        self._er = (net / path.replace(0.0, np.nan)).fillna(0.0)

    def signals_on(self, date: pd.Timestamp) -> Sequence[Signal]:
        if self._score is None:
            raise RuntimeError("fit() must be called before signals_on()")

        srow = row_asof(self._score, date)
        erow = row_asof(self._er, date)
        if srow is None or erow is None:
            return []

        out = []
        for symbol in srow.index:
            score = float(srow[symbol])
            er = float(erow[symbol])
            if not np.isfinite(score) or not np.isfinite(er):
                continue
            out.append(
                Signal(
                    symbol=symbol,
                    score=clip(score),
                    # ER of ~0.5+ is a genuinely persistent trend; below ~0.2
                    # the "trend" is mostly noise, so conviction collapses.
                    confidence=float(np.clip(er * 1.4, 0.0, 0.85)),
                    agent=self.name,
                    reason=f"EMA{self.fast}/{self.slow} + {self.channel}d channel, ER={er:.2f}",
                    parts={"efficiency_ratio": er},
                )
            )
        return out
