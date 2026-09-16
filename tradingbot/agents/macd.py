"""MACD agent: line/signal crossovers confirmed by histogram momentum.

Three pieces, combined:

* **Histogram sign and slope** -- is momentum building or fading? The slope
  matters more than the level: a shrinking histogram is an early warning that
  the crossover is about to reverse.
* **Line vs signal** -- the classic cross.
* **Line vs zero** -- a cross above zero means the longer-term trend agrees,
  which is worth more conviction than a cross in isolation.

Confidence is the histogram normalised against its *own* recent dispersion, so
a stock with naturally wide MACD swings is not rewarded for a move that is
ordinary for it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from tradingbot.agents.base import Agent, Signal, clip
from tradingbot.data.schema import History, closes, row_asof
from tradingbot.indicators import macd


class MACDAgent(Agent):
    name = "macd"
    weight = 1.0
    min_history = 40

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, slope_window: int = 3) -> None:
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.slope_window = slope_window
        self.min_history = max(self.min_history, slow + signal)
        self._score: pd.DataFrame | None = None
        self._conf: pd.DataFrame | None = None
        self._hist: pd.DataFrame | None = None

    def fit(self, history: History) -> None:
        px = closes(history)
        frames = {s: macd(df["close"], self.fast, self.slow, self.signal)
                  for s, df in history.items()}

        hist = pd.DataFrame({s: f["histogram"] for s, f in frames.items()}).sort_index()
        line = pd.DataFrame({s: f["macd"] for s, f in frames.items()}).sort_index()
        self._hist = hist

        # Scale by the slow EMA so the histogram is comparable across prices.
        scale = px.ewm(span=self.slow, adjust=False).mean()
        norm = hist / scale.replace(0.0, np.nan)

        slope = hist.diff(self.slope_window) / scale.replace(0.0, np.nan)

        # Normalise against the histogram's own dispersion -> self-calibrating.
        hist_sd = norm.rolling(60, min_periods=20).std().replace(0.0, np.nan)
        hist_z = norm / hist_sd
        slope_z = slope / hist_sd

        score = 0.45 * np.tanh(hist_z) + 0.35 * np.tanh(slope_z) + 0.20 * np.tanh(line / scale)
        self._score = score.clip(-1.0, 1.0)

        # Conviction grows with |z| but is capped: a huge MACD move is often a
        # gap or a corporate action, not information.
        self._conf = (hist_z.abs() / 2.0).clip(0.0, 0.85).where(hist_z.notna(), 0.0)

    def signals_on(self, date: pd.Timestamp) -> Sequence[Signal]:
        if self._score is None:
            raise RuntimeError("fit() must be called before signals_on()")
        srow = row_asof(self._score, date)
        crow = row_asof(self._conf, date)
        hrow = row_asof(self._hist, date)
        if srow is None or crow is None:
            return []

        out = []
        for symbol in srow.index:
            score, conf = float(srow[symbol]), float(crow[symbol])
            if not np.isfinite(score) or not np.isfinite(conf):
                continue
            if conf < 0.05:
                continue
            h = float(hrow[symbol]) if hrow is not None else float("nan")
            out.append(
                Signal(
                    symbol=symbol,
                    score=clip(score),
                    confidence=conf,
                    agent=self.name,
                    reason=f"MACD({self.fast},{self.slow},{self.signal}) hist={h:.3f}",
                    parts={"histogram": h},
                )
            )
        return out
