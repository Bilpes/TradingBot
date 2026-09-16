"""Mean-reversion agent: fade stretched moves, but only in ranging regimes.

The naive version of this agent loses money in a trend, because it keeps
shorting strength all the way up. The fix is to make it *self-aware*: it
computes its own efficiency ratio and only expresses conviction when the market
is actually oscillating. In a strong trend it goes quiet rather than arguing
with the trend agent -- disagreement is then resolved by the consensus layer
through a confidence penalty, not by one agent shouting louder.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from tradingbot.agents.base import Agent, Signal, clip
from tradingbot.data.schema import History, closes, row_asof


class MeanReversionAgent(Agent):
    name = "mean_reversion"
    weight = 0.7
    min_history = 30

    def __init__(self, lookback: int = 20, er_window: int = 20, stretch: float = 2.0) -> None:
        self.lookback = lookback
        self.er_window = er_window
        self.stretch = stretch
        self.min_history = max(self.min_history, lookback, er_window)
        self._score: pd.DataFrame | None = None
        self._er: pd.DataFrame | None = None
        self._z: pd.DataFrame | None = None

    def fit(self, history: History) -> None:
        px = closes(history)

        ma = px.rolling(self.lookback, min_periods=self.lookback).mean()
        sd = px.rolling(self.lookback, min_periods=self.lookback).std().replace(0.0, np.nan)
        z = (px - ma) / sd
        self._z = z

        # Contrarian: above the mean -> negative score. tanh(z/stretch) makes a
        # 2-sigma stretch read as ~0.76 and a 4-sigma blow-off saturate at 1.
        self._score = (-np.tanh(z / self.stretch)).clip(-1.0, 1.0).fillna(0.0)

        delta = px.diff()
        net = delta.rolling(self.er_window, min_periods=self.er_window).sum().abs()
        path = delta.abs().rolling(self.er_window, min_periods=self.er_window).sum()
        self._er = (net / path.replace(0.0, np.nan)).fillna(0.0)

    def signals_on(self, date: pd.Timestamp) -> Sequence[Signal]:
        if self._score is None:
            raise RuntimeError("fit() must be called before signals_on()")

        srow = row_asof(self._score, date)
        erow = row_asof(self._er, date)
        zrow = row_asof(self._z, date)
        if srow is None or erow is None:
            return []

        out = []
        for symbol in srow.index:
            score = float(srow[symbol])
            er = float(erow[symbol])
            if not np.isfinite(score) or not np.isfinite(er):
                continue
            # Conviction only when the regime is choppy (low ER) AND the price
            # is actually stretched. A quiet, centred market earns nothing.
            regime = float(np.clip((0.45 - er) / 0.35, 0.0, 1.0))
            stretch = abs(float(zrow[symbol])) if zrow is not None and np.isfinite(float(zrow[symbol])) else 0.0
            stretched = float(np.clip(stretch / self.stretch, 0.0, 1.0))
            confidence = float(np.clip(0.85 * regime * stretched, 0.0, 0.8))
            if confidence < 0.02:
                continue  # stay silent instead of adding a low-information vote
            out.append(
                Signal(
                    symbol=symbol,
                    score=clip(score),
                    confidence=confidence,
                    agent=self.name,
                    reason=f"{stretch:+.1f} sigma vs {self.lookback}d mean, ER={er:.2f}",
                    parts={"efficiency_ratio": er, "z": stretch},
                )
            )
        return out
