"""Cross-sectional momentum: buy relative winners, avoid relative losers.

Classic 12-1 construction (Jegadeesh & Titman style): measure the return over
``lookback`` sessions but *skip* the most recent ``skip`` sessions, so the
signal is not contaminated by the short-horizon reversal that dominates the
last month of returns. The raw return is then z-scored *across* the universe on
each date -- momentum is a ranking game, so only the relative position matters.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from tradingbot.agents.base import Agent, Signal, clip
from tradingbot.data.schema import History, closes, row_asof


class CrossSectionalMomentumAgent(Agent):
    name = "momentum"
    weight = 1.2
    min_history = 60

    def __init__(self, lookback: int = 126, skip: int = 21) -> None:
        if skip >= lookback:
            raise ValueError("skip must be smaller than lookback")
        self.lookback = lookback
        self.skip = skip
        self.min_history = max(self.min_history, lookback)
        self._z: pd.DataFrame | None = None
        self._mom: pd.DataFrame | None = None
        self._dispersion: pd.DataFrame | None = None

    def fit(self, history: History) -> None:
        px = closes(history)
        mom = px.shift(self.skip) / px.shift(self.lookback) - 1.0

        mu = mom.mean(axis=1)
        sd = mom.std(axis=1).replace(0.0, np.nan)
        z = mom.sub(mu, axis=0).div(sd, axis=0)

        # tanh keeps a single parabolic name from saturating the whole panel.
        self._mom = mom
        self._z = np.tanh(z / 1.5).fillna(0.0)
        # Cross-sectional spread of momentum: a wide spread means the ranking is
        # meaningful, a flat one means we are guessing.
        self._dispersion = mom.std(axis=1).to_frame("spread")

    def signals_on(self, date: pd.Timestamp) -> Sequence[Signal]:
        if self._z is None:
            raise RuntimeError("fit() must be called before signals_on()")

        zrow = row_asof(self._z, date)
        mrow = row_asof(self._mom, date)
        disp = row_asof(self._dispersion, date)
        if zrow is None or mrow is None:
            return []

        spread = float(disp["spread"]) if disp is not None else 0.0
        if not np.isfinite(spread):
            # A panel of one has no cross-sectional spread. Silence is correct;
            # a fabricated confidence would be a lie the consensus layer trusts.
            return []
        # Confidence rises with cross-sectional separation, capped at 0.9: a
        # ranking signal should never claim near-certainty on its own.
        base_conf = float(np.clip(spread / 0.20, 0.05, 0.90))

        out = []
        for symbol in zrow.index:
            score = float(zrow[symbol])
            raw = float(mrow[symbol])
            if not np.isfinite(score) or not np.isfinite(raw):
                continue
            out.append(
                Signal(
                    symbol=symbol,
                    score=clip(score),
                    confidence=base_conf,
                    agent=self.name,
                    reason=f"{self.lookback}-{self.skip}d rel. momentum {raw:+.1%}",
                    parts={"momentum": raw, "z": float(zrow[symbol])},
                )
            )
        return out
