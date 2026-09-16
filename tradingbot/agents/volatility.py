"""Volatility regime agent -- a *gate*, not a voter.

This agent has no directional opinion and never claims one. It answers a
different question: "is the current risk environment one where the directional
agents' opinions deserve full size?"

It measures realised volatility of the equal-weight universe against its own
long-run distribution. In calm or normal regimes it passes confidence through
close to 1.0. In an elevated regime it scales conviction down hard, which
shrinks every position the risk layer takes. That is the mechanism by which a
system can be right about direction and still get smaller when the tape turns
violent -- the failure mode that kills most momentum bots.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from tradingbot.agents.base import Agent, Signal
from tradingbot.data.schema import History, closes, row_asof


class VolatilityRegimeAgent(Agent):
    name = "vol_regime"
    weight = 1.0
    min_history = 40

    def __init__(self, vol_window: int = 21, ref_window: int = 252, floor: float = 0.15) -> None:
        if not 0.0 <= floor <= 1.0:
            raise ValueError("floor must be in [0, 1]")
        self.vol_window = vol_window
        self.ref_window = ref_window
        self.floor = floor
        self.min_history = max(self.min_history, vol_window)
        self._mult: pd.DataFrame | None = None
        self._vol: pd.DataFrame | None = None

    def fit(self, history: History) -> None:
        px = closes(history)
        # Equal-weight universe return -> a single market-level risk series.
        mkt = px.pct_change().mean(axis=1)

        realised = mkt.rolling(self.vol_window, min_periods=self.vol_window).std() * np.sqrt(252)
        reference = (
            mkt.rolling(self.ref_window, min_periods=self.vol_window).std().shift(1) * np.sqrt(252)
        )

        ratio = (realised / reference.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        # 1.0 at or below the reference level, decaying above it, never below
        # the floor so the system degrades instead of switching off entirely.
        mult = (1.0 / ratio.clip(lower=1.0)).fillna(1.0).clip(self.floor, 1.0)

        self._mult = mult.to_frame("mult")
        self._vol = pd.concat([realised.to_frame("realised"), reference.to_frame("reference")], axis=1)

    def signals_on(self, date: pd.Timestamp) -> Sequence[Signal]:
        if self._mult is None:
            raise RuntimeError("fit() must be called before signals_on()")

        mrow = row_asof(self._mult, date)
        vrow = row_asof(self._vol, date)
        if mrow is None:
            return []

        mult = float(mrow["mult"])
        realised = float(vrow["realised"]) if vrow is not None else float("nan")

        return [
            Signal(
                symbol="*",
                score=0.0,
                confidence=mult,
                agent=self.name,
                role="gate",
                reason=f"realised vol {realised:.1%} vs reference -> x{mult:.2f}",
                parts={"multiplier": mult, "realised_vol": realised},
            )
        ]
