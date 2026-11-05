"""TimeSeries agent — statistical structure the ML models don't see.

  • Hurst exponent: persistence regime (H>0.55 trends, H<0.45 reverts)
  • EWMA volatility forecast (RiskMetrics λ=0.94) → position-size context
  • Holt linear-trend forecast, 5 sessions out → directional vote
  • Bollinger z-score: stretched-ness for mean-reversion exits
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .. import indicators as ta
from ..models import AgentReport
from .base import BaseAgent, MarketContext


class TimeSeriesAgent(BaseAgent):
    name = "TimeSeries"
    description = "Hurst persistence, EWMA vol forecast, Holt trend, Bollinger stretch"

    def run(self, ctx: MarketContext) -> AgentReport:
        df = ctx.daily
        if df is None or len(df) < 130:
            return AgentReport(agent=self.name, status="SKIP", note="need 130+ sessions")

        close = df["close"]
        logret = np.log(close / close.shift(1)).dropna()

        # ---- persistence ---------------------------------------------------
        h = ta.hurst_exponent(logret.tail(180))
        persistence = "trending" if h > 0.55 else ("mean-reverting" if h < 0.45 else "random-walk")

        # ---- vol forecast ----------------------------------------------------
        vol_fc = ta.ewma_vol(logret.tail(250), lam=0.94)          # annualised
        vol_pct = vol_fc * 100

        # ---- Holt forecast -----------------------------------------------------
        fc_dir = 0.0
        fc_ret = 0.0
        try:
            from statsmodels.tsa.holtwinters import Holt
            fit = Holt(close.tail(180).values, damped_trend=True, initialization_method="estimated").fit(
                smoothing_level=0.25, smoothing_trend=0.05, optimized=False)
            fc = fit.forecast(5)
            fc_ret = (fc[-1] / close.iloc[-1] - 1)
            fc_dir = float(np.clip(np.tanh(fc_ret * 40) * 60, -60, 60))
        except Exception:
            pass

        # ---- stretch -----------------------------------------------------------
        mid_up, mid, mid_dn = ta.bollinger(close, 20, 2.0)
        band = (mid_up - mid_dn).iloc[-1]
        z = (close.iloc[-1] - mid.iloc[-1]) / band if band > 0 else 0.0

        score = fc_dir
        if persistence == "trending":
            score += 15 if fc_dir >= 0 else -15          # trend-following book
        elif persistence == "mean-reverting":
            if z > 2.0:
                score -= 25                              # stretched, fade risk
            elif z < -2.0:
                score += 25
        score = float(np.clip(score, -100, 100))

        note = (f"H {h:.2f} {persistence} | vol-fc {vol_pct:.0f}% | "
                f"Holt5 {fc_ret * 100:+.1f}% | BBz {z:.1f}")

        return AgentReport(
            agent=self.name, score=score, confidence=0.55,
            note=note,
            data={
                "hurst": round(h, 3), "persistence": persistence,
                "vol_forecast_ann_pct": round(vol_pct, 1),
                "holt5_ret_pct": round(fc_ret * 100, 2),
                "bollinger_z": round(float(z), 2),
            })
