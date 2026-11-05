"""MarketStructure agent — classifies the day/regime before anything trades.

Outputs a regime label + a score that biases the whole book:
  STRONG_TREND_UP / TREND_UP / NEUTRAL / TREND_DOWN / STRONG_TREND_DOWN / HIGH_VOL

Blend (20 years of NSE pattern-reading, encoded):
  • EMA20/50/200 stack + slope on the daily NIFTY
  • ADX(14) — trend strength (>25 trending, <20 range)
  • breadth — % of Nifty-50 constituents above their EMA50
  • India VIX level & shock (VIX spikes decide if intraday size is halved)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ta
from ..models import AgentReport
from .base import BaseAgent, MarketContext


class MarketStructureAgent(BaseAgent):
    name = "MarketStructure"
    description = "Index regime: trend/range/vol classification via EMA stack, ADX, breadth, VIX"

    def run(self, ctx: MarketContext) -> AgentReport:
        idx = ctx.index_daily
        if idx is None or len(idx) < 210:
            return AgentReport(agent=self.name, status="SKIP", note="not enough index history")

        close = idx["close"]
        e20 = ta.ema(close, 20).iloc[-1]
        e50 = ta.ema(close, 50).iloc[-1]
        e200 = ta.ema(close, 200).iloc[-1]
        slope20 = (e20 / ta.ema(close, 20).iloc[-6] - 1) * 100

        adx = ta.adx(idx.tail(120), 14).iloc[-1]
        px = close.iloc[-1]

        # ---- regime classification -------------------------------------
        stack_up = e20 > e50 > e200
        stack_dn = e20 < e50 < e200
        above200 = px > e200

        if ctx.vix > 24:
            regime = "HIGH_VOL"
        elif stack_up and adx > 22:
            regime = "STRONG_TREND_UP"
        elif stack_dn and adx > 22:
            regime = "STRONG_TREND_DOWN"
        elif above200 and slope20 > 0.2:
            regime = "TREND_UP"
        elif (not above200) and slope20 < -0.2:
            regime = "TREND_DOWN"
        else:
            regime = "NEUTRAL"

        # ---- score: how friendly is the regime for fresh longs? --------
        score_map = {
            "STRONG_TREND_UP": 70, "TREND_UP": 40, "NEUTRAL": 0,
            "TREND_DOWN": -40, "STRONG_TREND_DOWN": -70, "HIGH_VOL": -25,
        }
        score = score_map[regime]
        # breadth tilt: >60% above EMA50 → +10; <40% → −10
        if ctx.breadth >= 60:
            score += 10
        elif ctx.breadth <= 40:
            score -= 10
        # VIX shock tilt
        if ctx.vix_chg > 12:
            score -= 20
        elif ctx.vix_chg < -10:
            score += 5
        score = float(np.clip(score, -100, 100))

        note = (f"{regime} | ADX {adx:.0f} | EMA20/50/200 "
                f"{'↑' if stack_up else ('↓' if stack_dn else '→')} | breadth {ctx.breadth:.0f}% "
                f"| VIX {ctx.vix:.1f}")

        return AgentReport(
            agent=self.name, score=score,
            confidence=0.85 if adx > 25 or adx < 15 else 0.6,
            note=note,
            data={
                "regime": regime, "adx": round(float(adx), 1),
                "breadth": round(ctx.breadth, 1), "vix": ctx.vix,
                "vix_chg": round(ctx.vix_chg, 1),
                "ema_stack": "up" if stack_up else ("down" if stack_dn else "mixed"),
            })
