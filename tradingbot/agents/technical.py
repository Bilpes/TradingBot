"""Technical agent — multi-timeframe trend + momentum + participation quality.

Long-only logic (longs are where Indian retail equity alpha lives) mirrored
for intraday shorts when the tape says so:

  daily:  EMA9>EMA21>EMA50 stack, RSI 45–68 sweet spot in uptrends,
          MACD histogram direction, price vs EMA50 (pullback depth),
          ADX filter, volume confirmation.
  intraday (5m): price vs session VWAP, VWAP slope, ORB position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ta
from ..models import AgentReport
from .base import BaseAgent, MarketContext


class TechnicalAgent(BaseAgent):
    name = "Technical"
    description = "EMA stack, RSI, MACD, ADX, volume, VWAP — multi-timeframe alignment"

    def run(self, ctx: MarketContext) -> AgentReport:
        df = ctx.daily
        if df is None or len(df) < 60:
            return AgentReport(agent=self.name, status="SKIP", note="insufficient history")

        close = df["close"]
        ltp = float(close.iloc[-1])

        e9 = ta.ema(close, 9)
        e21 = ta.ema(close, 21)
        e50 = ta.ema(close, 50)
        e200 = ta.ema(close, 200) if len(close) >= 200 else None
        rsi = ta.rsi(close, 14).iloc[-1]
        _, _, hist = ta.macd(close)
        hist_now = hist.iloc[-1]
        hist_rising = hist.iloc[-1] > hist.iloc[-2]
        adx = ta.adx(df.tail(120), 14).iloc[-1]
        vol = df["volume"]
        vol_z = (vol.iloc[-1] - vol.rolling(20).mean().iloc[-1]) / max(vol.rolling(20).std().iloc[-1], 1)

        score = 0.0
        parts = []

        # ---- trend stack (daily) ----------------------------------------
        stack = e9.iloc[-1] > e21.iloc[-1] > e50.iloc[-1]
        anti_stack = e9.iloc[-1] < e21.iloc[-1] < e50.iloc[-1]
        if stack:
            score += 30
            parts.append("EMA stack ↑")
        elif anti_stack:
            score -= 30
            parts.append("EMA stack ↓")
        if e200 is not None:
            if ltp > e200.iloc[-1]:
                score += 10
            else:
                score -= 10

        # ---- RSI zone (sweet spot 45–68 in uptrends) ---------------------
        if 45 <= rsi <= 68:
            score += 12
            parts.append(f"RSI {rsi:.0f} zone")
        elif rsi > 75:
            score -= 8
            parts.append(f"RSI {rsi:.0f} hot")
        elif rsi < 35:
            score -= 10
            parts.append(f"RSI {rsi:.0f} weak")

        # ---- MACD momentum ----------------------------------------------
        if hist_now > 0 and hist_rising:
            score += 12
        elif hist_now < 0 and not hist_rising:
            score -= 12

        # ---- volume participation ----------------------------------------
        if vol_z > 1.0:
            score += 8
            parts.append(f"vol z {vol_z:.1f}")
        elif vol_z < -1.0:
            score -= 4

        # ---- ADX quality filter ------------------------------------------
        if adx > 25:
            score += 6
        elif adx < 15:
            score -= 4

        # ---- intraday VWAP context ---------------------------------------
        vwap_note = ""
        if ctx.intraday is not None and len(ctx.intraday) > 10:
            vwap = ta.vwap(ctx.intraday.tail(75))
            vw = vwap.iloc[-1]
            vwap_slope = vw - vwap.iloc[max(0, len(vwap) - 7)]
            if ltp > vw and vwap_slope > 0:
                score += 10
                vwap_note = "above rising VWAP"
            elif ltp < vw and vwap_slope < 0:
                score -= 10
                vwap_note = "below falling VWAP"
            else:
                vwap_note = "mixed VWAP"

        score = float(np.clip(score, -100, 100))
        note = f"{parts[0] if parts else 'neutral'} · RSI {rsi:.0f} · ADX {adx:.0f} · {vwap_note or 'no intraday'}"

        return AgentReport(
            agent=self.name, score=score,
            confidence=0.7 if adx > 20 else 0.5,
            note=note,
            data={
                "rsi": round(float(rsi), 1), "adx": round(float(adx), 1),
                "vol_z": round(float(vol_z), 2),
                "hist_rising": bool(hist_rising),
                "stack": "up" if stack else ("down" if anti_stack else "mixed"),
                "vwap": vwap_note,
            })
