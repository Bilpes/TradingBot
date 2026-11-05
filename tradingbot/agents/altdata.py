"""Alternative-data agent — the footprints retail charts miss.

  • Volume anomaly: today's surge vs 20-day norm (z-score) — institutional
    participation detection
  • 52-week high/low proximity — where supply and trapped positions sit
  • Relative strength vs Nifty (20-session) — is money rotating INTO this name
  • FII/DII flow attempt (NSE endpoint, best-effort with cache fallback)

Each output nudges conviction rather than creating entries by itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ta
from ..models import AgentReport
from .base import BaseAgent, MarketContext


class AltDataAgent(BaseAgent):
    name = "AltData"
    description = "Volume anomalies, 52w position, relative strength, FII/DII flow"

    def run(self, ctx: MarketContext) -> AgentReport:
        df = ctx.daily
        if df is None or len(df) < 60:
            return AgentReport(agent=self.name, status="SKIP", note="insufficient history")

        close = df["close"]
        vol = df["volume"]

        # ---- volume anomaly (today vs 20d) ----------------------------------
        vol_mean = vol.rolling(20).mean().iloc[-2]
        vol_std = max(vol.rolling(20).std().iloc[-2], 1.0)
        vol_z = (vol.iloc[-1] - vol_mean) / vol_std
        up_day = close.iloc[-1] > close.iloc[-2]
        # surge on an up-day = accumulation; on a down-day = distribution
        if vol_z > 1.5:
            score = 25 if up_day else -25
        elif vol_z > 0.5:
            score = 10 if up_day else -10
        else:
            score = 0.0

        # ---- 52w position -----------------------------------------------------
        hi52 = close.rolling(252, min_periods=60).max().iloc[-1]
        lo52 = close.rolling(252, min_periods=60).min().iloc[-1]
        pos = (close.iloc[-1] - lo52) / max(hi52 - lo52, 1e-9)     # 0..1
        if pos > 0.8:
            score += 15       # near highs — momentum money here
        elif pos < 0.2:
            score -= 15       # near lows — supply overhead

        # ---- relative strength -------------------------------------------------
        rs = ta.relative_strength(close, ctx.index_daily["close"], 20) \
            if ctx.index_daily is not None and len(ctx.index_daily) > 25 else 0.0
        score += float(np.clip(rs * 3, -20, 20))

        # ---- FII/DII (best effort) ---------------------------------------------
        fii = ctx.fii_dii.get("fii_eq_cr")
        dii = ctx.fii_dii.get("dii_eq_cr")
        if fii is not None and fii > 2000:
            score += 10
        elif fii is not None and fii < -2000:
            score -= 10
        # delivery % if present
        dely = ctx.fii_dii.get("delivery_pct")
        if dely is not None and dely > 70 and up_day:
            score += 5

        score = float(np.clip(score, -100, 100))
        note = (f"vol z {vol_z:+.1f} {'accum' if (vol_z > 1 and up_day) else ''} | "
                f"52w pos {pos:.0%} | RS20 {rs:+.1f}%")

        return AgentReport(
            agent=self.name, score=score, confidence=0.6,
            note=note,
            data={
                "vol_z": round(float(vol_z), 2),
                "up_day": bool(up_day),
                "week52_pos": round(float(pos), 2),
                "rel_strength_20": round(float(rs), 2),
                "fii_eq_cr": fii, "dii_eq_cr": dii,
                "delivery_pct": dely,
            })
