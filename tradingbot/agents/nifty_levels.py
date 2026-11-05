"""NiftyLevels agent — CPR, classic pivots, PDH/PDL, opening range.

The intraday playbook of every serious NSE scalper, formalised:
  • CPR (Central Pivot Range) from yesterday's H/L/C — narrow CPR ⇒ trend-day
    odds; price opening above CPR ⇒ bullish bias for the session.
  • Classic pivots R1/R2/S1/S2 as intra-day magnets/rejection zones.
  • Previous Day High/Low — the levels institutions defend.
  • 15-minute Opening Range (ORB) — the day's first decision zone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ta
from ..models import AgentReport
from .base import BaseAgent, MarketContext
from ..config import ORB_MINUTES


class NiftyLevelsAgent(BaseAgent):
    name = "NiftyLevels"
    description = "CPR / pivot / PDH-PDL / opening-range levels and session bias"

    def run(self, ctx: MarketContext) -> AgentReport:
        idx = ctx.index_daily
        intra = ctx.index_intraday
        if idx is None or len(idx) < 5:
            return AgentReport(agent=self.name, status="SKIP", note="no index data")

        prev = idx.iloc[-2] if len(idx) >= 2 else idx.iloc[-1]
        cpr = ta.central_pivot_range(prev["high"], prev["low"], prev["close"])
        piv = ta.classic_pivots(prev["high"], prev["low"], prev["close"])
        pdh, pdl = prev["high"], prev["low"]

        # weekly pivot (last 5 sessions)
        wk = idx.tail(6).head(5)
        wcpr = ta.central_pivot_range(wk["high"].max(), wk["low"].min(), wk["close"].iloc[-1])

        ltp = float(intra["close"].iloc[-1]) if intra is not None and len(intra) else float(idx["close"].iloc[-1])

        # opening range (first 15 min)
        orb_h = orb_l = None
        orb_width_atr = None
        if intra is not None and len(intra):
            intra = intra.copy()
            intra["ist"] = pd.to_datetime(intra["ts"], utc=True).dt.tz_convert("Asia/Kolkata")
            first = intra[intra["ist"].dt.strftime("%H:%M") < "09:30"]
            if len(first):
                orb_h, orb_l = float(first["high"].max()), float(first["low"].min())
                atr_idx = ta.atr(idx.tail(60), 14).iloc[-1]
                if atr_idx > 0:
                    orb_width_atr = (orb_h - orb_l) / atr_idx

        # session bias
        above_cpr = ltp > cpr["tc"]
        bias = "BULLISH" if above_cpr else ("BEARISH" if ltp < cpr["bc"] else "NEUTRAL")

        # trend-day heuristic: narrow CPR + open beyond CPR
        narrow = cpr["width_pct"] < 0.25
        trend_day = bool(narrow and bias != "NEUTRAL")

        score = {"BULLISH": 55, "NEUTRAL": 0, "BEARISH": -55}[bias]
        if trend_day and bias == "BULLISH":
            score = 75
        elif trend_day and bias == "BEARISH":
            score = -75
        score = float(np.clip(score + (10 if (orb_width_atr or 0) > 0.8 else 0), -100, 100))

        note = (f"{bias} | CPR {'narrow' if narrow else 'wide'} | "
                f"LTP {'>' if above_cpr else '<='} CPR-TC | PDH {pdh:.0f}/PDL {pdl:.0f}")

        return AgentReport(
            agent=self.name, score=score, confidence=0.7,
            note=note,
            data={
                "cpr": {k: round(v, 2) if isinstance(v, float) else v for k, v in cpr.items()},
                "weekly_cpr": {k: round(v, 2) if isinstance(v, float) else v for k, v in wcpr.items()},
                "pivots": {k: round(v, 1) for k, v in piv.items()},
                "pdh": round(float(pdh), 2), "pdl": round(float(pdl), 2),
                "orb_high": round(orb_h, 2) if orb_h else None,
                "orb_low": round(orb_l, 2) if orb_l else None,
                "orb_width_atr": round(orb_width_atr, 2) if orb_width_atr is not None else None,
                "narrow_cpr": narrow, "trend_day": trend_day, "bias": bias,
            })
