"""Macro agent — India-specific top-down gates and tilts.

  • India VIX level & overnight shock (>12% spike ⇒ halve intraday size,
    >25 ⇒ no fresh swing books)
  • USD/INR jump (>0.5% session move) — risk-off for financials, tailwind for
    exporters (IT/pharma) — informational tilt only
  • Crude move (>3%) — inflation-sensitive sectors flagged
  • S&P 500 overnight cue — gap-risk awareness for the morning
"""

from __future__ import annotations

import numpy as np

from ..models import AgentReport
from .base import BaseAgent, MarketContext


class MacroAgent(BaseAgent):
    name = "Macro"
    description = "VIX / USDINR / crude / overnight-cue gates and sector tilts"

    def run(self, ctx: MarketContext) -> AgentReport:
        m = ctx.macro or {}
        vix = ctx.vix
        vix_chg = ctx.vix_chg
        usdinr_chg = m.get("usdinr_chg_pct")
        crude_chg = m.get("crude_chg_pct")
        spx_cue = m.get("spx_overnight_pct")

        score = 0.0
        flags: list = []

        # ---- VIX regime ----------------------------------------------------
        if vix > 25:
            score -= 45
            flags.append("VIX>25: swing books OFF")
        elif vix > 20:
            score -= 25
            flags.append("VIX>20: size down")
        elif vix < 12:
            score += 10
            flags.append("VIX<12: grind regime")

        if vix_chg > 12:
            score -= 30
            flags.append(f"VIX spike {vix_chg:+.0f}%: intraday size ×0.5")
        elif vix_chg < -10:
            score += 10

        # ---- FX --------------------------------------------------------------
        if usdinr_chg is not None:
            if usdinr_chg > 0.5:
                score -= 10
                flags.append(f"INR weak {usdinr_chg:+.2f}%")
            elif usdinr_chg < -0.5:
                score += 5
                flags.append(f"INR strong {usdinr_chg:+.2f}%")

        # ---- crude ------------------------------------------------------------
        if crude_chg is not None and abs(crude_chg) > 3:
            flags.append(f"crude {crude_chg:+.1f}%: OMC/paints sensitive")
            score -= 5 if crude_chg > 0 else 0

        # ---- overnight cue ------------------------------------------------------
        if spx_cue is not None:
            if spx_cue < -1.5:
                score -= 15
                flags.append(f"S&P overnight {spx_cue:+.1f}%: gap-down risk")
            elif spx_cue > 1.0:
                score += 8

        score = float(np.clip(score, -100, 100))
        note = f"VIX {vix:.1f} ({vix_chg:+.1f}%) | " + ("; ".join(flags[:3]) if flags else "calm")

        return AgentReport(
            agent=self.name, score=score, confidence=0.65,
            note=note,
            data={
                "vix": vix, "vix_chg_pct": round(vix_chg, 1),
                "usdinr_chg_pct": usdinr_chg, "crude_chg_pct": crude_chg,
                "spx_overnight_pct": spx_cue, "flags": flags,
            })
