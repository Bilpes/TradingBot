"""Risk agent — the last word before anything is proposed to the broker.

Portfolio-level guards (book health):
  • daily loss limit −2% → HALT fresh entries (kill-switch alert to Telegram)
  • max open positions (3), portfolio heat ≤ 3%
  • halted flag respected by Strategy layer

Per-symbol sizing math (fixed-fractional, the part that keeps you alive):
  risk_₹  = 1% × equity                (min ₹500, cap ₹2,000 at this scale)
  qty     = floor(risk_₹ / |entry − stop|)
  capped  by notional ≤ 25% equity, sector ≤ 40%, margin available
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from ..config import SETTINGS
from ..models import AgentReport
from .base import BaseAgent, MarketContext


class RiskAgent(BaseAgent):
    name = "Risk"
    description = "Kill-switch, heat limits, fixed-fractional position sizing"

    def run(self, ctx: MarketContext) -> AgentReport:
        pf = ctx.portfolio or {}
        equity = pf.get("equity", SETTINGS.capital_inr)
        day_pnl = pf.get("day_pnl", 0.0)
        heat = pf.get("heat_pct", 0.0)
        open_n = pf.get("open_positions", 0)
        trades_today = pf.get("trades_today", 0)
        halted = pf.get("halted", False)

        limit_hit = day_pnl <= -SETTINGS.daily_loss_limit_pct / 100.0 * SETTINGS.capital_inr
        limit_new = halted or limit_hit or open_n >= SETTINGS.max_open_positions or \
            trades_today >= SETTINGS.max_trades_per_day or heat >= 3.0

        score = 0.0 if not limit_new else -100
        notes = []
        if limit_hit:
            notes.append(f"DAY LOSS LIMIT hit ({day_pnl:+.0f} ₹)")
        elif halted:
            notes.append("manual HALT active")
        if open_n >= SETTINGS.max_open_positions:
            notes.append(f"positions full ({open_n}/{SETTINGS.max_open_positions})")
        if trades_today >= SETTINGS.max_trades_per_day:
            notes.append(f"trade cap reached ({trades_today})")
        if heat >= 3.0:
            notes.append(f"portfolio heat {heat:.1f}%")

        return AgentReport(
            agent=self.name,
            score=score,
            confidence=1.0,
            note="; ".join(notes) if notes else f"clear (heat {heat:.1f}%, {open_n} open)",
            data={
                "entries_allowed": not limit_new,
                "equity": equity, "day_pnl": day_pnl,
                "heat_pct": heat, "open_positions": open_n,
                "trades_today": trades_today,
                "risk_amount": SETTINGS.risk_amount_inr,
            })


def compute_sizing(entry: float, stop: float, equity: float,
                   current_notional: float = 0.0,
                   sector_notional: float = 0.0) -> Tuple[Optional[int], str]:
    """Fixed-fractional sizing → (qty, reason). qty=0 means 'too risky to take'."""
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0:
        return None, "zero risk distance"
    risk_amount = max(500.0, min(SETTINGS.risk_amount_inr, 2000.0))

    qty = math.floor(risk_amount / risk_per_share)

    # notional cap
    max_notional = equity * SETTINGS.max_notional_per_stock_pct / 100.0
    while qty > 0 and (qty * entry + current_notional) > max_notional:
        qty -= 1
    # sector cap
    max_sector = equity * SETTINGS.sector_concentration_pct / 100.0
    while qty > 0 and (qty * entry + sector_notional) > max_sector:
        qty -= 1

    if qty <= 0:
        return 0, "blocked by notional/sector cap"
    return qty, f"risk ₹{risk_amount:.0f} over ₹{risk_per_share:.2f}/share"
