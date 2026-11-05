"""Portfolio: equity curve, P&L in ₹, exposure & heat tracking."""

from __future__ import annotations

from typing import Dict, List

from ..config import SETTINGS
from ..models import Position


class Portfolio:
    def __init__(self, capital: float):
        self.initial_capital = float(capital)
        self.realized = 0.0            # net of charges, lifetime
        self.realized_day = 0.0
        self.unrealized = 0.0          # MTM of open positions (updated by engine)
        self.halted = False            # manual kill-switch

    # ------------------------------------------------------------------
    @property
    def equity(self) -> float:
        return self.initial_capital + self.realized + self.unrealized

    def summary(self, positions: List[Position], trades_today: int) -> Dict:
        self.unrealized = sum(p.open_pnl_after_charges for p in positions)
        exposure = sum(p.notional for p in positions)
        heat_pct = sum(
            (abs(p.entry_price - p.stop) * p.qty) / self.equity * 100
            for p in positions if self.equity > 0)
        return {
            "capital": self.initial_capital,
            "equity": round(self.equity, 2),
            "realized_total": round(self.realized, 2),
            "realized_day": round(self.realized_day, 2),
            "unrealized": round(self.unrealized, 2),
            "day_pnl": round(self.realized_day + self.unrealized, 2),
            "total_pnl": round(self.realized + self.unrealized, 2),
            "total_pnl_pct": round((self.realized + self.unrealized) / self.initial_capital * 100, 2),
            "exposure": round(exposure, 2),
            "exposure_pct": round(exposure / self.equity * 100, 1) if self.equity else 0,
            "heat_pct": round(heat_pct, 2),
            "open_positions": len(positions),
            "max_positions": SETTINGS.max_open_positions,
            "trades_today": trades_today,
            "halted": self.halted,
            "risk_per_trade": SETTINGS.risk_amount_inr,
        }
