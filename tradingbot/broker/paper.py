"""Paper broker: simulated fills with realistic India charges & slippage.

Fills execute at last traded price ± slippage (default 5 bps). The India levy
model (STT / exchange txn / SEBI / GST / stamp duty) mirrors Dhan's published
pricing so paper P&L ≈ live P&L when the user eventually flips the switch.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

from ..config import CHARGES, SETTINGS
from ..logger import get_logger
from ..models import Fill

log = get_logger("paper")


def estimate_charges(symbol: str, side: str, qty: int, price: float, style: str) -> float:
    """Total brokerage+levies for ONE leg (buy or sell). style: CNC | MIS."""
    turnover = qty * price
    if style == "CNC":
        brokerage = 0.0                                   # Dhan: ₹0 delivery
        stt = CHARGES["stt_cnc"] * turnover
        stamp = CHARGES["stamp_cnc_buy"] * turnover if side == "BUY" else 0.0
    else:
        brokerage = min(CHARGES["intraday_brokerage_pct"] * turnover,
                        CHARGES["intraday_brokerage_cap"])
        stt = CHARGES["stt_mis_sell"] * turnover if side == "SELL" else 0.0
        stamp = CHARGES["stamp_mis_buy"] * turnover if side == "BUY" else 0.0
    txn = CHARGES["exchange_txn"] * turnover
    sebi = CHARGES["sebi"] * turnover
    gst = CHARGES["gst"] * (brokerage + txn + sebi)
    return round(brokerage + stt + txn + sebi + gst + stamp, 2)


class PaperBroker:
    """Simulated order execution. Same interface as the live Dhan path."""

    def __init__(self, capital: float = None):
        self.cash: float = float(SETTINGS.capital_inr if capital is None else capital)
        self.slippage_bps = SETTINGS.slippage_bps
        self.orders_today: int = 0

    # ------------------------------------------------------------------
    def _slippage(self, price: float, side: str) -> float:
        tick = price * self.slippage_bps / 10_000.0
        return price + tick if side == "BUY" else price - tick

    def execute(self, symbol: str, side: str, qty: int, price: float,
                style: str) -> Optional[Fill]:
        """market-fill both sides. Returns Fill or None (insufficient cash)."""
        fill_price = round(self._slippage(price, side), 2)
        charges = estimate_charges(symbol, side, qty, fill_price, style)
        value = fill_price * qty

        if side == "BUY":
            if value + charges > self.cash:
                log.warning("paper reject %s: need ₹%.0f, cash ₹%.0f", symbol, value + charges, self.cash)
                return None
            self.cash -= (value + charges)
        else:
            self.cash += (value - charges)

        self.orders_today += 1
        fill = Fill(
            symbol=symbol, side=side, qty=qty, price=fill_price,
            charges=charges,
            order_id=f"PAPER-{datetime.now().strftime('%H%M%S')}-{self.orders_today}",
            ts=datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat(timespec="seconds"),
        )
        log.info("PAPER FILL %s %s %d @ ₹%.2f charges ₹%.2f (%s)",
                 side, symbol, qty, fill_price, charges, style)
        return fill

    # rounded lot-free equity qty (min 1 share)
    @staticmethod
    def clip_qty(qty: int) -> int:
        return max(int(qty), 0)
