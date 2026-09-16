"""Portfolio book-keeping: cash, positions, fills, and exposure accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional

import pandas as pd


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class Order:
    symbol: str
    side: Side
    shares: int
    reason: str = ""
    created: Optional[pd.Timestamp] = None
    stop: Optional[float] = None


@dataclass(frozen=True)
class Fill:
    date: pd.Timestamp
    symbol: str
    side: Side
    shares: int
    price: float
    commission: float
    reason: str = ""

    @property
    def notional(self) -> float:
        return self.shares * self.price

    @property
    def cash_flow(self) -> float:
        """Signed cash impact: negative when buying, positive when selling."""
        signed = self.notional if self.side is Side.BUY else -self.notional
        return -(signed + self.commission)


@dataclass
class Portfolio:
    cash: float
    commission_bps: float = 5.0      # 0.05% per side
    slippage_bps: float = 5.0        # adverse price movement per side
    sector_of: Callable[[str], str] = lambda s: "Unknown"

    positions: Dict[str, int] = field(default_factory=dict)
    avg_cost: Dict[str, float] = field(default_factory=dict)
    stops: Dict[str, float] = field(default_factory=dict)
    fills: List[Fill] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))

    # -- accounting ---------------------------------------------------------
    def equity(self, prices: Dict[str, float]) -> float:
        holdings = sum(q * prices.get(s, 0.0) for s, q in self.positions.items() if q)
        return self.cash + holdings

    def weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        eq = self.equity(prices)
        if eq <= 0:
            return {}
        return {s: q * prices.get(s, 0.0) / eq for s, q in self.positions.items() if q}

    def gross_exposure(self, prices: Dict[str, float]) -> float:
        return sum(abs(w) for w in self.weights(prices).values())

    def sector_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for symbol, w in self.weights(prices).items():
            key = self.sector_of(symbol)
            out[key] = out.get(key, 0.0) + w
        return out

    # -- execution ----------------------------------------------------------
    def execute(self, order: Order, reference_price: float, date: pd.Timestamp) -> Optional[Fill]:
        """Fill ``order`` at ``reference_price`` plus slippage. Never overdrafts."""
        if order.shares <= 0 or reference_price <= 0:
            return None

        slip = self.slippage_bps / 10_000.0
        price = reference_price * (1 + slip) if order.side is Side.BUY else reference_price * (1 - slip)
        notional = order.shares * price
        commission = notional * self.commission_bps / 10_000.0

        if order.side is Side.BUY:
            if notional + commission > self.cash + 1e-9:
                # Spend what we actually have rather than failing silently.
                affordable = int((self.cash - commission) / (price * (1 + self.commission_bps / 10_000.0)))
                if affordable <= 0:
                    return None
                order = Order(order.symbol, order.side, affordable, order.reason, order.created, order.stop)
                notional = order.shares * price
                commission = notional * self.commission_bps / 10_000.0

            prev_q = self.positions.get(order.symbol, 0)
            prev_c = self.avg_cost.get(order.symbol, 0.0)
            new_q = prev_q + order.shares
            self.avg_cost[order.symbol] = (prev_q * prev_c + notional) / new_q if new_q else 0.0
            self.positions[order.symbol] = new_q
            if order.stop is not None:
                self.stops[order.symbol] = order.stop
            self.cash -= notional + commission
        else:
            held = self.positions.get(order.symbol, 0)
            if held <= 0:
                return None
            shares = min(order.shares, held)
            notional = shares * price
            commission = notional * self.commission_bps / 10_000.0
            self.positions[order.symbol] = held - shares
            self.cash += notional - commission
            if self.positions[order.symbol] == 0:
                self.positions.pop(order.symbol, None)
                self.avg_cost.pop(order.symbol, None)
                self.stops.pop(order.symbol, None)
            order = Order(order.symbol, order.side, shares, order.reason, order.created, order.stop)

        fill = Fill(date, order.symbol, order.side, order.shares, price, commission, order.reason)
        self.fills.append(fill)
        return fill

    def mark(self, date: pd.Timestamp, prices: Dict[str, float]) -> None:
        self.equity_curve.loc[date] = self.equity(prices)

    # -- trade reconstruction ----------------------------------------------
    def round_trips(self) -> List[dict]:
        """Pair fills into round trips using FIFO-ish average cost.

        Returns per-trade P&L including commissions, which is what win rate and
        profit factor are computed on. Gross-of-cost figures would flatter the
        strategy and are deliberately not reported.
        """
        open_pos: Dict[str, dict] = {}
        trades: List[dict] = []
        for f in self.fills:
            cost = f.notional + f.commission
            if f.side is Side.BUY:
                cur = open_pos.setdefault(f.symbol, {"shares": 0, "cost": 0.0, "entry": f.date})
                cur["shares"] += f.shares
                cur["cost"] += cost
            else:
                cur = open_pos.get(f.symbol)
                if not cur or cur["shares"] <= 0:
                    continue
                frac = f.shares / cur["shares"]
                entry_cost = cur["cost"] * frac
                proceeds = f.notional - f.commission
                trades.append(
                    {
                        "symbol": f.symbol,
                        "entry_date": cur["entry"],
                        "exit_date": f.date,
                        "shares": f.shares,
                        "pnl": proceeds - entry_cost,
                        "return": (proceeds - entry_cost) / entry_cost if entry_cost > 0 else 0.0,
                        "reason": f.reason,
                    }
                )
                cur["shares"] -= f.shares
                cur["cost"] -= entry_cost
                if cur["shares"] <= 0:
                    open_pos.pop(f.symbol, None)
        return trades
