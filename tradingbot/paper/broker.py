"""Paper trading with a realistic NSE cost model.

Paper trading that ignores costs teaches you nothing: a strategy with a thin
edge is often entirely consumed by STT, exchange charges and GST. Every rate
below is a named constant so the numbers on the dashboard can be audited
line by line.

Default mode is delivery (CNC), matching a two-week swing paper trade. Switch
``product`` to ``INTRADAY`` and the STT/DP treatment changes accordingly.

**These rates are the ones published broadly for NSE equity and are provided so
the model is complete -- verify them against your own Dhan contract note before
comparing paper results to real fills.**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class NSECostModel:
    """Per-transaction charges, as fractions of turnover unless noted."""

    product: str = "CNC"            # CNC = delivery, INTRADAY = MIS
    brokerage_per_order: float = 0.0     # Dhan: Rs 0 delivery, Rs 20 intraday
    stt_delivery: float = 0.001          # 0.1% both sides
    stt_intraday_sell: float = 0.00025   # 0.025% sell side only
    exchange_txn: float = 0.0000297      # NSE 0.00297%
    sebi_turnover: float = 0.000001      # 0.0001%
    stamp_buy: float = 0.00015           # 0.015% on buy (delivery)
    gst_rate: float = 0.18               # on brokerage + exchange charges
    dp_charge_per_sell: float = 20.0     # flat, per scrip per day (approx)
    slippage_bps: float = 10.0           # our own execution assumption

    def charges(self, side: str, quantity: int, price: float) -> Dict[str, float]:
        turnover = quantity * price
        intraday = self.product.upper() == "INTRADAY"

        brokerage = self.brokerage_per_order or (20.0 if intraday else 0.0)
        if side == "BUY":
            stt = 0.0 if intraday else turnover * self.stt_delivery
        else:
            stt = turnover * (self.stt_intraday_sell if intraday else self.stt_delivery)

        exch = turnover * self.exchange_txn
        sebi = turnover * self.sebi_turnover
        stamp = turnover * self.stamp_buy if side == "BUY" else 0.0
        gst = (brokerage + exch) * self.gst_rate
        dp = 0.0 if (side == "BUY" or intraday) else self.dp_charge_per_sell

        return {
            "brokerage": brokerage,
            "stt": stt,
            "exchange": exch,
            "sebi": sebi,
            "stamp": stamp,
            "gst": gst,
            "dp": dp,
            "total": brokerage + stt + exch + sebi + stamp + gst + dp,
        }


@dataclass
class PaperPosition:
    symbol: str
    sector: str
    quantity: int
    avg_price: float
    opened: str
    stop: Optional[float] = None

    def market_value(self, ltp: float) -> float:
        return self.quantity * ltp

    def pnl(self, ltp: float) -> float:
        return (ltp - self.avg_price) * self.quantity


@dataclass
class PaperBroker:
    """A simulated book with real cost accounting.

    Fills are applied at the *reference* price adjusted by slippage, never at a
    better price than the market offered. Quantity is trimmed to what cash
    allows rather than allowing an overdraft, and integer shares only -- NSE
    equities trade in single units.
    """

    capital: float = 100_000.0
    costs: NSECostModel = field(default_factory=NSECostModel)
    cash: float = field(init=False)
    positions: Dict[str, PaperPosition] = field(default_factory=dict, init=False)
    trade_log: List[dict] = field(default_factory=list, init=False)
    peak_equity: float = field(init=False)

    def __post_init__(self) -> None:
        self.cash = self.capital
        self.peak_equity = self.capital

    # -- accounting ---------------------------------------------------------
    def equity(self, prices: Dict[str, float]) -> float:
        return self.cash + sum(
            p.market_value(prices.get(p.symbol, p.avg_price))
            for p in self.positions.values()
        )

    def invested(self, prices: Dict[str, float]) -> float:
        return sum(p.market_value(prices.get(p.symbol, p.avg_price))
                   for p in self.positions.values())

    def exposure(self, prices: Dict[str, float]) -> float:
        eq = self.equity(prices)
        return self.invested(prices) / eq if eq > 0 else 0.0

    def drawdown(self, prices: Dict[str, float]) -> float:
        eq = self.equity(prices)
        self.peak_equity = max(self.peak_equity, eq)
        return eq / self.peak_equity - 1.0

    def sector_exposure(self, prices: Dict[str, float]) -> Dict[str, float]:
        eq = self.equity(prices)
        out: Dict[str, float] = {}
        if eq <= 0:
            return out
        for p in self.positions.values():
            out[p.sector] = out.get(p.sector, 0.0) + p.market_value(
                prices.get(p.symbol, p.avg_price)) / eq
        return out

    # -- execution ----------------------------------------------------------
    def buy(self, symbol: str, sector: str, quantity: int, price: float,
            ts: str, stop: Optional[float] = None, reason: str = "") -> Optional[dict]:
        if quantity <= 0 or price <= 0:
            return None
        slip = price * self.costs.slippage_bps / 10_000.0
        fill_price = price + slip

        # Trim to what cash actually covers, charges included.
        while quantity > 0:
            turnover = quantity * fill_price
            total = turnover + self.costs.charges("BUY", quantity, fill_price)["total"]
            if total <= self.cash + 1e-9:
                break
            quantity -= 1
        if quantity <= 0:
            return None

        charges = self.costs.charges("BUY", quantity, fill_price)
        self.cash -= quantity * fill_price + charges["total"]

        existing = self.positions.get(symbol)
        if existing:
            new_q = existing.quantity + quantity
            existing.avg_price = (
                existing.avg_price * existing.quantity + fill_price * quantity
            ) / new_q
            existing.quantity = new_q
            if stop is not None:
                existing.stop = max(existing.stop or 0.0, stop)
        else:
            self.positions[symbol] = PaperPosition(
                symbol=symbol, sector=sector, quantity=quantity,
                avg_price=fill_price, opened=ts, stop=stop)

        record = {"ts": ts, "symbol": symbol, "side": "BUY", "quantity": quantity,
                  "price": fill_price, "charges": charges["total"], "reason": reason,
                  "charges_detail": charges}
        self.trade_log.append(record)
        return record

    def sell(self, symbol: str, price: float, ts: str, reason: str = "") -> Optional[dict]:
        position = self.positions.get(symbol)
        if position is None or price <= 0:
            return None
        slip = price * self.costs.slippage_bps / 10_000.0
        fill_price = price - slip
        quantity = position.quantity
        charges = self.costs.charges("SELL", quantity, fill_price)

        self.cash += quantity * fill_price - charges["total"]
        del self.positions[symbol]

        record = {"ts": ts, "symbol": symbol, "side": "SELL", "quantity": quantity,
                  "price": fill_price, "charges": charges["total"], "reason": reason,
                  "pnl": (fill_price - position.avg_price) * quantity - charges["total"],
                  "charges_detail": charges}
        self.trade_log.append(record)
        return record

    # -- reporting ----------------------------------------------------------
    def snapshot(self, prices: Dict[str, float], ts: str) -> List[dict]:
        return [
            {
                "symbol": p.symbol,
                "sector": p.sector,
                "quantity": p.quantity,
                "avg_price": p.avg_price,
                "ltp": prices.get(p.symbol, p.avg_price),
                "pnl": p.pnl(prices.get(p.symbol, p.avg_price)),
            }
            for p in self.positions.values()
        ]

    def realised_stats(self) -> dict:
        sells = [t for t in self.trade_log if t["side"] == "SELL"]
        if not sells:
            return {"n_trades": 0, "win_rate": 0.0, "realised_pnl": 0.0,
                    "profit_factor": 0.0, "total_charges": 0.0}
        pnls = [t["pnl"] for t in sells]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        gross_loss = -sum(losses)
        return {
            "n_trades": len(sells),
            "win_rate": len(wins) / len(sells),
            "realised_pnl": sum(pnls),
            "profit_factor": (sum(wins) / gross_loss) if gross_loss > 0 else float("inf"),
            "total_charges": sum(t.get("charges", 0.0) for t in self.trade_log),
        }
