"""Domain models: agent reports, signals, positions, trades.

Everything crosses threads, so these are plain dataclasses (JSON-safe via
`asdict` / custom serialisers used by the API layer).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


def _clean(o: Any) -> Any:
    """Convert numpy scalars/bools → plain Python so json.dumps never chokes
    (numpy 2.x renders np.bool_ as class name 'bool')."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (datetime,)):
        return o.isoformat(timespec="seconds")
    return o


# ---------------------------------------------------------------- agents ----
@dataclass
class AgentReport:
    """Uniform output of every agent. score ∈ [-100, +100] (bullish ↔ bearish)."""
    agent: str
    status: str = "OK"                     # OK | SKIP | ERROR
    score: float = 0.0
    confidence: float = 0.5                # 0..1
    note: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return _clean({
            "agent": self.agent, "status": self.status, "score": round(self.score, 1),
            "confidence": round(self.confidence, 2), "note": self.note,
            "data": self.data, "elapsed_ms": self.elapsed_ms,
        })


# ---------------------------------------------------------------- signals ---
@dataclass
class Signal:
    """A fully-constructed trade proposal (pre-risk-gate)."""
    symbol: str
    side: str                              # BUY | SELL (SELL = intraday short, MIS only)
    style: str                             # MIS (intraday) | CNC (swing/delivery)
    entry: float
    stop: float
    target1: float
    target2: float
    qty: int
    score: float                           # fused composite score
    ml_prob: float                         # ML P(up)
    setup: str                             # ORB | VWAP_PULLBACK | BREAKOUT_20D | EMA_PULLBACK ...
    rationale: List[str] = field(default_factory=list)
    regime: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    status: str = "PROPOSED"               # PROPOSED -> TAKEN / REJECTED / EXPIRED

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def risk_amount(self) -> float:
        return self.risk_per_share * self.qty

    def to_dict(self) -> Dict[str, Any]:
        return _clean({
            "symbol": self.symbol, "side": self.side, "style": self.style,
            "entry": round(self.entry, 2), "stop": round(self.stop, 2),
            "target1": round(self.target1, 2), "target2": round(self.target2, 2),
            "qty": self.qty, "score": round(self.score, 1), "ml_prob": round(self.ml_prob, 3),
            "setup": self.setup, "rationale": self.rationale, "regime": self.regime,
            "created_at": self.created_at, "status": self.status,
            "risk_amount": round(self.risk_amount, 0),
        })


# ---------------------------------------------------------------- broker ----
@dataclass
class Fill:
    symbol: str
    side: str
    qty: int
    price: float
    charges: float
    order_id: str
    ts: str


# ---------------------------------------------------------------- position --
@dataclass
class Position:
    """Open (paper) position with full lifecycle state."""
    position_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    symbol: str = ""
    side: str = "BUY"                       # BUY long | SELL short
    style: str = "MIS"                      # MIS intraday | CNC swing
    qty: int = 0
    entry_price: float = 0.0
    entry_ts: str = ""
    stop: float = 0.0
    target1: float = 0.0
    target2: float = 0.0
    ltp: float = 0.0
    setup: str = ""
    regime: str = ""
    rationale: List[str] = field(default_factory=list)
    score: float = 0.0
    ml_prob: float = 0.0
    partial_booked: bool = False            # T1 booked → trailing leg
    entry_charges: float = 0.0
    realized_charges: float = 0.0
    realized_pnl: float = 0.0               # booked so far (T1 partial)
    status: str = "OPEN"                    # OPEN | CLOSED
    high_water: float = 0.0                 # best price seen (trailing)
    days_held: int = 0

    # ---- P&L in ₹ ------------------------------------------------------
    @property
    def unrealized_pnl(self) -> float:
        direction = 1 if self.side == "BUY" else -1
        return (self.ltp - self.entry_price) * direction * self.qty

    @property
    def open_pnl_after_charges(self) -> float:
        return self.unrealized_pnl - self.entry_charges

    @property
    def notional(self) -> float:
        return self.ltp * self.qty if self.ltp else self.entry_price * self.qty

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.open_pnl_after_charges

    def to_dict(self) -> Dict[str, Any]:
        return _clean({
            "position_id": self.position_id, "symbol": self.symbol, "side": self.side,
            "style": self.style, "qty": self.qty,
            "entry_price": round(self.entry_price, 2), "ltp": round(self.ltp, 2),
            "stop": round(self.stop, 2), "target1": round(self.target1, 2),
            "target2": round(self.target2, 2), "setup": self.setup, "regime": self.regime,
            "entry_ts": self.entry_ts, "days_held": self.days_held,
            "unrealized_pnl": round(self.open_pnl_after_charges, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "total_pnl": round(self.total_pnl, 2),
            "partial_booked": self.partial_booked, "status": self.status,
            "rationale": self.rationale[:3],
        })


# ---------------------------------------------------------------- trade -----
@dataclass
class Trade:
    """A closed trade (or a partial close booked against a Position)."""
    trade_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    position_id: str = ""
    symbol: str = ""
    side: str = "BUY"
    style: str = "MIS"
    qty: int = 0
    entry_price: float = 0.0
    exit_price: float = 0.0
    entry_ts: str = ""
    exit_ts: str = ""
    gross_pnl: float = 0.0
    charges: float = 0.0
    net_pnl: float = 0.0                    # what the user actually keeps, ₹
    setup: str = ""
    exit_reason: str = ""                   # TARGET1 | TARGET2 | STOP | TRAIL | SQUAREOFF | TIME_EXIT | MANUAL
    held_days: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return _clean({
            "trade_id": self.trade_id, "symbol": self.symbol, "side": self.side,
            "style": self.style, "qty": self.qty,
            "entry_price": round(self.entry_price, 2), "exit_price": round(self.exit_price, 2),
            "entry_ts": self.entry_ts, "exit_ts": self.exit_ts,
            "net_pnl": round(self.net_pnl, 2), "charges": round(self.charges, 2),
            "exit_reason": self.exit_reason, "setup": self.setup, "held_days": self.held_days,
        })
