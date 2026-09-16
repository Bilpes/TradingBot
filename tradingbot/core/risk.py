"""Position sizing and hard risk limits.

Sizing is the intersection of two independent constraints, and the *tighter*
one always wins:

1. **Volatility targeting** -- scale the weight so the position contributes
   roughly ``target_vol`` of annualised risk, then multiply by confidence.
   Confidence enters *here and only here*: a high-conviction idea gets a bigger
   slice, but never a slice that breaks a limit.
2. **Fixed-fractional stop risk** -- cap the position so that being stopped out
   at the ATR stop costs no more than ``risk_per_trade`` of equity.

Deliberate omission: Kelly sizing. Kelly needs a trustworthy estimate of edge
and odds, and a backtest cannot honestly supply one -- in-sample estimates are
the very thing that makes Kelly blow up out of sample. Claiming to "buy with
confidence" via Kelly would be the most confident-sounding and least reliable
thing this module could do, so it uses a hard weight cap instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade: float = 0.01        # equity fraction lost if the stop hits
    max_position_weight: float = 0.12   # hard cap on any single name
    max_sector_weight: float = 0.30     # hard cap on any one sector
    max_gross_exposure: float = 1.00    # no leverage by default
    max_positions: int = 12
    # Target *portfolio* volatility contribution of one position, annualised.
    # weight = target_vol / asset_vol, so a 40%-vol name gets 5% of equity and a
    # 20%-vol name gets 10%. This is deliberately not the asset's own vol: with
    # target_vol set to a whole-asset level (e.g. 0.20) the weight cap binds for
    # every normal equity and vol targeting silently does nothing at all.
    target_vol: float = 0.02
    vol_floor: float = 0.10             # below this, assume it (avoids huge sizes)
    vol_lookback: int = 21
    atr_window: int = 14
    stop_atr_mult: float = 2.5
    min_confidence: float = 0.25        # consensus gate for new entries
    entry_threshold: float = 0.20       # consensus score gate for new entries
    exit_threshold: float = 0.00        # exit when conviction goes non-positive
    # Exit gates sit *below* the entry gates on purpose. Using one threshold for
    # both creates a system that buys at confidence >= 0.25 and dumps the
    # instant confidence ticks to 0.249 -- a one-day round trip that pays
    # commissions forever. Hysteresis plus a minimum holding period is what
    # turns a signal into a position. Stops ignore both: risk exits are never
    # delayed by a holding period.
    exit_confidence: float = 0.12
    min_holding_days: int = 5
    min_notional: float = 500.0

    def __post_init__(self) -> None:
        for f in ("risk_per_trade", "max_position_weight", "max_sector_weight",
                  "max_gross_exposure", "target_vol"):
            v = getattr(self, f)
            if v <= 0:
                raise ValueError(f"{f} must be positive, got {v}")
        if self.max_position_weight > self.max_gross_exposure:
            raise ValueError("max_position_weight cannot exceed max_gross_exposure")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if not 0.0 <= self.exit_confidence <= 1.0:
            raise ValueError("exit_confidence must be in [0, 1]")
        if self.min_holding_days < 0:
            raise ValueError("min_holding_days cannot be negative")


def true_range(df: pd.DataFrame) -> pd.Series:
    """Wilder true range: max(high-low, |high-prev_close|, |low-prev_close|)."""
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average true range, Wilder-smoothed."""
    return true_range(df).ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def realised_vol(closes: pd.Series, window: int = 21) -> pd.Series:
    return closes.pct_change().rolling(window, min_periods=window).std() * np.sqrt(252)


@dataclass(frozen=True)
class SizingResult:
    shares: int
    stop: Optional[float]
    target_weight: float
    binding: str
    reason: str


def size_position(
    equity: float,
    price: float,
    atr_value: float,
    confidence: float,
    vol: float,
    cfg: RiskConfig,
) -> SizingResult:
    """Shares to buy, or 0 with an explanation of what blocked the trade."""
    if equity <= 0 or price <= 0:
        return SizingResult(0, None, 0.0, "invalid", "non-positive equity or price")
    if confidence < cfg.min_confidence:
        return SizingResult(0, None, 0.0, "confidence",
                            f"confidence {confidence:.2f} < {cfg.min_confidence:.2f}")

    vol_used = max(vol if np.isfinite(vol) and vol > 0 else cfg.vol_floor, cfg.vol_floor)
    vol_weight = cfg.target_vol / vol_used
    target_weight = min(vol_weight, cfg.max_position_weight) * confidence

    shares_vol = equity * target_weight / price

    stop = None
    shares_risk = float("inf")
    if np.isfinite(atr_value) and atr_value > 0:
        stop = price - cfg.stop_atr_mult * atr_value
        per_share_risk = price - stop
        if per_share_risk > 0:
            shares_risk = equity * cfg.risk_per_trade / per_share_risk

    shares = int(floor(min(shares_vol, shares_risk)))
    binding = "stop_risk" if shares_risk <= shares_vol else "vol_target"

    if shares <= 0:
        return SizingResult(0, stop, target_weight, "size", "rounded to zero shares")
    if shares * price < cfg.min_notional:
        return SizingResult(0, stop, target_weight, "notional",
                            f"{shares} shares < min notional {cfg.min_notional:.0f}")

    return SizingResult(shares, stop, target_weight, binding,
                        f"vol weight {vol_weight:.2f} -> {target_weight:.2%} of equity")
