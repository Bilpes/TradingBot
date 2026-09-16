"""Core data containers shared by providers, agents and the engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping

import pandas as pd

#: Column contract for every OHLCV frame in the system.
OHLCV = ("open", "high", "low", "close", "volume")

#: A history is a mapping of ticker -> OHLCV DataFrame indexed by session date.
History = Dict[str, pd.DataFrame]


@dataclass(frozen=True)
class SectorInfo:
    """Static classification of one ticker."""

    symbol: str
    sector: str
    name: str


@dataclass(frozen=True)
class Bar:
    """A single session for a single instrument."""

    symbol: str
    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


def validate_history(history: Mapping[str, pd.DataFrame]) -> History:
    """Check the OHLCV contract and return a copy with sorted, deduped indexes.

    Raises ``ValueError`` on anything that would silently corrupt a backtest:
    missing columns, NaN prices, non-positive prices, an unsorted index, or a
    frame where ``high`` is below ``low``.
    """
    if not history:
        raise ValueError("history is empty")

    cleaned: History = {}
    for symbol, frame in history.items():
        missing = [c for c in OHLCV if c not in frame.columns]
        if missing:
            raise ValueError(f"{symbol}: missing columns {missing}")

        df = frame[list(OHLCV)].copy()
        df.index = pd.to_datetime(df.index)
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        if df[["open", "high", "low", "close"]].isna().any().any():
            raise ValueError(f"{symbol}: NaN in price columns")
        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            raise ValueError(f"{symbol}: non-positive price")
        if (df["volume"] < 0).any():
            raise ValueError(f"{symbol}: negative volume")
        bad = df["high"] < df["low"]
        if bad.any():
            first = df.index[bad][0].date()
            raise ValueError(f"{symbol}: high < low on {first}")

        cleaned[symbol] = df
    return cleaned


def closes(history: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide close-price matrix (dates x symbols), forward-filled across gaps."""
    return pd.DataFrame({s: df["close"] for s, df in history.items()}).sort_index().ffill()


def volumes(history: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({s: df["volume"] for s, df in history.items()}).sort_index().ffill().fillna(0.0)


def row_asof(frame: pd.DataFrame, date: pd.Timestamp) -> pd.Series | None:
    """Last row at or before ``date``, or ``None`` if there is no such row.

    This is the single place where agents read their precomputed indicators, so
    the "no future data" rule lives here: it is a ``<=`` lookup, never ``==``.
    """
    if frame is None or frame.empty:
        return None
    prior = frame.index[frame.index <= date]
    if len(prior) == 0:
        return None
    return frame.loc[prior[-1]]
