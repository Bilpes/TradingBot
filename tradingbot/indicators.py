"""Vectorised, causal technical indicators.

Every function here uses only rolling or backward-looking operations, so an
agent built on them cannot look into the future. ``tests/test_indicators.py``
asserts this property directly by truncating the series and comparing the tail.

Conventions:
* Input is a ``pandas.Series`` of closes (or a DataFrame for volume-aware ones).
* Output is a ``Series``/``DataFrame`` aligned to the input index.
* Warmup regions are ``NaN``, never zero-filled -- a zero RSI and a
  not-yet-computed RSI must not look the same to an agent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "sma", "ema", "rsi", "macd", "atr", "true_range", "bollinger",
    "volume_zscore", "volume_spike", "efficiency_ratio", "rolling_std",
    "annualised_vol", "cross",
]


def sma(series: pd.Series, window: int) -> pd.Series:
    if window < 1:
        raise ValueError("window must be >= 1")
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    if span < 1:
        raise ValueError("span must be >= 1")
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder RSI in [0, 100]. NaN until ``window`` deltas are available."""
    if window < 2:
        raise ValueError("window must be >= 2")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder smoothing == EMA with alpha = 1/window.
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # A pure-uptrend window has avg_loss == 0 -> RSI is 100 by definition.
    out = out.where(avg_loss != 0.0, 100.0)
    # A flat window has both zero -> undefined, leave as NaN.
    out = out.where(~((avg_gain == 0.0) & (avg_loss == 0.0)))
    return out


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Standard MACD. Returns columns ``macd``, ``signal``, ``histogram``."""
    if fast >= slow:
        raise ValueError("fast span must be shorter than slow span")
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "histogram": line - sig})


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(
        alpha=1.0 / window, adjust=False, min_periods=window
    ).mean()


def bollinger(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    mid = sma(close, window)
    sd = close.rolling(window, min_periods=window).std()
    upper = mid + num_std * sd
    lower = mid - num_std * sd
    width = (upper - lower) / mid
    # %B: where the close sits inside the band, 0 = lower, 1 = upper.
    pct_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame({"mid": mid, "upper": upper, "lower": lower, "width": width, "pct_b": pct_b})


def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    """How many standard deviations today's volume is above its own average."""
    mu = volume.rolling(window, min_periods=window).mean()
    sd = volume.rolling(window, min_periods=window).std().replace(0.0, np.nan)
    return (volume - mu) / sd


def volume_spike(volume: pd.Series, window: int = 20, threshold: float = 2.0) -> pd.Series:
    """Boolean-ish 0/1 series: volume at least ``threshold`` x its rolling mean."""
    mu = volume.rolling(window, min_periods=window).mean()
    return (volume >= threshold * mu).astype("float64").where(mu.notna())


def efficiency_ratio(close: pd.Series, window: int = 20) -> pd.Series:
    """Kaufman efficiency ratio in [0, 1]: net move / total path length."""
    delta = close.diff()
    net = delta.rolling(window, min_periods=window).sum().abs()
    path = delta.abs().rolling(window, min_periods=window).sum().replace(0.0, np.nan)
    return (net / path).fillna(0.0)


def rolling_std(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window, min_periods=window).std()


def annualised_vol(close: pd.Series, window: int = 21, periods: int = 252) -> pd.Series:
    return close.pct_change().rolling(window, min_periods=window).std() * np.sqrt(periods)


def cross(fast: pd.Series, slow: pd.Series) -> pd.DataFrame:
    """Detect crossovers. ``golden`` = fast crosses above slow, ``death`` below."""
    above = fast > slow
    # fill_value= keeps the bool dtype. Without it, shift() produces object
    # dtype and `~True` evaluates to -2 -- truthy -- so every bar after a
    # crossover re-fires as a fresh crossover.
    prev_above = above.shift(1, fill_value=False)
    return pd.DataFrame(
        {
            "golden": (above & ~prev_above).astype("float64"),
            "death": (~above & prev_above).astype("float64"),
            "spread": (fast - slow) / slow,
        }
    )
