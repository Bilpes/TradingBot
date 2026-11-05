"""Pure-pandas technical indicator library — no TA-Lib dependency.

Every function is vectorised, NaN-safe at the edges, and unit-tested
(tests/test_indicators.py). These feed the Technical, TimeSeries and ML agents.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    dn = -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1.0 / period, adjust=False).mean().replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False).mean().fillna(0)


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP — call on a single trading day's intraday frame."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].replace(0, np.nan)
    return (tp * vol).cumsum() / vol.cumsum()


def bollinger(close: pd.Series, period: int = 20, k: float = 2.0):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std(ddof=0)
    return mid + k * sd, mid, mid - k * sd


def hurst_exponent(log_returns: pd.Series, max_lag: int = 20) -> float:
    """Hurst exponent of the PRICE PATH implied by log-returns.

    The lagged-increment method must run on the integrated series: for
    fractional Brownian motion, Var[P(t+lag) − P(t)] scales as lag^(2H).
      H > 0.55 → trending / persistent price path
      H < 0.45 → mean-reverting
    (On raw returns the differences cancel the mean and the estimator
    degenerates to 0 — the classic implementation bug this function avoids.)
    """
    r = pd.Series(log_returns).dropna().values.astype(float)
    if len(r) < max_lag + 5:
        return 0.5
    p = np.cumsum(r)                       # integrate → price path
    lags = range(2, max_lag + 1)
    tau = [math.sqrt(np.mean(np.square(p[lag:] - p[:-lag]))) for lag in lags]
    xs = np.log(np.array([lag for lag in lags], dtype=float))
    ys = np.log(np.maximum(np.array(tau, dtype=float), 1e-12))
    poly = np.polyfit(xs, ys, 1)
    return float(np.clip(poly[0], 0.0, 1.0))


def central_pivot_range(prev_h: float, prev_l: float, prev_c: float):
    """CPR: pivot, top-center, bottom-center. Narrow CPR ⇒ trend-day odds rise."""
    pivot = (prev_h + prev_l + prev_c) / 3.0
    bc = (prev_h + prev_l) / 2.0
    tc = pivot - (bc - pivot)
    return {"pivot": pivot, "tc": max(tc, bc), "bc": min(tc, bc),
            "width_pct": (max(tc, bc) - min(tc, bc)) / pivot * 100.0 if pivot else 0.0}


def classic_pivots(prev_h: float, prev_l: float, prev_c: float) -> dict:
    p = (prev_h + prev_l + prev_c) / 3.0
    r1 = 2 * p - prev_l
    s1 = 2 * p - prev_h
    r2 = p + (prev_h - prev_l)
    s2 = p - (prev_h - prev_l)
    return {"pivot": p, "r1": r1, "r2": r2, "s1": s1, "s2": s2,
            "r3": prev_h + 2 * (p - prev_l), "s3": prev_l - 2 * (prev_h - p)}


def realized_vol(close: pd.Series, window: int = 20, annualize: bool = True) -> float:
    r = np.log(close / close.shift(1)).dropna().tail(window)
    if len(r) < 5:
        return 0.0
    v = r.std(ddof=0) * math.sqrt(252) if annualize else r.std(ddof=0)
    return float(v)


def ewma_vol(log_returns: pd.Series, lam: float = 0.94, annualize: bool = True) -> float:
    """RiskMetrics-style EWMA volatility forecast (1-day ahead)."""
    r = pd.Series(log_returns).dropna()
    if len(r) < 30:
        return float(r.std(ddof=0) if len(r) else 0.0)
    w = lam ** np.arange(len(r) - 1, -1, -1)
    w /= w.sum()
    var = float((w * (r.values ** 2)).sum())
    return math.sqrt(var * 252) if annualize else math.sqrt(var)


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0) -> pd.Series:
    """Supertrend direction series: +1 up, -1 down (index-aligned)."""
    a = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(1, index=df.index)
    up_prev, dn_prev = upper.iloc[0], lower.iloc[0]
    for i in range(len(df)):
        if i == 0:
            st.iloc[i] = lower.iloc[0]
            continue
        up = upper.iloc[i] if upper.iloc[i] < up_prev or df["close"].iloc[i - 1] > up_prev else up_prev
        dn = lower.iloc[i] if lower.iloc[i] > dn_prev or df["close"].iloc[i - 1] < dn_prev else dn_prev
        direction.iloc[i] = 1 if df["close"].iloc[i] > up_prev else (-1 if df["close"].iloc[i] < dn_prev else direction.iloc[i - 1])
        st.iloc[i] = dn if direction.iloc[i] == 1 else up
        up_prev, dn_prev = up, dn
    return direction


def relative_strength(close: pd.Series, bench_close: pd.Series, window: int = 20) -> float:
    """RS = stock return − benchmark return over `window` sessions (%)."""
    r_stock = (close.iloc[-1] / close.iloc[-window - 1] - 1) * 100 if len(close) > window else 0.0
    r_bench = (bench_close.iloc[-1] / bench_close.iloc[-window - 1] - 1) * 100 if len(bench_close) > window else 0.0
    return r_stock - r_bench
