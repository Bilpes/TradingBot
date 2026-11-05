"""Feature engineering for the ML prediction stack.

Design rules (leakage-safe):
  - features at row t use ONLY data up to and including t
  - labels use future returns strictly AFTER t (shift(-h))
  - the last h rows carry NaN labels and are dropped for training

Two label horizons:
  h=1  → next-session edge (intraday-ish probability)
  h=5  → 5-session forward return (swing probability)
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .. import indicators as ta

FEATURE_NAMES: List[str] = [
    "ret1", "ret3", "ret5", "ret10", "ret20",
    "rsi14", "macd_hist_norm", "ema9_21_spread", "dist_ema50", "dist_ema200",
    "atr_pct", "bb_z", "vol_z", "dist_20d_high", "dist_52w_high",
    "gap_pct", "bench_ret1", "bench_ret5", "bench_dist_ema50",
    "rel_strength_20", "hurst", "vix", "vix_chg5", "sentiment",
]


def build_features(df: pd.DataFrame, bench_df: pd.DataFrame,
                   vix: float = 14.0, vix_chg5: float = 0.0,
                   sentiment: float = 0.0) -> pd.DataFrame:
    """df: daily OHLCV of the symbol (chronological). bench_df: NIFTY daily."""
    f = pd.DataFrame(index=df.index)
    close = df["close"]
    logret = np.log(close / close.shift(1))

    for k in (1, 3, 5, 10, 20):
        f[f"ret{k}"] = close.pct_change(k)

    f["rsi14"] = ta.rsi(close, 14)
    _, _, hist = ta.macd(close)
    f["macd_hist_norm"] = hist / close
    e9, e21 = ta.ema(close, 9), ta.ema(close, 21)
    f["ema9_21_spread"] = (e9 - e21) / close
    f["dist_ema50"] = (close / ta.ema(close, 50) - 1)
    f["dist_ema200"] = (close / ta.ema(close, 200) - 1)

    a = ta.atr(df, 14)
    f["atr_pct"] = a / close

    mid_up, mid, mid_dn = ta.bollinger(close, 20, 2.0)
    band = (mid_up - mid_dn).replace(0, np.nan)
    f["bb_z"] = (close - mid) / band

    f["vol_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std().replace(0, np.nan)
    f["dist_20d_high"] = close / close.rolling(20).max() - 1
    f["dist_52w_high"] = close / close.rolling(252, min_periods=60).max() - 1
    f["gap_pct"] = (df["open"] / close.shift(1) - 1).fillna(0)

    if bench_df is not None and len(bench_df) > 60:
        b = bench_df.set_index("ts") if "ts" in bench_df.columns else bench_df
        s = df.set_index("ts") if "ts" in df.columns else df
        common = s.index.intersection(b.index)
        if len(common) > 60:
            b = b.loc[common]
            s = s.loc[common]
            bc = b["close"]
            f["bench_ret1"] = bc.pct_change(1).values
            f["bench_ret5"] = bc.pct_change(5).values
            f["bench_dist_ema50"] = (bc / bc.ewm(span=50, adjust=False).mean() - 1).values
            rs = (s["close"].pct_change(20) - bc.pct_change(20)).values
            f["rel_strength_20"] = rs

    f["hurst"] = pd.Series(logret).rolling(120, min_periods=60).apply(
        lambda x: ta.hurst_exponent(pd.Series(x)), raw=False).values

    f["vix"] = vix
    f["vix_chg5"] = vix_chg5
    f["sentiment"] = sentiment
    return f


def make_labels(df: pd.DataFrame, features: pd.DataFrame, horizon: int = 5,
                threshold: float = 0.005) -> Optional[pd.Series]:
    """1 if forward `horizon`-session return > threshold (covers costs+slippage)."""
    fwd = (df["close"].shift(-horizon) / df["close"] - 1)
    y = (fwd > threshold).astype(float)
    y[fwd.isna()] = np.nan
    if y.notna().sum() < 100:
        return None
    return y


def aligned_dataset(df: pd.DataFrame, bench_df: pd.DataFrame, horizon: int,
                    vix: float = 14.0, vix_chg5: float = 0.0,
                    sentiment: float = 0.0, threshold: float = 0.005):
    f = build_features(df, bench_df, vix, vix_chg5, sentiment)
    y = make_labels(df, f, horizon, threshold)
    X = f[FEATURE_NAMES]
    mask = X.notna().all(axis=1) & y.notna()
    return X[mask].astype(float), y[mask].astype(int), X
