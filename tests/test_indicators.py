"""Indicator correctness — the math the whole desk leans on."""
import numpy as np
import pandas as pd
import pytest

from tradingbot import indicators as ta


def make_df(n=300, seed=7):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0005, 0.02, n)
    close = 1000 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.008, n)))
    open_ = low + (high - low) * rng.random(n)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": rng.integers(1e5, 5e5, n)})


def test_ema_converges_to_price():
    df = make_df()
    flat = pd.Series([100.0] * 100)
    assert ta.ema(flat, 20).iloc[-1] == pytest.approx(100.0)


def test_rsi_bounds():
    df = make_df()
    r = ta.rsi(df["close"], 14)
    assert r.dropna().between(0, 100).all()


def test_atr_positive_and_smooth():
    df = make_df()
    a = ta.atr(df, 14)
    assert (a.dropna() > 0).all()
    assert len(a) == len(df)


def test_adx_range():
    df = make_df()
    a = ta.adx(df, 14)
    assert a.dropna().between(0, 100).all()


def test_cpr_ordering():
    c = ta.central_pivot_range(120, 100, 110)
    assert c["tc"] >= c["bc"]
    assert c["pivot"] >= c["bc"] - 1e-9


def test_pivots_symmetry():
    p = ta.classic_pivots(120, 100, 110)
    assert p["pivot"] == pytest.approx(110)
    assert p["r1"] == pytest.approx(120)
    assert p["s1"] == pytest.approx(100)


def test_hurst_trending_series_is_high():
    rng = np.random.default_rng(3)
    n = 400
    drift = rng.normal(0.003, 0.002, n)          # persistent positive drift
    r = pd.Series(drift)
    h = ta.hurst_exponent(r)
    assert h > 0.55


def test_ewma_vol_positive():
    df = make_df()
    r = np.log(df["close"] / df["close"].shift(1)).dropna()
    v = ta.ewma_vol(r)
    assert 0 < v < 5.0                            # annualised sanity


def test_supertrend_directions():
    df = make_df()
    d = ta.supertrend(df)
    assert set(d.unique()).issubset({-1, 1})


def test_max_drawdown():
    eq = pd.Series([100, 120, 90, 110])
    assert ta.max_drawdown(eq) == pytest.approx((90 - 120) / 120)
