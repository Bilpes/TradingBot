import numpy as np
import pandas as pd
import pytest

from tradingbot.indicators import (
    atr, bollinger, cross, efficiency_ratio, ema, macd, rsi, sma,
    volume_spike, volume_zscore,
)


@pytest.fixture
def series():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2021-01-04", periods=200)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, 200))), index=idx)


@pytest.fixture
def ohlc(series):
    return pd.DataFrame({
        "high": series * 1.01, "low": series * 0.99, "close": series,
        "volume": np.linspace(1e6, 2e6, len(series)),
    }, index=series.index)


# -- causality: the property that makes all of this usable ------------------
@pytest.mark.parametrize("factory,kwargs", [
    (sma, {"window": 20}), (ema, {"span": 20}), (rsi, {"window": 14}),
    (efficiency_ratio, {"window": 20}),
])
def test_series_indicator_does_not_look_forward(series, factory, kwargs):
    cut = series.index[150]
    full = factory(series, **kwargs).loc[:cut]
    truncated = factory(series.loc[:cut], **kwargs)
    pd.testing.assert_series_equal(full, truncated)


def test_frame_indicator_does_not_look_forward(series):
    cut = series.index[150]
    pd.testing.assert_frame_equal(
        bollinger(series, window=20).loc[:cut], bollinger(series.loc[:cut], window=20))


def test_macd_is_causal(series):
    cut = series.index[150]
    full = macd(series)
    trunc = macd(series.loc[:cut])
    pd.testing.assert_frame_equal(full.loc[:cut], trunc)


def test_atr_is_causal(ohlc):
    cut = ohlc.index[150]
    pd.testing.assert_series_equal(
        atr(ohlc["high"], ohlc["low"], ohlc["close"]).loc[:cut],
        atr(ohlc["high"].loc[:cut], ohlc["low"].loc[:cut], ohlc["close"].loc[:cut]),
    )


# -- correctness against hand-computable cases ------------------------------
def test_rsi_is_100_for_a_monotonic_rise():
    up = pd.Series(np.arange(1.0, 60.0), index=pd.bdate_range("2021-01-04", periods=59))
    out = rsi(up, 14)
    assert out.iloc[-1] == pytest.approx(100.0)


def test_rsi_is_0_for_a_monotonic_fall():
    down = pd.Series(np.arange(60.0, 1.0, -1.0), index=pd.bdate_range("2021-01-04", periods=59))
    assert rsi(down, 14).iloc[-1] == pytest.approx(0.0, abs=1e-9)


def test_rsi_stays_in_range(series):
    out = rsi(series, 14).dropna()
    assert ((out >= 0) & (out <= 100)).all()


def test_rsi_warmup_is_nan_not_zero(series):
    """A not-yet-computed RSI must not be indistinguishable from RSI 0."""
    out = rsi(series, 14)
    assert out.iloc[:14].isna().all()


def test_macd_histogram_is_line_minus_signal(series):
    frame = macd(series)
    pd.testing.assert_series_equal(
        frame["histogram"].dropna(),
        (frame["macd"] - frame["signal"]).dropna(),
        check_names=False,
    )


def test_macd_rejects_inverted_spans(series):
    with pytest.raises(ValueError):
        macd(series, fast=26, slow=12)


def test_bollinger_pct_b_is_half_at_the_mean():
    # Constructed so the last close sits exactly on the 20-day mean. A constant
    # series would make the band degenerate (upper == lower) and pct_b undefined.
    rng = np.random.default_rng(0)
    values = 100 + rng.normal(0, 1.0, 60)
    # Fixed point: if the last value equals the mean of the other 19 in the
    # window, it also equals the mean of all 20 -- so pct_b must be exactly 0.5.
    values[-1] = values[-20:-1].mean()
    series = pd.Series(values, index=pd.bdate_range("2021-01-04", periods=60))
    out = bollinger(series, window=20)
    assert out["pct_b"].iloc[-1] == pytest.approx(0.5, abs=1e-9)


def test_volume_spike_fires_only_above_threshold():
    idx = pd.bdate_range("2021-01-04", periods=40)
    vol = pd.Series([1000.0] * 39 + [10_000.0], index=idx)
    out = volume_spike(vol, window=20, threshold=2.0)
    assert out.iloc[-1] == 1.0
    assert out.iloc[:-1].dropna().sum() == 0.0


def test_volume_zscore_sign_follows_the_spike():
    idx = pd.bdate_range("2021-01-04", periods=40)
    vol = pd.Series([1000.0] * 39 + [5000.0], index=idx)
    assert volume_zscore(vol, 20).iloc[-1] > 3.0


def test_efficiency_ratio_is_one_for_a_straight_line():
    idx = pd.bdate_range("2021-01-04", periods=40)
    line = pd.Series(np.linspace(100, 200, 40), index=idx)
    assert efficiency_ratio(line, 20).iloc[-1] == pytest.approx(1.0, abs=1e-9)


def test_cross_detects_golden_cross():
    idx = pd.bdate_range("2021-01-04", periods=5)
    fast = pd.Series([9.0, 9.0, 11.0, 12.0, 13.0], index=idx)
    slow = pd.Series([10.0] * 5, index=idx)
    out = cross(fast, slow)
    assert out["golden"].sum() == 1.0
    assert out["golden"].iloc[2] == 1.0


def test_sma_rejects_bad_window(series):
    with pytest.raises(ValueError):
        sma(series, 0)
