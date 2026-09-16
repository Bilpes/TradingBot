import pandas as pd
import pytest

from tradingbot.core.risk import RiskConfig, atr, realised_vol, size_position, true_range


def test_confidence_scales_size_monotonically():
    cfg = RiskConfig()
    sizes = [
        size_position(100_000, 100.0, 2.0, c, 0.25, cfg).shares
        for c in (0.3, 0.5, 0.7, 0.9)
    ]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_below_min_confidence_is_a_flat_no():
    cfg = RiskConfig(min_confidence=0.4)
    out = size_position(100_000, 100.0, 2.0, 0.39, 0.25, cfg)
    assert out.shares == 0
    assert out.binding == "confidence"


def test_higher_volatility_produces_a_smaller_position():
    cfg = RiskConfig()
    calm = size_position(100_000, 100.0, 2.0, 0.8, 0.15, cfg).shares
    wild = size_position(100_000, 100.0, 2.0, 0.8, 0.60, cfg).shares
    assert wild < calm


def test_max_position_weight_is_a_hard_ceiling():
    cfg = RiskConfig(max_position_weight=0.05)
    out = size_position(100_000, 10.0, 0.05, 1.0, 0.10, cfg)
    assert out.shares * 10.0 <= 0.05 * 100_000 + 1e-6


def test_a_wide_stop_makes_the_stop_risk_constraint_bind():
    cfg = RiskConfig(risk_per_trade=0.01)
    tight = size_position(100_000, 100.0, 1.0, 1.0, 0.20, cfg)
    wide = size_position(100_000, 100.0, 8.0, 1.0, 0.20, cfg)
    assert wide.binding == "stop_risk"
    assert wide.shares < tight.shares
    # Losing the stop must not cost more than risk_per_trade of equity.
    assert wide.shares * (100.0 - wide.stop) <= 0.01 * 100_000 + 1e-6


def test_min_notional_blocks_dust_positions():
    cfg = RiskConfig(min_notional=10_000.0)
    out = size_position(20_000, 500.0, 5.0, 0.3, 0.20, cfg)
    assert out.shares == 0
    assert out.binding == "notional"


def test_degenerate_inputs_do_not_produce_a_size():
    cfg = RiskConfig()
    assert size_position(0, 100.0, 1.0, 0.8, 0.2, cfg).shares == 0
    assert size_position(100_000, 0.0, 1.0, 0.8, 0.2, cfg).shares == 0
    # Missing volatility falls back to the floor instead of dividing by zero.
    out = size_position(100_000, 100.0, 1.0, 0.8, float("nan"), cfg)
    assert out.shares > 0


def test_true_range_accounts_for_gaps():
    df = pd.DataFrame(
        {
            "open": [10, 10, 20.0],
            "high": [11, 11, 21.0],
            "low": [9, 9, 19.0],
            "close": [10, 10, 20.0],
        }
    )
    tr = true_range(df)
    # Gaps up from 10 to 20: the range must span the gap, not just the bar.
    assert tr.iloc[2] == pytest.approx(11.0)


def test_atr_and_realised_vol_are_finite_and_positive(tiny_history):
    df = tiny_history["TEST"]
    assert (atr(df, 14).dropna() > 0).all()
    assert (realised_vol(df["close"], 21).dropna() > 0).all()


def test_risk_config_rejects_nonsense():
    with pytest.raises(ValueError):
        RiskConfig(risk_per_trade=0.0)
    with pytest.raises(ValueError):
        RiskConfig(max_position_weight=1.5, max_gross_exposure=1.0)
    with pytest.raises(ValueError):
        RiskConfig(min_confidence=1.5)
