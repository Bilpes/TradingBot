import pandas as pd
import pytest

from tradingbot.core.portfolio import Order, Portfolio, Side

D = pd.Timestamp("2022-01-03")
D2 = pd.Timestamp("2022-01-04")


def test_buy_then_sell_returns_cash_net_of_costs():
    p = Portfolio(cash=100_000, commission_bps=10.0, slippage_bps=0.0)
    p.execute(Order("AAPL", Side.BUY, 100, "in"), 100.0, D)
    assert p.cash == pytest.approx(100_000 - 100 * 100.0 * (1 + 10 / 10_000))
    p.execute(Order("AAPL", Side.SELL, 100, "out"), 100.0, D2)
    assert p.positions == {}
    assert p.cash < 100_000  # round-trip costs are real
    # 10 bps on 10,000 of notional, charged on each side: 2 * $10.
    assert p.cash == pytest.approx(100_000 - 2 * 100 * 100.0 * 10 / 10_000)


def test_slippage_moves_the_fill_against_us():
    flat = Portfolio(cash=100_000, commission_bps=0.0, slippage_bps=0.0)
    slipped = Portfolio(cash=100_000, commission_bps=0.0, slippage_bps=20.0)
    flat.execute(Order("X", Side.BUY, 10, "in"), 100.0, D)
    slipped.execute(Order("X", Side.BUY, 10, "in"), 100.0, D)
    assert slipped.fills[0].price > flat.fills[0].price
    assert slipped.cash < flat.cash


def test_cannot_overdraft_cash():
    p = Portfolio(cash=1_000, commission_bps=0.0, slippage_bps=0.0)
    fill = p.execute(Order("X", Side.BUY, 100, "too big"), 100.0, D)
    assert fill is not None
    assert fill.shares == 10  # trimmed to what we can afford
    assert p.cash >= 0


def test_order_beyond_cash_is_refused_entirely_when_nothing_is_affordable():
    p = Portfolio(cash=5, commission_bps=10.0, slippage_bps=0.0)
    assert p.execute(Order("X", Side.BUY, 100, "nope"), 100.0, D) is None


def test_sell_without_a_position_is_a_no_op():
    p = Portfolio(cash=1_000)
    assert p.execute(Order("X", Side.SELL, 10, "ghost"), 100.0, D) is None
    assert p.cash == 1_000


def test_average_cost_updates_across_adds():
    p = Portfolio(cash=100_000, commission_bps=0.0, slippage_bps=0.0)
    p.execute(Order("X", Side.BUY, 100, "a"), 100.0, D)
    p.execute(Order("X", Side.BUY, 100, "b"), 200.0, D2)
    assert p.avg_cost["X"] == pytest.approx(150.0)


def test_round_trip_pnl_is_net_of_commissions():
    p = Portfolio(cash=100_000, commission_bps=10.0, slippage_bps=0.0)
    p.execute(Order("X", Side.BUY, 100, "in"), 100.0, D)
    p.execute(Order("X", Side.SELL, 100, "out"), 100.0, D2)
    trades = p.round_trips()
    assert len(trades) == 1
    assert trades[0]["pnl"] == pytest.approx(-2 * 100 * 100.0 * 10 / 10_000)
    assert trades[0]["pnl"] < 0  # a flat round trip loses the costs


def test_exposure_and_sector_accounting():
    p = Portfolio(cash=100_000, commission_bps=0.0, slippage_bps=0.0,
                  sector_of=lambda s: "Tech" if s == "AAPL" else "Energy")
    p.execute(Order("AAPL", Side.BUY, 100, "a"), 100.0, D)
    p.execute(Order("XOM", Side.BUY, 100, "b"), 100.0, D)
    prices = {"AAPL": 100.0, "XOM": 100.0}
    assert p.gross_exposure(prices) == pytest.approx(0.20)
    assert p.sector_weights(prices) == {"Tech": pytest.approx(0.10), "Energy": pytest.approx(0.10)}
    assert p.equity(prices) == pytest.approx(100_000)


def test_mark_builds_an_equity_curve():
    p = Portfolio(cash=100_000, commission_bps=0.0, slippage_bps=0.0)
    p.execute(Order("X", Side.BUY, 100, "a"), 100.0, D)
    p.mark(D, {"X": 100.0})
    p.mark(D2, {"X": 110.0})
    assert p.equity_curve.loc[D2] == pytest.approx(100_000 + 100 * 10.0)
