import pytest

from tradingbot.paper.broker import NSECostModel, PaperBroker


def test_starts_with_the_paper_capital():
    broker = PaperBroker(capital=100_000)
    assert broker.cash == 100_000
    assert broker.equity({}) == 100_000


def test_buy_and_sell_round_trip_loses_the_charges():
    costs = NSECostModel(product="CNC", brokerage_per_order=0.0, slippage_bps=0.0)
    broker = PaperBroker(capital=100_000, costs=costs)
    broker.buy("HDFCBANK", "Banking", 100, 1000.0, "t1")
    broker.sell("HDFCBANK", 1000.0, "t2")
    assert broker.positions == {}
    assert broker.cash < 100_000
    assert broker.realised_stats()["realised_pnl"] < 0  # a flat trade loses STT + charges


def test_delivery_costs_include_stt_on_both_sides():
    costs = NSECostModel(product="CNC")
    buy = costs.charges("BUY", 100, 1000.0)
    sell = costs.charges("SELL", 100, 1000.0)
    assert buy["stt"] == pytest.approx(100_000 * 0.001)
    assert sell["stt"] == pytest.approx(100_000 * 0.001)
    assert buy["stamp"] > 0 and sell["stamp"] == 0
    assert sell["dp"] > 0 and buy["dp"] == 0


def test_intraday_costs_differ_from_delivery():
    delivery = NSECostModel(product="CNC").charges("SELL", 100, 1000.0)
    intraday = NSECostModel(product="INTRADAY", brokerage_per_order=20.0).charges("SELL", 100, 1000.0)
    assert intraday["stt"] < delivery["stt"]       # 0.025% sell-only vs 0.1%
    assert intraday["dp"] == 0 and delivery["dp"] > 0
    assert intraday["brokerage"] == 20.0


def test_gst_is_charged_on_brokerage_plus_exchange_only():
    costs = NSECostModel(product="CNC", brokerage_per_order=100.0)
    charges = costs.charges("BUY", 100, 1000.0)
    assert charges["gst"] == pytest.approx((100.0 + charges["exchange"]) * 0.18)


def test_slippage_moves_the_fill_against_us():
    flat = PaperBroker(capital=100_000, costs=NSECostModel(slippage_bps=0.0))
    slipped = PaperBroker(capital=100_000, costs=NSECostModel(slippage_bps=50.0))
    flat.buy("X", "IT", 10, 1000.0, "t")
    slipped.buy("X", "IT", 10, 1000.0, "t")
    assert slipped.trade_log[0]["price"] > flat.trade_log[0]["price"]
    flat.sell("X", 1000.0, "t")
    slipped.sell("X", 1000.0, "t")
    assert slipped.trade_log[1]["price"] < flat.trade_log[1]["price"]


def test_quantity_is_trimmed_to_available_cash_rather_than_overdrawing():
    broker = PaperBroker(capital=100_000, costs=NSECostModel(slippage_bps=0.0))
    fill = broker.buy("X", "IT", 1_000, 1000.0, "t")   # 1,000,000 requested on 100k
    assert fill is not None
    assert fill["quantity"] == 99          # 99 * 1000 + charges fits, 100 does not
    assert broker.cash >= 0


def test_a_buy_that_cannot_afford_even_one_share_is_refused():
    broker = PaperBroker(capital=500, costs=NSECostModel())
    assert broker.buy("X", "IT", 10, 1000.0, "t") is None


def test_adding_to_a_position_blends_the_average_cost():
    broker = PaperBroker(capital=1_000_000, costs=NSECostModel(slippage_bps=0.0))
    broker.buy("X", "IT", 100, 1000.0, "t1")
    broker.buy("X", "IT", 100, 1200.0, "t2")
    assert broker.positions["X"].quantity == 200
    assert broker.positions["X"].avg_price == pytest.approx(1100.0)


def test_stop_is_kept_at_the_tighter_level_when_adding():
    broker = PaperBroker(capital=1_000_000, costs=NSECostModel(slippage_bps=0.0))
    broker.buy("X", "IT", 100, 1000.0, "t1", stop=950.0)
    broker.buy("X", "IT", 100, 1000.0, "t2", stop=900.0)
    assert broker.positions["X"].stop == 950.0


def test_selling_a_position_we_do_not_hold_is_a_no_op():
    broker = PaperBroker(capital=100_000)
    assert broker.sell("GHOST", 100.0, "t") is None
    assert broker.cash == 100_000


def test_exposure_and_sector_accounting():
    broker = PaperBroker(capital=100_000, costs=NSECostModel(slippage_bps=0.0))
    broker.buy("HDFCBANK", "Banking", 10, 1000.0, "t")
    broker.buy("TCS", "IT", 10, 1000.0, "t")
    prices = {"HDFCBANK": 1000.0, "TCS": 1000.0}
    # Not exactly 0.20: transaction charges reduce cash, so equity dips a hair
    # below the starting capital and exposure reads marginally high.
    assert broker.exposure(prices) == pytest.approx(0.20, abs=1e-3)
    sectors = broker.sector_exposure(prices)
    assert sectors["Banking"] == pytest.approx(0.10, abs=1e-3)
    assert sectors["IT"] == pytest.approx(0.10, abs=1e-3)


def test_drawdown_tracks_the_peak():
    broker = PaperBroker(capital=100_000, costs=NSECostModel(slippage_bps=0.0))
    broker.buy("X", "IT", 10, 1000.0, "t")
    assert broker.drawdown({"X": 1000.0}) == pytest.approx(0.0, abs=1e-3)
    assert broker.drawdown({"X": 500.0}) == pytest.approx(-0.05, abs=1e-3)
    assert broker.peak_equity == pytest.approx(100_000)


def test_realised_stats_are_empty_but_well_formed_before_any_sell():
    stats = PaperBroker(capital=100_000).realised_stats()
    assert stats["n_trades"] == 0 and stats["profit_factor"] == 0.0


def test_win_rate_and_profit_factor_after_real_trades():
    broker = PaperBroker(capital=1_000_000, costs=NSECostModel(slippage_bps=0.0,
                                                              brokerage_per_order=0.0,
                                                              stt_delivery=0.0,
                                                              exchange_txn=0.0,
                                                              sebi_turnover=0.0,
                                                              stamp_buy=0.0,
                                                              dp_charge_per_sell=0.0))
    broker.buy("W", "IT", 10, 1000.0, "t1")
    broker.sell("W", 1100.0, "t2")            # +1000
    broker.buy("L", "IT", 10, 1000.0, "t3")
    broker.sell("L", 950.0, "t4")             # -500
    stats = broker.realised_stats()
    assert stats["n_trades"] == 2
    assert stats["win_rate"] == pytest.approx(0.5)
    assert stats["profit_factor"] == pytest.approx(2.0)
    assert stats["realised_pnl"] == pytest.approx(500.0)
