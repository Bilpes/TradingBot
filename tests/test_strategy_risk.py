"""Charges model, sizing math, risk gates, strategy fusion decisions."""
import numpy as np
import pandas as pd
import pytest

from tradingbot.broker.paper import estimate_charges
from tradingbot.config import AGENT_WEIGHTS
from tradingbot.models import AgentReport
from tradingbot.agents.risk import compute_sizing


def test_cnc_buy_charges_match_expected():
    # ₹1000 price, 10 shares → turnover 10,000
    c = estimate_charges("TEST", "BUY", 10, 1000.0, "CNC")
    stt = 0.001 * 10000            # 10
    stamp = 0.00015 * 10000        # 1.5
    txn = 0.0000297 * 10000        # ~0.297
    sebi = 0.000001 * 10000        # 0.01
    gst = 0.18 * txn               # brokerage 0 → gst on txn+sebi only
    assert c == pytest.approx(stt + stamp + txn + sebi + gst, rel=1e-3)


def test_mis_charges_include_brokerage_cap():
    big = estimate_charges("TEST", "BUY", 1000, 1000.0, "MIS")   # ₹10,00,000 turnover
    small = estimate_charges("TEST", "BUY", 10, 100.0, "MIS")    # ₹1,000 turnover
    # big → brokerage capped at ₹20, plus txn/sebi/gst/stamp → well above ₹20
    assert big > 20
    # small → brokerage 0.03% of 1,000 = ₹0.30 (uncapped), total stays tiny
    assert 0 < small < big


def test_cnc_sell_charges_include_stt():
    sell = estimate_charges("TEST", "SELL", 10, 1000.0, "CNC")
    buy = estimate_charges("TEST", "BUY", 10, 1000.0, "CNC")
    # delivery STT applies on BOTH legs; stamp only on buy → buy leg costlier
    assert buy > sell > 0


def test_sizing_respects_risk_amount_then_notional_cap():
    qty, reason = compute_sizing(entry=1000.0, stop=990.0, equity=100_000)
    # raw risk sizing: ₹1,000 / ₹10 = 100 shares, BUT 25% notional cap
    # (₹25,000 / ₹1,000) binds first → 25 shares. Conservative by design.
    assert qty == 25
    assert "risk" in reason


def test_sizing_caps_by_notional():
    qty, _ = compute_sizing(entry=5000.0, stop=4950.0, equity=100_000)
    # raw risk sizing: 1000/50 = 20 shares = ₹1,00,000 notional > 25% cap (₹25,000) → 5 shares
    assert qty == 5


def test_sizing_zero_when_wide_stop():
    qty, _ = compute_sizing(entry=20000.0, stop=10000.0, equity=50_000)
    assert qty == 0


def test_fused_score_weighting():
    from tradingbot.engine.strategy import StrategyFusion
    reports = [
        AgentReport(agent="Technical", score=60, confidence=1.0, status="OK"),
        AgentReport(agent="MLPrediction", score=-40, confidence=1.0, status="OK"),
        AgentReport(agent="TimeSeries", score=0, confidence=1.0, status="OK"),
    ]
    s = StrategyFusion.fused_score(reports)
    w = AGENT_WEIGHTS
    expected = (60 * w["Technical"] - 40 * w["MLPrediction"]) / (w["Technical"] + w["MLPrediction"] + w["TimeSeries"])
    assert s == pytest.approx(expected, abs=0.5)


def test_error_reports_dont_poison_score():
    from tradingbot.engine.strategy import StrategyFusion
    reports = [
        AgentReport(agent="Technical", score=60, confidence=1.0, status="OK"),
        AgentReport(agent="MLPrediction", score=99, confidence=0.9, status="ERROR"),
    ]
    s = StrategyFusion.fused_score(reports)
    # errored agent contributes 0 (confidence zeroed)
    expected = 60 * AGENT_WEIGHTS["Technical"] / AGENT_WEIGHTS["Technical"]
    assert s == pytest.approx(expected, abs=1e-6)
