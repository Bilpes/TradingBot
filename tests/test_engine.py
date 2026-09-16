import pandas as pd
import pytest

from tradingbot.agents.momentum import CrossSectionalMomentumAgent
from tradingbot.core.engine import Engine, EngineConfig
from tradingbot.core.portfolio import Side
from tradingbot.core.risk import RiskConfig


@pytest.fixture(scope="module")
def result(synthetic_history):
    return Engine().run(synthetic_history)


def test_equity_curve_covers_every_session(result, synthetic_history):
    n = len(next(iter(synthetic_history.values())))
    assert len(result.equity) == n
    assert result.equity.notna().all()
    assert (result.equity > 0).all()


def test_report_contains_the_core_statistics(result):
    for key in ("total_return", "cagr", "sharpe", "max_drawdown", "win_rate", "n_trades"):
        assert key in result.stats


def test_the_strategy_actually_trades(result):
    assert result.stats["n_trades"] > 0
    assert len(result.fills) > 0
    assert not result.decisions.empty


def test_no_trade_executes_on_the_session_that_produced_its_signal(result):
    """Signals are taken on the close of t and must fill no earlier than t+1."""
    dec = result.decisions
    entries = dec[(dec.stage == "decide") & (dec.action == "entry_signal")]
    fills = dec[(dec.stage == "fill") & (dec.action == "buy")]
    assert not entries.empty and not fills.empty

    for _, fill in fills.iterrows():
        prior = entries[
            (entries.symbol == fill.symbol) & (entries.date < fill.date)
        ]
        assert not prior.empty, f"buy of {fill.symbol} on {fill.date} had no earlier signal"
        same_day = entries[
            (entries.symbol == fill.symbol) & (entries.date == fill.date)
        ]
        assert same_day.empty, f"{fill.symbol} filled on the same session it was signalled"


def test_position_count_never_exceeds_the_cap(result, synthetic_history):
    held = {}
    for f in result.fills:
        held[f.symbol] = held.get(f.symbol, 0) + (f.shares if f.side is Side.BUY else -f.shares)
        held = {k: v for k, v in held.items() if v > 0}
        assert len(held) <= RiskConfig().max_positions


def test_exposure_stays_within_the_leverage_limit(result):
    # Sizing respects the cap at decision time; execution at the next open can
    # drift it slightly, so the assertion allows a small band.
    assert result.stats["max_exposure"] <= 1.10


def test_an_absurd_confidence_bar_stops_the_system_trading(synthetic_history):
    cfg = EngineConfig(risk=RiskConfig(min_confidence=0.995, entry_threshold=0.99))
    res = Engine(config=cfg).run(synthetic_history)
    buys = [f for f in res.fills if f.side is Side.BUY]
    assert buys == []


def test_runs_are_deterministic_for_the_same_input(synthetic_history):
    a = Engine().run(synthetic_history).equity
    b = Engine().run(synthetic_history).equity
    pd.testing.assert_series_equal(a, b)


def test_a_single_agent_behaves_differently_from_the_ensemble(synthetic_history):
    solo = Engine(agents=[CrossSectionalMomentumAgent()]).run(synthetic_history)
    full = Engine().run(synthetic_history)
    assert solo.stats["n_trades"] != full.stats["n_trades"] or (
        solo.stats["total_return"] != full.stats["total_return"]
    )


def test_rejections_are_logged_with_a_reason(result):
    rejected = result.decisions[result.decisions.action == "rejected"]
    assert rejected.empty or (rejected.reason.str.len() > 0).all()


def test_stops_are_attached_to_entries_and_trigger(result):
    stops = result.decisions[result.decisions.stage == "stop"]
    if not stops.empty:
        assert (stops.shares > 0).all()
        assert stops.reason.str.startswith("stop").all()


def test_signal_exits_respect_the_minimum_holding_period(synthetic_history):
    """Signal exits honour the holding period; stops are allowed to override it.

    Entry and exit share a hysteresis band plus a holding period; without them
    the book round-trips on noise and pays the spread for the privilege. Stops
    are deliberately exempt -- risk exits must never be delayed by a hold timer.
    """
    cfg = EngineConfig(risk=RiskConfig(min_holding_days=10))
    res = Engine(config=cfg).run(synthetic_history)
    assert res.trades, "expected trades to inspect"

    dates = res.equity.index
    positions = {d: i for i, d in enumerate(dates)}
    signal_exits = [t for t in res.trades if not str(t["reason"]).startswith("stop")]
    assert signal_exits, "no signal exits produced"

    held = [positions[t["exit_date"]] - positions[t["entry_date"]] for t in signal_exits]
    assert min(held) >= 10, f"shortest signal hold was {min(held)} sessions"


def test_hysteresis_reduces_turnover_vs_a_symmetric_threshold(synthetic_history):
    churny = Engine(
        config=EngineConfig(risk=RiskConfig(exit_confidence=0.25, min_holding_days=0))
    ).run(synthetic_history)
    patient = Engine(
        config=EngineConfig(risk=RiskConfig(exit_confidence=0.12, min_holding_days=5))
    ).run(synthetic_history)
    assert len(patient.fills) < len(churny.fills)
