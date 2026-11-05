"""End-to-end smoke: store roundtrip, simulator determinism, one full engine
cycle in offline sim mode (agents → fusion → paper fills → positions)."""
import json
import time

import numpy as np
import pandas as pd
import pytest

from tradingbot.data.simulator import SymbolSimulator
from tradingbot.data.store import Store
from tradingbot.models import Position, Trade


def test_simulator_deterministic():
    a = SymbolSimulator("RELIANCE", 400)
    b = SymbolSimulator("RELIANCE", 400)
    da, db = a.daily(), b.daily()
    assert len(da) == len(db)
    assert da[-1]["close"] == db[-1]["close"]


def test_simulator_has_enough_history():
    s = SymbolSimulator("TCS", 400)
    bars = s.daily()
    assert len(bars) == 400
    closes = [b["close"] for b in bars]
    assert min(closes) > 0
    assert max(closes) > min(closes) * 0.5


def test_store_roundtrip(tmp_path):
    st = Store(tmp_path / "t.db")
    st.add_symbol("TCS")
    st.add_symbol("INFY")
    assert st.watchlist() == ["INFY", "TCS"]
    st.remove_symbol("TCS")
    assert st.watchlist() == ["INFY"]

    p = Position(symbol="TCS", side="BUY", style="CNC", qty=10,
                 entry_price=3500.0, entry_ts="2025-01-01T09:20:00",
                 stop=3450.0, target1=3600.0, target2=3650.0, ltp=3520.0)
    st.save_position(p)
    loaded = st.open_positions()
    assert len(loaded) == 1
    assert loaded[0].symbol == "TCS"
    assert loaded[0].qty == 10

    t = Trade(symbol="TCS", qty=10, entry_price=3500, exit_price=3600,
              exit_ts="2025-01-03T15:20:00", gross_pnl=1000, charges=20, net_pnl=980)
    st.save_trade(t)
    assert st.trades()[0]["net_pnl"] == 980
    assert st.realized_pnl_total() == 980

    st.save_equity_point(100980.0, 980.0, 0.0)
    assert st.equity_curve()[-1]["equity"] == 100980.0


def test_full_engine_cycle_sim(tmp_path, monkeypatch):
    """The money test: one complete cycle offline — agents run, decisions are
    made, state is JSON-safe for the dashboard."""
    from tradingbot.config import data_dir
    import tradingbot.config as cfg
    monkeypatch.setattr(cfg, "data_dir", lambda: tmp_path)

    # fresh store in tmp
    from tradingbot.data.store import Store as S
    store = S(tmp_path / "engine.db")
    for s in ["RELIANCE", "TCS", "INFY"]:
        store.add_symbol(s)

    from tradingbot.data.market_data import MarketData
    md = MarketData(["RELIANCE", "TCS", "INFY"], mode="sim")

    from tradingbot.engine.orchestrator import TradingEngine
    eng = TradingEngine(store, md, ["RELIANCE", "TCS", "INFY"])

    stats = eng.run_cycle(do_entries=True)
    assert stats["scanned"] == 3

    state = eng.state()
    # JSON-serialisable (dashboard contract)
    blob = json.dumps(state)
    assert "portfolio" in blob and "positions" in blob

    pf = state["portfolio"]
    assert pf["capital"] == 100000
    assert pf["equity"] > 0
    # any executed trade must respect position sizing caps
    for sig in state["signals"]:
        if sig.get("qty"):
            assert sig["qty"] <= 250
