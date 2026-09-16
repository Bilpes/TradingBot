import pandas as pd
import pytest

from tradingbot.agents.registry import profile_for
from tradingbot.core.consensus import ConsensusConfig
from tradingbot.core.risk import RiskConfig
from tradingbot.data.providers import SyntheticProvider
from tradingbot.markets.nse import NSE_SECTORS, NSE_UNIVERSE, nse_sector_of
from tradingbot.notify.telegram import FakeTransport, TelegramNotifier
from tradingbot.paper.broker import NSECostModel, PaperBroker
from tradingbot.store.db import Store
from tradingbot.workers.coordinator import Coordinator
from tradingbot.workers.sector_worker import SectorWorker, stale_sectors


@pytest.fixture
def nse_history():
    return SyntheticProvider(sessions=400, seed=13, sector_map=nse_sector_of,
                        universe=list(NSE_UNIVERSE)).load([], "2021-01-04")


@pytest.fixture
def session(nse_history):
    return pd.Timestamp(max(df.index[-1] for df in nse_history.values())).strftime("%Y-%m-%d")


# -- workers ----------------------------------------------------------------
def test_worker_only_sees_its_own_sector(nse_history):
    symbols = [s for s in nse_history if nse_sector_of(s) == "Banking"]
    worker = SectorWorker("Banking", symbols, profile_for("Banking"), "/tmp/unused.sqlite3")
    assert set(worker.slice_history(nse_history)) == set(symbols)


def test_worker_writes_signals_for_every_symbol_it_owns(tmp_path, nse_history, session):
    db = str(tmp_path / "t.sqlite3")
    symbols = [s for s in nse_history if nse_sector_of(s) == "IT"]
    worker = SectorWorker("IT", symbols, profile_for("IT"), db)
    written = worker.run_cycle(nse_history, session)
    assert written > 0

    store = Store(db)
    rows = store.latest_signals(session)
    store.close()
    assert {r["symbol"] for r in rows} <= set(symbols)
    assert {r["sector"] for r in rows} == {"IT"}


def test_worker_survives_a_missing_symbol(tmp_path, nse_history, session):
    db = str(tmp_path / "t.sqlite3")
    worker = SectorWorker("Metal", ["TATASTEEL", "NOT_IN_HISTORY"],
                          profile_for("Metal"), db)
    assert worker.run_cycle(nse_history, session) > 0


def test_worker_run_is_reproducible(tmp_path, nse_history, session):
    db = str(tmp_path / "t.sqlite3")
    symbols = [s for s in nse_history if nse_sector_of(s) == "Pharma"]
    counts = []
    for i in range(2):
        worker = SectorWorker("Pharma", symbols, profile_for("Pharma"), db)
        counts.append(worker.run_cycle(nse_history, session))
    assert counts[0] == counts[1]


def test_stale_sectors_detects_a_worker_that_never_reported(tmp_path, nse_history, session):
    db = str(tmp_path / "t.sqlite3")
    symbols = [s for s in nse_history if nse_sector_of(s) == "Banking"]
    SectorWorker("Banking", symbols, profile_for("Banking"), db).run_cycle(nse_history, session)
    quiet = stale_sectors(db, session, ["Banking", "Metal", "IT"])
    assert quiet == ["Metal", "IT"]


# -- coordinator ------------------------------------------------------------
def _populated_store(tmp_path, nse_history, session, risk=None, consensus=None):
    from tradingbot.live import PaperRunConfig, run_in_process

    db = str(tmp_path / "paper.sqlite3")
    store = Store(db)
    broker = PaperBroker(capital=100_000, costs=NSECostModel(product="CNC"))
    notifier = TelegramNotifier(bot_token="t", chat_id="1",
                                transport=FakeTransport(), min_seconds_between=0.0)
    result = run_in_process(
        history=nse_history, store=store, broker=broker,
        profiles={s: profile_for(s) for s in NSE_SECTORS}, notifier=notifier,
        config=PaperRunConfig(risk=risk or RiskConfig(),
                              consensus=consensus or ConsensusConfig(),
                              sectors=NSE_SECTORS),
        sector_of=nse_sector_of,
    )
    return store, broker, notifier, result


def test_end_to_end_cycle_produces_decisions_with_reasoning(tmp_path, nse_history, session):
    store, broker, notifier, result = _populated_store(tmp_path, nse_history, session)
    try:
        assert result["signals"] > 0
        decisions = store.recent_decisions(limit=500)
        assert decisions, "coordinator recorded no decisions"

        row = decisions[0]
        assert row["votes_json"] and row["votes_json"] != "{}"
        assert row["sector"]
        assert row["direction"] in {"long", "short", "flat"}
        # Every non-actionable decision must say why.
        for item in decisions:
            if not item["actionable"]:
                assert item["rejection"], f"decision {item['id']} rejected without a reason"
    finally:
        store.close()


def test_cycle_marks_equity_and_snapshots_positions(tmp_path, nse_history, session):
    store, broker, _, _ = _populated_store(tmp_path, nse_history, session)
    try:
        curve = store.equity_curve()
        assert curve, "no equity point written"
        assert curve[-1]["total"] == pytest.approx(broker.equity(
            {s: float(df['close'].iloc[-1]) for s, df in nse_history.items()}))
    finally:
        store.close()


def test_every_fill_is_audited_by_an_alert_row(tmp_path, nse_history, session):
    """Telegram being unconfigured must not make trades invisible.

    The alert row is written from the delivery callback either way, so the
    dashboard can always show what happened even when the message never sent.
    """
    from tradingbot.live import PaperRunConfig, run_in_process

    db = str(tmp_path / "p.sqlite3")
    store = Store(db)
    broker = PaperBroker(capital=100_000)
    notifier = TelegramNotifier(bot_token="", chat_id="", transport=FakeTransport())
    run_in_process(nse_history, store, broker, {s: profile_for(s) for s in NSE_SECTORS},
                   notifier, PaperRunConfig(sectors=NSE_SECTORS), sector_of=nse_sector_of)
    alerts = store.alerts(limit=200)
    fills = store.fills()
    store.close()

    assert fills, "expected the cycle to trade on this history"
    assert len(alerts) >= len(fills)
    assert all(a["delivered"] == 0 for a in alerts)
    assert any("not configured" in (a["error"] or "") for a in alerts)


def test_a_strict_quorum_stops_the_system_trading(tmp_path, nse_history, session):
    strict = ConsensusConfig(min_voters=99, min_agreeing=99)
    store, broker, _, result = _populated_store(
        tmp_path, nse_history, session, consensus=strict)
    try:
        assert result["report"].actionable == 0
        assert broker.positions == {}
        assert broker.cash == 100_000
    finally:
        store.close()


def test_position_cap_holds_under_the_coordinator(tmp_path, nse_history, session):
    store, broker, _, _ = _populated_store(
        tmp_path, nse_history, session, risk=RiskConfig(max_positions=3))
    try:
        assert len(broker.positions) <= 3
    finally:
        store.close()


def test_coordinator_stands_down_when_no_worker_reported(tmp_path, session):
    db = str(tmp_path / "empty.sqlite3")
    store = Store(db)
    broker = PaperBroker(capital=100_000)
    coordinator = Coordinator(store=store, broker=broker,
                              notifier=TelegramNotifier(bot_token="", chat_id=""))
    try:
        report = coordinator.run_cycle(session, prices={})
        assert report.decisions == 0
        assert broker.cash == 100_000
        # The failure must be visible, not silent.
        assert any(a["kind"] == "risk" for a in store.alerts(limit=10))
    finally:
        store.close()


def test_coordinator_ignores_a_malformed_signal_row(tmp_path, nse_history, session):
    db = str(tmp_path / "p.sqlite3")
    store = Store(db)
    symbols = [s for s in nse_history if nse_sector_of(s) == "Banking"]
    SectorWorker("Banking", symbols, profile_for("Banking"), db).run_cycle(nse_history, session)

    with store.tx() as conn:
        conn.execute(
            "INSERT INTO signals(ts, session_date, sector, symbol, agent, role, score,"
            " confidence, reason, parts_json) VALUES('now', ?, 'Banking', 'BAD', 'rsi',"
            " 'directional', 99, 0.5, 'out of range', '{}')", (session,))

    broker = PaperBroker(capital=100_000)
    coordinator = Coordinator(store=store, broker=broker,
                              notifier=TelegramNotifier(bot_token="", chat_id=""),
                              sector_of=nse_sector_of)
    try:
        prices = {s: float(df["close"].iloc[-1]) for s, df in nse_history.items()}
        report = coordinator.run_cycle(session, prices)
        assert report.decisions > 0  # the good rows still processed
        assert all(d["symbol"] != "BAD" for d in
                   [dict(r) for r in store.recent_decisions(limit=500)])
    finally:
        store.close()


# -- the "low load on the main app" claim -----------------------------------
def test_sector_workers_partition_the_work(tmp_path, nse_history, session):
    """Each worker handles a strict subset; none touches another sector's names."""
    db = str(tmp_path / "p.sqlite3")
    seen: dict[str, set[str]] = {}
    for sector in NSE_SECTORS:
        symbols = [s for s in nse_history if nse_sector_of(s) == sector]
        if not symbols:
            continue
        worker = SectorWorker(sector, symbols, profile_for(sector), db)
        worker.run_cycle(nse_history, session)
        seen[sector] = set(worker.slice_history(nse_history))

    all_symbols = [s for v in seen.values() for s in v]
    assert len(all_symbols) == len(set(all_symbols)), "a symbol was handled by two sectors"
    assert set(all_symbols) == {s for s in nse_history if nse_sector_of(s) in seen}


def test_per_sector_work_is_much_smaller_than_the_whole_universe(nse_history, tmp_path):
    """Quantifies the isolation claim instead of just asserting it."""
    import time

    full = list(nse_history)
    sector_symbols = [s for s in full if nse_sector_of(s) == "Banking"]

    def timeit(symbols):
        db = str(tmp_path / f"{len(symbols)}.sqlite3")
        worker = SectorWorker("Banking", symbols, profile_for("Banking"), db)
        session = pd.Timestamp(max(nse_history[s].index[-1] for s in symbols)).strftime("%Y-%m-%d")
        start = time.perf_counter()
        worker.run_cycle(nse_history, session)
        return time.perf_counter() - start

    one_sector = timeit(sector_symbols)
    everything = timeit(full)
    assert one_sector < everything
    # Banking is 5 of ~45 names; require a real, not nominal, saving.
    assert one_sector < everything / 2


def test_the_long_only_book_never_buys_a_negative_score(tmp_path, nse_history, session):
    """Regression: 'actionable' is direction-agnostic, and a long book is not.

    Without this gate the coordinator bought whatever cleared the quorum,
    including strongly bearish names, and posted a 0% win rate -- reliably long
    exactly what its own agents expected to fall.
    """
    from tradingbot.live import PaperRunConfig, run_in_process

    db = str(tmp_path / "longonly.sqlite3")
    store = Store(db)
    broker = PaperBroker(capital=100_000)
    run_in_process(
        nse_history, store, broker, {s: profile_for(s) for s in NSE_SECTORS},
        TelegramNotifier(bot_token="", chat_id=""),
        PaperRunConfig(sectors=NSE_SECTORS), sector_of=nse_sector_of)

    buys = [f for f in store.fills() if f["side"] == "BUY"]
    assert buys, "expected at least one entry on this history"

    scores = {d["symbol"]: d["score"] for d in
              [dict(r) for r in store.recent_decisions(limit=2000)]}
    for fill in buys:
        score = scores.get(fill["symbol"])
        assert score is not None, f"buy of {fill['symbol']} has no recorded decision"
        assert score >= 0.0, f"bought {fill['symbol']} on a negative score {score:+.2f}"
    store.close()
