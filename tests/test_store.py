import pytest

from tradingbot.store.db import Store


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "t.sqlite3"))
    yield s
    s.close()


def test_schema_is_created(store):
    tables = {r["name"] for r in store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"signals", "decisions", "orders", "fills", "positions", "equity",
            "alerts", "predictions", "run_meta"} <= tables


def test_wal_mode_is_on(store):
    assert store._conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_signals_round_trip(store):
    n = store.write_signals([{
        "session_date": "2026-09-16", "sector": "Banking", "symbol": "HDFCBANK",
        "agent": "rsi", "score": 0.4, "confidence": 0.6, "reason": "r", "parts": {"rsi": 62.0},
    }])
    assert n == 1
    rows = store.latest_signals("2026-09-16")
    assert rows[0]["symbol"] == "HDFCBANK"
    assert rows[0]["parts_json"] == '{"rsi": 62.0}'


def test_write_signals_handles_empty_input(store):
    assert store.write_signals([]) == 0


def test_meta_round_trips_structured_values(store):
    store.set_meta("k", {"a": [1, 2, 3]})
    assert store.get_meta("k") == {"a": [1, 2, 3]}
    assert store.get_meta("absent", "fallback") == "fallback"


def test_transaction_rolls_back_on_error(store):
    with pytest.raises(ValueError):
        with store.tx() as conn:
            conn.execute("INSERT INTO run_meta(key, value) VALUES('x', '1')")
            raise ValueError("boom")
    assert store.get_meta("x") is None


def test_calibration_table_compares_claimed_to_realised(store):
    for conf, win in [(0.9, 1), (0.9, 1), (0.9, 0), (0.3, 0), (0.3, 0)]:
        store._conn.execute(
            "INSERT INTO predictions(ts, symbol, direction, confidence, horizon_days,"
            " entry_price, settled, exit_price, outcome_return, win)"
            " VALUES('t','X','long',?,5,100,1,110,0.1,?)", (conf, win))
    store._conn.commit()

    table = store.calibration_table(bins=5)
    assert table
    high = next(r for r in table if r["bucket"] == "80%-100%")
    assert high["n"] == 3
    assert high["realised_win_rate"] == pytest.approx(2 / 3)
    low = next(r for r in table if r["bucket"] == "20%-40%")
    assert low["realised_win_rate"] == pytest.approx(0.0)


def test_brier_score_matches_the_definition(store):
    for conf, win in [(0.8, 1), (0.2, 0)]:
        store._conn.execute(
            "INSERT INTO predictions(ts, symbol, direction, confidence, horizon_days,"
            " entry_price, settled, exit_price, outcome_return, win)"
            " VALUES('t','X','long',?,5,100,1,110,0.1,?)", (conf, win))
    store._conn.commit()
    # Each prediction missed by 0.2, so mean squared error is 0.2^2 = 0.04.
    assert store.brier_score() == pytest.approx(0.04)


def test_brier_score_is_none_with_no_settled_predictions(store):
    assert store.brier_score() is None
    assert store.calibration_table() == []


def _decision(store, session_date):
    """Predictions are graded against the session calendar, so make one."""
    store.write_decision({
        "session_date": session_date, "symbol": "X", "sector": "IT", "score": 0.1,
        "confidence": 0.5, "agreement": 0.9, "dispersion": 0.1, "agreeing": 3,
        "dissenting": 0, "neutral": 0, "vetoed": False, "veto_reason": "",
        "actionable": True, "rejection": "", "direction": "long", "votes": {},
    })


def _prediction(store, session_date, **overrides):
    payload = {"session_date": session_date, "symbol": "X", "direction": "long",
               "confidence": 0.7, "horizon_days": 5, "entry_price": 100.0}
    payload.update(overrides)
    store.write_prediction(payload)


def test_settle_predictions_marks_against_current_price(store):
    for i, d in enumerate(["2026-09-01", "2026-09-08"]):
        _decision(store, d)
    _prediction(store, "2026-09-01", horizon_days=1)
    assert store.settle_predictions({"X": 110.0}) == 1
    row = store._conn.execute("SELECT * FROM predictions").fetchone()
    assert row["settled"] == 1
    assert row["outcome_return"] == pytest.approx(0.10)
    assert row["win"] == 1


def test_settle_predictions_inverts_for_shorts(store):
    for d in ["2026-09-01", "2026-09-08"]:
        _decision(store, d)
    _prediction(store, "2026-09-01", direction="short", horizon_days=1)
    store.settle_predictions({"X": 110.0})
    row = store._conn.execute("SELECT * FROM predictions").fetchone()
    assert row["outcome_return"] == pytest.approx(-0.10)
    assert row["win"] == 0


def test_settle_predictions_waits_for_a_price(store):
    for d in ["2026-09-01", "2026-09-08"]:
        _decision(store, d)
    _prediction(store, "2026-09-01")
    assert store.settle_predictions({}) == 0
    assert store._conn.execute("SELECT settled FROM predictions").fetchone()["settled"] == 0


def test_a_prediction_is_not_graded_before_its_horizon_elapses(store):
    """Regression: horizon_days used to be written and then ignored.

    Settling on the next available session graded a five-session confidence
    claim on a one-session outcome, so the calibration table -- the only thing
    that shows whether confidence means anything -- was measuring noise.
    """
    sessions = ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-08", "2026-09-09"]
    for d in sessions:
        _decision(store, d)
    _prediction(store, "2026-09-01", horizon_days=3)

    # Only one session has elapsed: too early, must stay open.
    store._conn.execute("DELETE FROM decisions WHERE session_date > '2026-09-02'")
    assert store.settle_predictions({"X": 110.0}) == 0
    assert store._conn.execute("SELECT settled FROM predictions").fetchone()["settled"] == 0

    # Once three sessions have passed, it settles.
    for d in sessions[2:]:
        _decision(store, d)
    assert store.settle_predictions({"X": 110.0}) == 1
    assert store._conn.execute("SELECT settled FROM predictions").fetchone()["settled"] == 1


def test_old_stores_gain_the_session_date_column(tmp_path):
    """The migration must be additive, not destructive."""
    path = tmp_path / "old.sqlite3"
    import sqlite3

    conn = sqlite3.connect(path)
    conn.execute("""CREATE TABLE predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, symbol TEXT NOT NULL,
        direction TEXT NOT NULL, confidence REAL NOT NULL, horizon_days INTEGER NOT NULL,
        entry_price REAL NOT NULL, settled INTEGER NOT NULL DEFAULT 0,
        exit_price REAL, outcome_return REAL, win INTEGER)""")
    conn.execute("INSERT INTO predictions(ts, symbol, direction, confidence, horizon_days,"
                 " entry_price) VALUES('t','X','long',0.7,5,100)")
    conn.commit()
    conn.close()

    store = Store(path)
    try:
        cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(predictions)")}
        assert "session_date" in cols
        assert store._conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 1
    finally:
        store.close()


def test_equity_and_position_snapshots(store):
    store.write_equity("t1", 50_000, 50_000, 100_000, 0.0, 0.5)
    store.snapshot_positions("t1", [{"symbol": "X", "sector": "IT", "quantity": 10,
                                     "avg_price": 100.0, "ltp": 110.0, "pnl": 100.0}])
    assert store.equity_curve()[0]["total"] == 100_000
    assert store.latest_positions()[0]["symbol"] == "X"


def test_alerts_record_delivery_failure(store):
    store.write_alert("telegram", "entry", "text", "X", delivered=False, error="rate limited")
    row = store.alerts()[0]
    assert row["delivered"] == 0 and row["error"] == "rate limited"


def test_session_dates_are_ordered_and_unique(store):
    for date in ("2026-09-15", "2026-09-14", "2026-09-15"):
        store.write_decision({
            "session_date": date, "symbol": "X", "sector": "IT", "score": 0.1,
            "confidence": 0.5, "agreement": 0.9, "dispersion": 0.1, "agreeing": 3,
            "dissenting": 0, "neutral": 0, "vetoed": False, "veto_reason": "",
            "actionable": True, "rejection": "", "direction": "long", "votes": {},
        })
    assert store.session_dates() == ["2026-09-14", "2026-09-15"]
