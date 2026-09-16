"""SQLite store -- the system of record *and* the inter-process bus.

Why SQLite and not Redis/RabbitMQ: it is durable, needs no daemon, is trivially
inspectable from the dashboard, and its WAL mode supports many readers with one
writer -- which is exactly this system's shape. Each process opens its own
connection; nothing is shared in memory between a sector worker and the
coordinator, which is the point (see ``workers/``).

Every decision row carries the full reasoning: per-agent votes, agreement,
dispersion, and the rule that rejected it if it did not trade. "Track every
decision and its reasoning" is a schema requirement here, not a feature.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT NOT NULL,
    session_date TEXT,
    sector       TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    agent        TEXT NOT NULL,
    role         TEXT NOT NULL DEFAULT 'directional',
    score        REAL NOT NULL,
    confidence   REAL NOT NULL,
    reason       TEXT,
    parts_json   TEXT
);
CREATE INDEX IF NOT EXISTS idx_signals_lookup ON signals(session_date, symbol, agent);

CREATE TABLE IF NOT EXISTS decisions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    session_date  TEXT,
    symbol        TEXT NOT NULL,
    sector        TEXT NOT NULL,
    score         REAL NOT NULL,
    confidence    REAL NOT NULL,
    agreement     REAL NOT NULL,
    dispersion    REAL NOT NULL,
    agreeing      INTEGER NOT NULL,
    dissenting    INTEGER NOT NULL,
    neutral       INTEGER NOT NULL,
    vetoed        INTEGER NOT NULL,
    veto_reason   TEXT,
    actionable    INTEGER NOT NULL,
    rejection     TEXT,
    direction     TEXT NOT NULL,
    votes_json    TEXT NOT NULL,
    action        TEXT,
    quantity      INTEGER,
    price         REAL,
    reason        TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_date ON decisions(session_date);
CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON decisions(symbol);

CREATE TABLE IF NOT EXISTS orders (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    symbol     TEXT NOT NULL,
    side       TEXT NOT NULL,
    quantity   INTEGER NOT NULL,
    price      REAL,
    status     TEXT NOT NULL,
    broker_ref TEXT,
    reason     TEXT
);

CREATE TABLE IF NOT EXISTS fills (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    symbol    TEXT NOT NULL,
    side      TEXT NOT NULL,
    quantity  INTEGER NOT NULL,
    price     REAL NOT NULL,
    charges   REAL NOT NULL DEFAULT 0,
    reason    TEXT
);

CREATE TABLE IF NOT EXISTS positions (
    ts        TEXT NOT NULL,
    symbol    TEXT NOT NULL,
    sector    TEXT NOT NULL,
    quantity  INTEGER NOT NULL,
    avg_price REAL NOT NULL,
    ltp       REAL NOT NULL,
    pnl       REAL NOT NULL,
    PRIMARY KEY (ts, symbol)
);

CREATE TABLE IF NOT EXISTS equity (
    ts         TEXT PRIMARY KEY,
    cash       REAL NOT NULL,
    invested   REAL NOT NULL,
    total      REAL NOT NULL,
    drawdown   REAL NOT NULL,
    exposure   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         TEXT NOT NULL,
    channel    TEXT NOT NULL,
    kind       TEXT NOT NULL,
    symbol     TEXT,
    text       TEXT NOT NULL,
    delivered  INTEGER NOT NULL DEFAULT 0,
    error      TEXT
);

-- Calibration ledger: what confidence did we claim, and what actually happened?
CREATE TABLE IF NOT EXISTS predictions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT NOT NULL,
    session_date   TEXT,
    symbol         TEXT NOT NULL,
    direction      TEXT NOT NULL,
    confidence     REAL NOT NULL,
    horizon_days   INTEGER NOT NULL,
    entry_price    REAL NOT NULL,
    settled        INTEGER NOT NULL DEFAULT 0,
    exit_price     REAL,
    outcome_return REAL,
    win            INTEGER
);

CREATE TABLE IF NOT EXISTS run_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    """Thin, explicit DAO. One connection per process."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Additive migrations for stores created by an older schema."""
        columns = {r["name"] for r in self._conn.execute("PRAGMA table_info(predictions)")}
        if "session_date" not in columns:
            self._conn.execute("ALTER TABLE predictions ADD COLUMN session_date TEXT")

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # -- meta ---------------------------------------------------------------
    def set_meta(self, key: str, value: Any) -> None:
        with self.tx() as c:
            c.execute("INSERT OR REPLACE INTO run_meta(key, value) VALUES(?, ?)",
                      (key, json.dumps(value)))

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute("SELECT value FROM run_meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    # -- signals ------------------------------------------------------------
    def write_signals(self, rows: Sequence[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        with self.tx() as c:
            c.executemany(
                """INSERT INTO signals(ts, session_date, sector, symbol, agent, role,
                                       score, confidence, reason, parts_json)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                [
                    (utcnow(), r.get("session_date"), r["sector"], r["symbol"], r["agent"],
                     r.get("role", "directional"), r["score"], r["confidence"],
                     r.get("reason", ""), json.dumps(r.get("parts", {})))
                    for r in rows
                ],
            )
        return len(rows)

    def latest_signals(self, session_date: str) -> List[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM signals WHERE session_date=? ORDER BY id", (session_date,)
        ).fetchall()

    # -- decisions ----------------------------------------------------------
    def write_decision(self, d: Dict[str, Any]) -> int:
        with self.tx() as c:
            cur = c.execute(
                """INSERT INTO decisions(ts, session_date, symbol, sector, score, confidence,
                       agreement, dispersion, agreeing, dissenting, neutral, vetoed, veto_reason,
                       actionable, rejection, direction, votes_json, action, quantity, price, reason)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (utcnow(), d.get("session_date"), d["symbol"], d["sector"], d["score"],
                 d["confidence"], d["agreement"], d["dispersion"], d["agreeing"],
                 d["dissenting"], d["neutral"], int(d["vetoed"]), d.get("veto_reason", ""),
                 int(d["actionable"]), d.get("rejection", ""), d["direction"],
                 json.dumps(d["votes"]), d.get("action"), d.get("quantity"),
                 d.get("price"), d.get("reason", "")),
            )
            return int(cur.lastrowid)

    def recent_decisions(self, limit: int = 200, session_date: Optional[str] = None) -> List[sqlite3.Row]:
        if session_date:
            return self._conn.execute(
                "SELECT * FROM decisions WHERE session_date=? ORDER BY id DESC LIMIT ?",
                (session_date, limit)).fetchall()
        return self._conn.execute(
            "SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

    # -- trading ------------------------------------------------------------
    def write_order(self, o: Dict[str, Any]) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO orders(ts, symbol, side, quantity, price, status, broker_ref, reason)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (utcnow(), o["symbol"], o["side"], o["quantity"], o.get("price"),
                 o.get("status", "pending"), o.get("broker_ref"), o.get("reason", "")),
            )

    def write_fill(self, f: Dict[str, Any]) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO fills(ts, symbol, side, quantity, price, charges, reason)
                   VALUES(?,?,?,?,?,?,?)""",
                (utcnow(), f["symbol"], f["side"], f["quantity"], f["price"],
                 f.get("charges", 0.0), f.get("reason", "")),
            )

    def fills(self) -> List[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM fills ORDER BY id").fetchall()

    def snapshot_positions(self, ts: str, rows: Iterable[Dict[str, Any]]) -> None:
        with self.tx() as c:
            c.executemany(
                """INSERT OR REPLACE INTO positions(ts, symbol, sector, quantity, avg_price, ltp, pnl)
                   VALUES(?,?,?,?,?,?,?)""",
                [(ts, r["symbol"], r["sector"], r["quantity"], r["avg_price"],
                  r["ltp"], r["pnl"]) for r in rows],
            )

    def latest_positions(self) -> List[sqlite3.Row]:
        row = self._conn.execute("SELECT MAX(ts) AS ts FROM positions").fetchone()
        if not row or row["ts"] is None:
            return []
        return self._conn.execute(
            "SELECT * FROM positions WHERE ts=? ORDER BY pnl DESC", (row["ts"],)).fetchall()

    def write_equity(self, ts: str, cash: float, invested: float, total: float,
                     drawdown: float, exposure: float) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT OR REPLACE INTO equity(ts, cash, invested, total, drawdown, exposure)
                   VALUES(?,?,?,?,?,?)""", (ts, cash, invested, total, drawdown, exposure))

    def equity_curve(self) -> List[sqlite3.Row]:
        return self._conn.execute("SELECT * FROM equity ORDER BY ts").fetchall()

    # -- alerts -------------------------------------------------------------
    def write_alert(self, channel: str, kind: str, text: str, symbol: str | None = None,
                    delivered: bool = False, error: str | None = None) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO alerts(ts, channel, kind, symbol, text, delivered, error)
                   VALUES(?,?,?,?,?,?,?)""",
                (utcnow(), channel, kind, symbol, text, int(delivered), error))

    def alerts(self, limit: int = 100) -> List[sqlite3.Row]:
        return self._conn.execute(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()

    # -- calibration --------------------------------------------------------
    def write_prediction(self, p: Dict[str, Any]) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO predictions(ts, session_date, symbol, direction, confidence,
                                           horizon_days, entry_price)
                   VALUES(?,?,?,?,?,?,?)""",
                (utcnow(), p.get("session_date"), p["symbol"], p["direction"],
                 p["confidence"], p["horizon_days"], p["entry_price"]))

    def settle_predictions(self, prices: Dict[str, float]) -> int:
        """Mark open predictions against current prices. Returns rows settled.

        A prediction is only graded once its own horizon has elapsed. Settling
        early grades a five-session confidence claim on a one-session outcome,
        which silently destroys the calibration measurement -- the one number
        that tells you whether "confidence" meant anything at all.
        """
        order = {d: i for i, d in enumerate(self.session_dates())}
        rows = self._conn.execute(
            "SELECT * FROM predictions WHERE settled=0").fetchall()
        settled = 0
        for row in rows:
            price = prices.get(row["symbol"])
            if price is None or not row["entry_price"] or row["entry_price"] <= 0:
                # An unset entry price is a data bug, not a 0% return. Skip it
                # rather than dividing by zero or recording a fake outcome.
                continue
            made = order.get(row["session_date"])
            if made is None:
                continue
            latest = max(order.values()) if order else made
            if latest - made < int(row["horizon_days"]):
                continue  # horizon not yet elapsed
            ret = price / row["entry_price"] - 1.0
            if row["direction"] == "short":
                ret = -ret
            with self.tx() as c:
                c.execute(
                    """UPDATE predictions SET settled=1, exit_price=?, outcome_return=?, win=?
                       WHERE id=?""",
                    (price, ret, int(ret > 0), row["id"]))
            settled += 1
        return settled

    def calibration_table(self, bins: int = 5) -> List[Dict[str, Any]]:
        """Did 70% confidence actually mean ~70% right? This answers it."""
        rows = self._conn.execute(
            "SELECT confidence, win FROM predictions WHERE settled=1 AND win IS NOT NULL"
        ).fetchall()
        if not rows:
            return []
        out: List[Dict[str, Any]] = []
        for i in range(bins):
            lo, hi = i / bins, (i + 1) / bins
            bucket = [r for r in rows if lo <= r["confidence"] < hi or (i == bins - 1 and r["confidence"] == 1.0)]
            if not bucket:
                continue
            out.append({
                "bucket": f"{lo:.0%}-{hi:.0%}",
                "n": len(bucket),
                "mean_confidence": sum(r["confidence"] for r in bucket) / len(bucket),
                "realised_win_rate": sum(r["win"] for r in bucket) / len(bucket),
            })
        return out

    def brier_score(self) -> Optional[float]:
        """Lower is better; 0.25 is a coin flip."""
        rows = self._conn.execute(
            "SELECT confidence, win FROM predictions WHERE settled=1 AND win IS NOT NULL"
        ).fetchall()
        if not rows:
            return None
        return sum((r["confidence"] - r["win"]) ** 2 for r in rows) / len(rows)

    # -- introspection ------------------------------------------------------
    def session_dates(self) -> List[str]:
        return [r["session_date"] for r in self._conn.execute(
            "SELECT DISTINCT session_date FROM decisions WHERE session_date IS NOT NULL ORDER BY session_date")]
