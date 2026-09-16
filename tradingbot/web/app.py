"""Decision dashboard.

Read-only over the same SQLite the workers and coordinator write to, so it adds
no load to the strategy loop and can be restarted freely.

Every decision is shown with its complete reasoning chain: each agent's vote,
whether it agreed or dissented, the blended confidence, and -- when nothing was
bought -- the exact rule that rejected it. A system that only shows you its
trades is hiding the half of its behaviour that matters most.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from tradingbot.store.db import Store

TEMPLATE_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

DEFAULT_DB = os.environ.get("TRADINGBOT_DB", "runs/paper.sqlite3")


def create_app(db_path: str | None = None) -> FastAPI:
    app = FastAPI(title="TradingBot", docs_url="/api/docs", openapi_url="/api/openapi.json")
    path = db_path or DEFAULT_DB

    def store() -> Store:
        # One connection per request; SQLite WAL makes concurrent reads cheap.
        return Store(path)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Any:
        return templates.TemplateResponse(request, "dashboard.html", {"db_path": path})

    @app.get("/api/state")
    def state() -> Dict[str, Any]:
        s = store()
        try:
            curve = s.equity_curve()
            positions = s.latest_positions()
            equity = curve[-1] if curve else None
            cal = s.calibration_table()
            brier = s.brier_score()
            stats = s.get_meta("paper_stats", {})
            return {
                "db_path": path,
                "equity": dict(equity) if equity else None,
                "curve": [dict(r) for r in curve],
                "positions": [dict(r) for r in positions],
                "calibration": cal,
                "brier_score": brier,
                "stats": stats,
                "sessions": s.session_dates(),
                "n_decisions": s._conn.execute("SELECT COUNT(*) c FROM decisions").fetchone()["c"],
                "n_signals": s._conn.execute("SELECT COUNT(*) c FROM signals").fetchone()["c"],
                "meta": s.get_meta("run_info", {}),
            }
        finally:
            s.close()

    @app.get("/api/decisions")
    def decisions(limit: int = Query(200, le=2000), session: Optional[str] = None) -> List[Dict]:
        s = store()
        try:
            rows = s.recent_decisions(limit=limit, session_date=session)
            out = []
            for row in rows:
                item = dict(row)
                item["votes"] = json.loads(item.pop("votes_json") or "{}")
                out.append(item)
            return out
        finally:
            s.close()

    @app.get("/api/decisions/{decision_id}")
    def decision(decision_id: int) -> Dict[str, Any]:
        s = store()
        try:
            row = s._conn.execute(
                "SELECT * FROM decisions WHERE id=?", (decision_id,)).fetchone()
            if row is None:
                raise HTTPException(404, "decision not found")
            item = dict(row)
            item["votes"] = json.loads(item.pop("votes_json") or "{}")
            return item
        finally:
            s.close()

    @app.get("/api/signals")
    def signals(session: Optional[str] = None, symbol: Optional[str] = None,
                limit: int = Query(500, le=5000)) -> List[Dict]:
        s = store()
        try:
            clauses, params = [], []
            if session:
                clauses.append("session_date=?")
                params.append(session)
            if symbol:
                clauses.append("symbol=?")
                params.append(symbol)
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = s._conn.execute(
                f"SELECT * FROM signals {where} ORDER BY id DESC LIMIT ?", (*params, limit)
            ).fetchall()
            out = []
            for row in rows:
                item = dict(row)
                item["parts"] = json.loads(item.pop("parts_json") or "{}")
                out.append(item)
            return out
        finally:
            s.close()

    @app.get("/api/alerts")
    def alerts(limit: int = Query(100, le=1000)) -> List[Dict]:
        s = store()
        try:
            return [dict(r) for r in s.alerts(limit=limit)]
        finally:
            s.close()

    @app.get("/api/rejections")
    def rejections(limit: int = Query(200, le=2000)) -> List[Dict]:
        """Every idea the system considered and declined, with the rule that declined it."""
        s = store()
        try:
            rows = s._conn.execute(
                """SELECT session_date, symbol, sector, score, confidence, agreeing,
                          dissenting, neutral, vetoed, rejection
                   FROM decisions WHERE actionable=0 ORDER BY id DESC LIMIT ?""", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            s.close()

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok", "db": path, "exists": str(Path(path).exists())}

    return app


app = create_app()
