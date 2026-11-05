"""FastAPI server: serves the dark trading-desk dashboard and the JSON API.

Endpoints:
  GET  /                 → dashboard (tradingbot/web/index.html)
  GET  /api/state        → full bot state (portfolio, positions, agents…)
  POST /api/watchlist    {"symbol": "TCS"}    → add symbol to watchlist
  DELETE /api/watchlist/{symbol}               → remove symbol
  POST /api/control      {"action": "halt"|"resume"|"squareoff"|"scan"}
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from ..logger import get_logger
from ..models import Position

log = get_logger("api")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class WatchRequest(BaseModel):
    symbol: str


class ControlRequest(BaseModel):
    action: str


def create_app(engine) -> FastAPI:
    app = FastAPI(title="AlphaDesk", version="1.0.0", docs_url="/api/docs")

    @app.get("/", response_class=HTMLResponse)
    def index():
        html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    @app.get("/api/state")
    def state():
        try:
            return JSONResponse(engine.state())
        except Exception as e:
            log.exception("state failed")
            raise HTTPException(500, str(e))

    @app.post("/api/watchlist")
    def add_symbol(req: WatchRequest):
        symbol = req.symbol.upper().strip()
        if not symbol or len(symbol) > 20:
            raise HTTPException(400, "invalid symbol")
        engine.store.add_symbol(symbol)
        if symbol not in engine.universe:
            engine.universe.append(symbol)
        if engine.md.mode != "sim" and engine.md.sim is not None:
            from ..data.simulator import SymbolSimulator
            engine.md.sim.sims.setdefault(symbol, SymbolSimulator(symbol))
        log.info("watchlist + %s", symbol)
        return {"ok": True, "watchlist": engine.store.watchlist()}

    @app.delete("/api/watchlist/{symbol}")
    def remove_symbol(symbol: str):
        engine.store.remove_symbol(symbol)
        log.info("watchlist − %s", symbol)
        return {"ok": True, "watchlist": engine.store.watchlist()}

    @app.post("/api/control")
    def control(req: ControlRequest):
        action = req.action.lower().strip()
        if action == "halt":
            halted = engine.toggle_halt()
            from ..notify.telegram import Telegram
            Telegram().risk_alert("Manual HALT " + ("ON" if halted else "OFF"))
            return {"ok": True, "halted": halted}
        if action == "squareoff":
            n = engine.square_off_all()
            return {"ok": True, "closed": n}
        if action == "scan":
            stats = engine.run_cycle(do_entries=True)
            return {"ok": True, "stats": stats}
        raise HTTPException(400, f"unknown action {action}")

    @app.exception_handler(Exception)
    async def unhandled(request, exc):
        log.exception("api error: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)

    return app
