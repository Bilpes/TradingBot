"""TradingEngine — the orchestrator that makes nine agents one product.

Cycle (every SCAN_INTERVAL seconds while the market session is live):
  1. refresh shared market state (index, VIX, breadth, macro, market news)
  2. per watchlist symbol: build MarketContext → run agents in parallel
  3. fuse reports → Signal (Strategy layer, regime-gated)
  4. size (Risk layer, fixed-fractional) → paper execute → Telegram alert
  5. manage open positions: stops, T1 partials, trailing, square-off clock
  6. persist equity point, refresh dashboard state

Invariants the engine enforces:
  • an agent crash never blocks trading (guarded_run normalises)
  • PAPER by default; LIVE path exists but is triple-gated (env flag +
    confirm file + per-day order cap) — see README "Going live"
"""

from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..agents import (AltDataAgent, MacroAgent, MLPredictionAgent,
                      MarketStructureAgent, NiftyLevelsAgent, RiskAgent,
                      SentimentAgent, TechnicalAgent, TimeSeriesAgent,
                      MarketContext, run_all_parallel, compute_sizing)
from ..broker.paper import PaperBroker, estimate_charges
from ..config import SETTINGS, data_dir
from ..data.fetchers import get_fii_dii, get_macro
from ..data.market_data import MarketData
from ..data.store import Store
from .. import indicators as ta
from ..logger import get_logger
from ..ml.predictor import Predictor, HORIZON_INTRADAY, HORIZON_SWING
from ..models import AgentReport, Position, Signal, Trade
from ..nlp.news import market_news, symbol_news
from .portfolio import Portfolio
from .strategy import StrategyFusion

log = get_logger("engine")
IST = timezone(timedelta(hours=5, minutes=30))

SECTOR_MAP = {  # approximate Nifty-50 sector buckets for concentration limits
    "BANK": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK",
             "BAJFINANCE", "BAJAJFINSV", "SBILIFE", "HDFCLIFE", "SHRIRAMFIN"],
    "IT": ["INFY", "TCS", "WIPRO", "TECHM", "HCLTECH", "PERSISTENT"],
    "AUTO": ["TATAMOTORS", "MARUTI", "EICHERMOT", "HEROMOTOCO", "BAJAJ-AUTO", "M&M"],
    "FMCG": ["ITC", "HINDUNILVR", "NESTLEIND", "TATACONSUM", "BRITANNIA", "UNITDSPR"],
    "ENERGY": ["RELIANCE", "ONGC", "BPCL", "COALINDIA", "NTPC", "POWERGRID", "ADANIGREEN"],
    "METALS": ["TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "JINDALSTEL"],
    "PHARMA": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP"],
    "OTHERS": ["LT", "BHARTIARTL", "TITAN", "ULTRACEMCO", "ASIANPAINT", "GRASIM",
               "ADANIENT", "ADANIPORTS", "BEL", "TRENT"],
}


def sector_of(symbol: str) -> str:
    for sec, names in SECTOR_MAP.items():
        if symbol in names:
            return sec
    return "OTHERS"


class TradingEngine:
    def __init__(self, store: Store, md: MarketData, symbols: List[str]):
        self.store = store
        self.md = md
        self.sim_mode = md.mode == "sim"
        self.portfolio = Portfolio(SETTINGS.capital_inr)
        self.portfolio.realized = store.realized_pnl_total()
        self.broker = PaperBroker()
        self.strategy = StrategyFusion(None, None)

        self.pred_intraday = Predictor(HORIZON_INTRADAY)
        self.pred_swing = Predictor(HORIZON_SWING)
        self.strategy = StrategyFusion(self.pred_intraday, self.pred_swing)

        self.agents = [
            MarketStructureAgent(), NiftyLevelsAgent(), TechnicalAgent(),
            MLPredictionAgent(self.pred_intraday, self.pred_swing),
            TimeSeriesAgent(), SentimentAgent(), AltDataAgent(),
            MacroAgent(), RiskAgent(),
        ]

        # universe = configured nifty50; watchlist = tradeable subset
        self.universe = symbols
        for s in store.watchlist():
            if s not in self.universe:
                self.universe.append(s)
        if not store.watchlist():
            for s in ["RELIANCE", "HDFCBANK", "INFY", "TATAMOTORS", "SBIN",
                      "ITC", "LT", "BHARTIARTL"]:
                store.add_symbol(s)

        # shared state
        self.shared: Dict = {}
        self.signals: deque = deque(maxlen=60)
        self.last_reports: Dict[str, List[dict]] = {}     # symbol -> reports
        self.market_status = "CLOSED"
        self.phase = "idle"
        self._started = datetime.now(IST)
        self._stop_file = data_dir() / "STOP"
        self._live_confirm = data_dir() / "LIVE_CONFIRM"

        # rebuild open positions
        self.positions: List[Position] = store.open_positions()
        for p in self.positions:
            p.ltp = self.md.ltp(p.symbol)

    # ==================================================================
    #  session helpers
    # ==================================================================
    def now_ist(self) -> datetime:
        return datetime.now(IST)

    def is_weekday(self) -> bool:
        return self.now_ist().weekday() < 5

    def market_open_now(self) -> bool:
        if self.sim_mode:
            return True                       # demo: pretend session is live
        if not self.is_weekday():
            return False
        n = self.now_ist()
        hm = n.hour * 60 + n.minute
        return 9 * 60 + 15 <= hm < 15 * 60 + 30

    def squareoff_due(self) -> bool:
        if self.sim_mode:
            return False
        n = self.now_ist()
        return self.is_weekday() and (n.hour * 60 + n.minute) >= 15 * 60 + 10

    def halted(self) -> bool:
        return self.portfolio.halted or self._stop_file.exists()

    # ==================================================================
    #  shared state
    # ==================================================================
    def refresh_shared(self) -> None:
        idx = self.md.index_daily()
        vix_now = self.md.india_vix()
        vix_prev = float(idx["close"].iloc[-1])  # placeholder; real VIX prev below
        try:
            import yfinance as yf
            hv = yf.Ticker("^INDIAVIX").history(period="10d")
            if len(hv) >= 2:
                vix_prev = float(hv["Close"].iloc[-2])
            elif len(hv) == 1:
                vix_prev = vix_now
        except Exception:
            vix_prev = vix_now
        vix_chg = (vix_now / vix_prev - 1) * 100 if vix_prev else 0.0

        breadth = self._compute_breadth()
        self.shared = {
            "index_daily": idx,
            "index_intraday": self.md.index_intraday(),
            "vix": vix_now,
            "vix_chg": vix_chg,
            "breadth": breadth,
            "macro": get_macro(),
            "fii_dii": get_fii_dii(),
            "market_news": market_news(),
            "prev_index_day": idx.iloc[-2] if len(idx) >= 2 else None,
        }

    def _compute_breadth(self) -> float:
        """% of universe stocks above their 50-EMA (cached dailies)."""
        above, total = 0, 0
        for s in self.universe[:50]:
            try:
                df = self.md.daily(s, 260)
                if len(df) < 55:
                    continue
                e50 = ta.ema(df["close"], 50).iloc[-1]
                total += 1
                if df["close"].iloc[-1] > e50:
                    above += 1
            except Exception:
                continue
        return above / total * 100 if total else 50.0

    # ==================================================================
    #  context building
    # ==================================================================
    def build_context(self, symbol: str) -> MarketContext:
        daily = self.md.daily(symbol, 400)
        intra = self.md.intraday_5m(symbol)
        vix = self.shared.get("vix", 14.0)
        vix_chg = self.shared.get("vix_chg", 0.0)
        sent = self.shared.get("_sym_sent", {}).get(symbol) or symbol_news(symbol)
        self.shared.setdefault("_sym_sent", {})[symbol] = sent

        feats = None
        try:
            from ..ml.features import build_features
            feats = build_features(daily, self.shared.get("index_daily"),
                                   vix, vix_chg, sent.get("score", 0.0))
        except Exception as e:
            log.debug("features failed for %s: %s", symbol, e)

        pf = self.portfolio.summary(self.positions, self.store.closed_positions_today())

        return MarketContext(
            symbol=symbol, style_hint="both",
            daily=daily, intraday=intra,
            index_daily=self.shared.get("index_daily"),
            index_intraday=self.shared.get("index_intraday"),
            vix=vix, vix_chg=vix_chg,
            sentiment=sent, market_sentiment=self.shared.get("market_news", {}),
            fii_dii=self.shared.get("fii_dii", {}), macro=self.shared.get("macro", {}),
            breadth=self.shared.get("breadth", 50.0),
            prev_index_day=self.shared.get("prev_index_day"),
            positions=self.positions, portfolio=pf,
            features_intraday=feats, features_swing=feats,
        )

    # ==================================================================
    #  main cycle
    # ==================================================================
    def run_cycle(self, do_entries: bool = True) -> dict:
        t0 = time.time()
        self.refresh_shared()

        # --- 1. manage exits first (protect capital) -----------------------
        self.manage_positions()

        # --- 2. entries ------------------------------------------------------
        stats = {"scanned": 0, "proposed": 0, "taken": 0}
        if do_entries and self.market_open_now() and not self.halted():
            for symbol in self.store.watchlist():
                try:
                    n = self.scan_symbol(symbol)
                    stats["scanned"] += 1
                    stats["taken"] += n
                except Exception as e:
                    log.exception("scan failed for %s: %s", symbol, e)

        # --- 3. persist ------------------------------------------------------
        pf = self.portfolio.summary(self.positions, self.store.closed_positions_today())
        self.store.save_equity_point(pf["equity"], pf["realized_total"], pf["unrealized"])
        self.market_status = "OPEN (sim)" if (self.sim_mode and self.market_open_now()) \
            else ("OPEN" if self.market_open_now() else "CLOSED")
        log.info("cycle done in %.1fs: scanned=%d taken=%d equity=%.0f",
                 time.time() - t0, stats["scanned"], stats["taken"], pf["equity"])
        return stats

    # ------------------------------------------------------------------
    def scan_symbol(self, symbol: str) -> int:
        ctx = self.build_context(symbol)
        reports = run_all_parallel(self.agents, ctx)
        self.last_reports[symbol] = [r.to_dict() for r in reports]
        self.store.save_agent_run(symbol, [r.to_dict() for r in reports])

        taken = 0
        for mode in ("intraday", "swing"):
            if len([p for p in self.positions if p.style == ("MIS" if mode == "intraday" else "CNC")]) \
                    >= SETTINGS.max_open_positions:
                continue
            # one open position per symbol per style — no pyramiding
            style = "MIS" if mode == "intraday" else "CNC"
            if any(p.symbol == symbol and p.style == style for p in self.positions):
                continue
            sig = self.strategy.decide(ctx, reports, self.portfolio, mode)
            if sig is None:
                continue
            self.signals.appendleft(sig.to_dict())

            # risk sizing
            cur_notional = sum(p.notional for p in self.positions if p.symbol == symbol)
            sec = sector_of(symbol)
            sec_notional = sum(p.notional for p in self.positions if sector_of(p.symbol) == sec)
            qty, reason = compute_sizing(sig.entry, sig.stop, self.portfolio.equity,
                                         cur_notional, sec_notional)
            if not qty:
                for s in self.signals:
                    if s["symbol"] == symbol and s["status"] == "PROPOSED":
                        s["status"] = "REJECTED"
                        s["reject_reason"] = reason
                continue

            sig.qty = qty
            fill = self.broker.execute(symbol, sig.side, qty, sig.entry, sig.style)
            if fill is None:
                continue
            pos = Position(
                symbol=symbol, side=sig.side, style=sig.style, qty=qty,
                entry_price=fill.price, entry_ts=fill.ts, stop=sig.stop,
                target1=sig.target1, target2=sig.target2, ltp=fill.price,
                setup=sig.setup, regime=sig.regime, rationale=sig.rationale,
                score=sig.score, ml_prob=sig.ml_prob, entry_charges=fill.charges,
                high_water=fill.price)
            self.positions.append(pos)
            self.store.save_position(pos)
            for s in self.signals:
                if s["symbol"] == symbol and s["status"] == "PROPOSED" and s["qty"] == 0:
                    s["qty"] = qty
                    s["status"] = "TAKEN"
            taken += 1
            from ..notify.telegram import Telegram
            Telegram().trade_entry(pos)
        return taken

    # ==================================================================
    #  position management
    # ==================================================================
    def manage_positions(self) -> None:
        e20_cache: Dict[str, pd.Series] = {}
        for pos in list(self.positions):
            ltp = self.md.ltp(pos.symbol)
            pos.ltp = ltp
            pos.high_water = max(pos.high_water, ltp) if pos.side == "BUY" \
                else min(pos.high_water, ltp)

            entry_dt = datetime.fromisoformat(pos.entry_ts)
            if entry_dt.tzinfo is None:              # legacy naive stamps → assume IST
                entry_dt = entry_dt.replace(tzinfo=IST)
            held_days = (self.now_ist() - entry_dt).days
            pos.days_held = held_days

            force_exit = self.squareoff_due() and pos.style == "MIS"
            time_exit = pos.style == "CNC" and held_days >= 12

            direction = 1 if pos.side == "BUY" else -1
            hit_stop = (ltp <= pos.stop) if pos.side == "BUY" else (ltp >= pos.stop)
            hit_t1 = (ltp >= pos.target1) if pos.side == "BUY" else (ltp <= pos.target1)
            hit_t2 = (ltp >= pos.target2) if pos.side == "BUY" else (ltp <= pos.target2)

            # trailing for swing after T1: exit rest if close < EMA20(daily)
            trail_exit = False
            if pos.style == "CNC" and pos.partial_booked:
                if pos.symbol not in e20_cache:
                    e20_cache[pos.symbol] = ta.ema(self.md.daily(pos.symbol)["close"], 20)
                if (pos.side == "BUY" and ltp < e20_cache[pos.symbol].iloc[-1]) or \
                        (pos.side == "SELL" and ltp > e20_cache[pos.symbol].iloc[-1]):
                    trail_exit = True

            if hit_stop or force_exit or time_exit or trail_exit:
                reason = "SQUAREOFF" if force_exit else ("TIME_EXIT" if time_exit else
                          ("TRAIL" if trail_exit else "STOP"))
                self._close_position(pos, pos.qty, ltp, reason)

            elif hit_t2 and pos.qty > 0:
                self._close_position(pos, pos.qty, ltp, "TARGET2")
            elif hit_t1 and not pos.partial_booked:
                k = max(1, int(pos.qty * (0.6 if pos.style == "MIS" else 0.5)))
                k = min(k, pos.qty - 1) if pos.qty > 1 else 0
                if k > 0:
                    self._close_position(pos, k, ltp, "TARGET1", partial=True)
                    # trail the runner: stop to breakeven +0.2R
                    r0 = abs(pos.entry_price - pos.stop) if pos.stop else ltp * 0.01
                    pos.stop = pos.entry_price + direction * 0.2 * r0
                    pos.partial_booked = True
                    self.store.save_position(pos)
                else:
                    pos.partial_booked = True   # single-share position: just trail

            self.store.save_position(pos)

    # ------------------------------------------------------------------
    def _close_position(self, pos: Position, qty: int, price: float,
                        reason: str, partial: bool = False) -> None:
        direction = 1 if pos.side == "BUY" else -1
        exit_side = "SELL" if pos.side == "BUY" else "BUY"
        fill = self.broker.execute(pos.symbol, exit_side, qty, price, pos.style)
        if fill is None:
            log.error("paper exit failed for %s — liquidity/cash mismatch", pos.symbol)
            return

        gross = (fill.price - pos.entry_price) * direction * qty
        alloc_entry = pos.entry_charges * (qty / max(pos.qty, 1))
        net = gross - alloc_entry - fill.charges

        pos.realized_pnl += net
        pos.realized_charges += alloc_entry + fill.charges
        pos.entry_charges -= alloc_entry
        pos.qty -= qty

        trade = Trade(position_id=pos.position_id, symbol=pos.symbol, side=pos.side,
                      style=pos.style, qty=qty, entry_price=pos.entry_price,
                      exit_price=fill.price, entry_ts=pos.entry_ts, exit_ts=fill.ts,
                      gross_pnl=gross, charges=alloc_entry + fill.charges,
                      net_pnl=net, setup=pos.setup, exit_reason=reason,
                      held_days=pos.days_held)
        self.store.save_trade(trade)
        self.portfolio.realized += net
        self.portfolio.realized_day += net

        if not partial or pos.qty <= 0:
            pos.status = "CLOSED"
            self.positions = [p for p in self.positions if p.position_id != pos.position_id]
        self.store.save_position(pos)

        from ..notify.telegram import Telegram
        Telegram().trade_exit(trade, pos)

    # ==================================================================
    #  manual controls
    # ==================================================================
    def square_off_all(self, style: Optional[str] = None) -> int:
        n = 0
        for pos in list(self.positions):
            if style and pos.style != style:
                continue
            self._close_position(pos, pos.qty, self.md.ltp(pos.symbol), "SQUAREOFF")
            n += 1
        return n

    def toggle_halt(self) -> bool:
        self.portfolio.halted = not self.portfolio.halted
        return self.portfolio.halted

    # ==================================================================
    #  API state
    # ==================================================================
    def state(self) -> dict:
        pf = self.portfolio.summary(self.positions, self.store.closed_positions_today())
        return {
            "product": "AlphaDesk",
            "mode": "PAPER" if not SETTINGS.live_trading else "LIVE (gated)",
            "data_feed": self.md.mode,
            "market": {"status": self.market_status,
                       "ist_time": self.now_ist().strftime("%H:%M:%S"),
                       "phase": self.phase},
            "portfolio": pf,
            "positions": [p.to_dict() for p in self.positions],
            "trades": self.store.trades(40),
            "watchlist": self.store.watchlist(),
            "universe_size": len(self.universe),
            "agents": {s: [r for r in reps] for s, reps in list(self.last_reports.items())[:8]},
            "agent_summary": self._agent_summary(),
            "signals": list(self.signals)[:20],
            "equity_curve": self.store.equity_curve(300),
            "ml": {"intraday": self.pred_intraday.info(),
                   "swing": self.pred_swing.info()},
            "stats": self.store.win_stats(),
            "uptime_started": self._started.isoformat(timespec="seconds"),
            "live_gates": {"live_trading_env": SETTINGS.live_trading,
                           "confirm_file": self._live_confirm.exists()},
        }

    def _agent_summary(self) -> List[dict]:
        """One row per agent — latest run across symbols (UI grid)."""
        rows: Dict[str, dict] = {}
        for sym, reps in self.last_reports.items():
            for r in reps:
                row = rows.setdefault(r["agent"], {
                    "agent": r["agent"], "status": r["status"], "score": 0.0,
                    "note": r["note"], "elapsed_ms": 0, "symbols": 0})
                row["score"] += r["score"]
                row["elapsed_ms"] += r.get("elapsed_ms", 0)
                row["symbols"] += 1
        for row in rows.values():
            row["score"] = round(row["score"] / max(row["symbols"], 1), 1)
        return sorted(rows.values(), key=lambda x: x["agent"])
