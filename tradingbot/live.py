"""Paper trading runner.

Wires the pieces together for a live-ish run: sector worker processes produce
signals, the coordinator applies consensus and risk, the paper broker fills with
real NSE charges, and Telegram reports each decision.

Nothing here places a real order. The only path to live trading is constructing
``DhanPaperOrders`` with a real ``dhanhq.Order`` and swapping the broker -- a
deliberately separate, deliberate step.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from tradingbot.agents.registry import SectorProfile
from tradingbot.core.consensus import ConsensusConfig
from tradingbot.core.risk import RiskConfig
from tradingbot.indicators import atr
from tradingbot.indicators import annualised_vol
from tradingbot.data.schema import History
from tradingbot.markets.nse import NSE_SECTORS, nse_sector_of
from tradingbot.notify.telegram import TelegramNotifier, daily_summary, risk_alert
from tradingbot.paper.broker import NSECostModel, PaperBroker
from tradingbot.store.db import Store
from tradingbot.workers.coordinator import Coordinator
from tradingbot.workers.sector_worker import MarketWorker, SectorWorker, stale_sectors

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)

log = logging.getLogger(__name__)


def is_market_open(now: Optional[datetime] = None) -> bool:
    """NSE regular session, Monday-Friday 09:15-15:30 IST.

    Exchange holidays are *not* modelled -- a holiday is indistinguishable from a
    data outage without a holiday calendar, so treat a silent session as
    suspicious rather than as a closed market.
    """
    now = now or datetime.now(IST)
    if now.weekday() >= 5:
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


@dataclass
class PaperRunConfig:
    capital: float = 100_000.0
    product: str = "CNC"
    poll_seconds: float = 300.0
    cycle_seconds: float = 60.0
    duration_days: int = 14
    sectors: List[str] = field(default_factory=lambda: list(NSE_SECTORS))
    risk: RiskConfig = field(default_factory=RiskConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    respect_market_hours: bool = True


def latest_prices(history: History) -> Dict[str, float]:
    return {s: float(df["close"].iloc[-1]) for s, df in history.items() if len(df)}


def latest_atr(history: History, window: int = 14) -> Dict[str, float]:
    out = {}
    for symbol, df in history.items():
        series = atr(df["high"], df["low"], df["close"], window)
        if len(series) and pd.notna(series.iloc[-1]):
            out[symbol] = float(series.iloc[-1])
    return out


def latest_vol(history: History, window: int = 21) -> Dict[str, float]:
    out = {}
    for symbol, df in history.items():
        series = annualised_vol(df["close"], window)
        if len(series) and pd.notna(series.iloc[-1]):
            out[symbol] = float(series.iloc[-1])
    return out


def run_in_process(
    history: History,
    store: Store,
    broker: PaperBroker,
    profiles: Dict[str, SectorProfile],
    notifier: Optional[TelegramNotifier] = None,
    config: Optional[PaperRunConfig] = None,
    sector_of: Callable[[str], str] = nse_sector_of,
) -> dict:
    """Run every sector worker inline, then one coordinator cycle.

    This is the single-process path used by tests and by
    ``cli.py paper-step``. It produces exactly the same store contents as the
    multiprocess path, which is what makes the parallel version safe to trust.
    """
    from tradingbot.agents.registry import profile_for

    config = config or PaperRunConfig()
    notifier = notifier or TelegramNotifier()
    session = pd.Timestamp(max(df.index[-1] for df in history.values())).strftime("%Y-%m-%d")

    # Cross-sectional agents run once over the whole universe; handing them a
    # single sector's slice would give them a cross-section of one.
    market_profile = profiles.get(MarketWorker.sector_tag) or profile_for("Banking")
    total_signals = MarketWorker(market_profile, store.path,
                                 sector_map=sector_of).run_cycle(history, session)

    for sector in config.sectors:
        symbols = [s for s in history if sector_of(s) == sector]
        if not symbols:
            continue
        worker = SectorWorker(sector, symbols, profiles.get(sector) or profile_for(sector),
                              store.path, sector_map=sector_of)
        total_signals += worker.run_cycle(history, session)

    coordinator = Coordinator(
        store=store, broker=broker, risk=config.risk, consensus=config.consensus,
        notifier=notifier, sector_of=sector_of,
    )
    report = coordinator.run_cycle(
        session_date=session,
        prices=latest_prices(history),
        atr_values=latest_atr(history, config.risk.atr_window),
        vol_values=latest_vol(history, config.risk.vol_lookback),
    )

    store.set_meta("paper_stats", broker.realised_stats())
    store.set_meta("run_info", {"mode": "paper", "universe": f"{len(history)} NSE symbols",
                                "session": session})
    return {"session": session, "signals": total_signals, "report": report,
            "equity": broker.equity(latest_prices(history))}


def serve_dashboard(db_path: str, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the dashboard. Binds 0.0.0.0 so it works behind a proxy."""
    import os

    import uvicorn

    os.environ["TRADINGBOT_DB"] = db_path
    from tradingbot.web.app import create_app

    uvicorn.run(create_app(db_path), host=host, port=port, log_level="info")


def run_forever(
    load_history: Callable[[], History],
    store_path: str,
    profiles: Dict[str, SectorProfile],
    config: Optional[PaperRunConfig] = None,
    notifier: Optional[TelegramNotifier] = None,
    sector_of: Callable[[str], str] = nse_sector_of,
) -> None:  # pragma: no cover - long-running loop
    """Two-week paper run: workers in parallel processes, coordinator in the main one."""
    config = config or PaperRunConfig()
    notifier = notifier or TelegramNotifier()
    store = Store(store_path)
    broker = PaperBroker(capital=config.capital,
                         costs=NSECostModel(product=config.product))

    history = load_history()
    store.set_meta("run_info", {"mode": "paper", "universe": f"{len(history)} NSE symbols",
                                "started": pd.Timestamp.now(IST).isoformat(),
                                "capital": config.capital})

    procs, stop = _spawn(store_path, load_history, profiles, config)
    deadline = time.monotonic() + config.duration_days * 86_400
    last_session = None
    try:
        while time.monotonic() < deadline:
            if config.respect_market_hours and not is_market_open():
                time.sleep(60)
                continue

            history = load_history()
            session = pd.Timestamp(max(df.index[-1] for df in history.values())).strftime("%Y-%m-%d")

            quiet = stale_sectors(store_path, session, config.sectors)
            if quiet:
                notifier.send(risk_alert("stale workers", f"no signals from: {', '.join(quiet)}"),
                              kind="risk")

            coordinator = Coordinator(store=store, broker=broker, risk=config.risk,
                                      consensus=config.consensus, notifier=notifier,
                                      sector_of=sector_of)
            report = coordinator.run_cycle(
                session_date=session,
                prices=latest_prices(history),
                atr_values=latest_atr(history, config.risk.atr_window),
                vol_values=latest_vol(history, config.risk.vol_lookback),
            )
            store.set_meta("paper_stats", broker.realised_stats())
            log.info("cycle %s: %d decisions, %d actionable, %d entries",
                     session, report.decisions, report.actionable, len(report.entries))

            if session != last_session and last_session is not None:
                notifier.send(daily_summary(
                    equity=broker.equity(latest_prices(history)),
                    cash=broker.cash,
                    exposure=broker.exposure(latest_prices(history)),
                    drawdown=broker.drawdown(latest_prices(history)),
                    positions=len(broker.positions),
                    stats=broker.realised_stats(),
                    session=last_session), kind="summary")
            last_session = session
            time.sleep(config.cycle_seconds)
    finally:
        stop.set()
        for proc in procs:
            proc.join(timeout=10)
        store.close()


def _spawn(store_path, load_history, profiles, config):  # pragma: no cover
    from tradingbot.workers.sector_worker import spawn_workers

    return spawn_workers(store_path=store_path, load_history=load_history,
                         sectors=config.sectors, poll_seconds=config.poll_seconds,
                         profiles=profiles,
                         market_profile=profiles.get(MarketWorker.sector_tag))
