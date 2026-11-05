"""AlphaDesk entry point.

    python main.py                # paper trading with best available feed
    python main.py --simulate     # force offline synthetic feed (demo)
    python main.py --once         # run one scan cycle and exit (CI / smoke test)
    python main.py --port 8000    # dashboard port

The API server runs on a background thread; the scheduler drives the desk
from the main thread. Ctrl-C shuts both down cleanly.
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

from tradingbot.api.server import create_app
from tradingbot.config import PROJECT_ROOT, SETTINGS, data_dir
from tradingbot.data.market_data import MarketData
from tradingbot.data.store import Store
from tradingbot.engine.orchestrator import TradingEngine
from tradingbot.engine.scheduler import Scheduler
from tradingbot.logger import get_logger
from tradingbot.notify.telegram import Telegram
from tradingbot import __product__, __version__

log = get_logger("main")


def main() -> None:
    ap = argparse.ArgumentParser(description=f"{__product__} — multi-agent Indian equity desk")
    ap.add_argument("--simulate", action="store_true", help="force offline synthetic feed")
    ap.add_argument("--once", action="store_true", help="run a single cycle then exit")
    ap.add_argument("--port", type=int, default=SETTINGS.port)
    ap.add_argument("--no-ui", action="store_true", help="skip API/dashboard (headless)")
    args = ap.parse_args()

    mode = "sim" if args.simulate else SETTINGS.resolved_data_mode()
    log.info("%s v%s starting · mode=%s paper=%s capital=₹%.0f risk=₹%.0f/trade",
             __product__, __version__, mode, not SETTINGS.live_trading,
             SETTINGS.capital_inr, SETTINGS.risk_amount_inr)

    store = Store(data_dir() / "alphadesk.db")
    settings_for_universe = SETTINGS.load_universe()
    md = MarketData(settings_for_universe, mode=mode)
    engine = TradingEngine(store, md, settings_for_universe)

    if args.once:
        stats = engine.run_cycle(do_entries=True)
        log.info("one-cycle done: %s", stats)
        print(f"one-cycle stats: {stats} · equity ₹{engine.portfolio.equity:,.0f}")
        return

    # uvicorn on a worker thread
    if not args.no_ui:
        app = create_app(engine)
        ui = threading.Thread(
            target=lambda: __import__("uvicorn").run(
                app, host="0.0.0.0", port=args.port, log_level="warning"),
            daemon=True, name="uvicorn")
        ui.start()
        log.info("dashboard → http://localhost:%d", args.port)
        time.sleep(1.5)

    tg = Telegram()
    if tg.enabled:
        tg.send(f"🚀 <b>{__product__}</b> desk online · feed <b>{md.mode}</b> · "
                f"paper mode · capital ₹{SETTINGS.capital_inr:,.0f}")

    sched = Scheduler(engine)
    sched.start()

    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        log.info("shutdown signal received")
    finally:
        # persist everything
        for p in engine.positions:
            store.save_position(p)
        log.info("desk closed. goodbye.")


if __name__ == "__main__":
    main()
