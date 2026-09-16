"""Simulate the paper run without waiting two weeks for it.

Steps the window forward one session at a time, running the full worker ->
consensus -> paper-broker cycle at each step. Same code path as a live run,
just with the clock moved by hand -- which is what makes the dashboard worth
looking at before any real session happens.

    python scripts/paper_walk.py --sessions 500 --steps 120 --db runs/paper.sqlite3
    python -m tradingbot.cli dashboard --db runs/paper.sqlite3

This is not a backtest and is not evidence of an edge: it runs on synthetic
factor data. It exists to exercise the operational path end to end and to fill
the decision ledger so the reasoning view has something to show.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from tradingbot.agents.registry import DEFAULT_PROFILES
from tradingbot.data.providers import SyntheticProvider
from tradingbot.live import (
    PaperRunConfig, latest_atr, latest_vol, latest_prices,
)
from tradingbot.markets.nse import NSE_SECTORS, NSE_UNIVERSE, nse_sector_of
from tradingbot.notify.telegram import FakeTransport, TelegramNotifier
from tradingbot.paper.broker import NSECostModel, PaperBroker
from tradingbot.store.db import Store
from tradingbot.workers.coordinator import Coordinator
from tradingbot.workers.sector_worker import MarketWorker, SectorWorker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=500)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--db", type=str, default="runs/paper.sqlite3")
    parser.add_argument("--fresh", action="store_true", help="delete the store first")
    args = parser.parse_args()

    if args.fresh:
        for suffix in ("", "-wal", "-shm"):
            Path(args.db + suffix).unlink(missing_ok=True)
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)

    full = SyntheticProvider(sessions=args.sessions, seed=args.seed,
                             sector_map=nse_sector_of,
                             universe=list(NSE_UNIVERSE)).load([], "2021-01-04")
    dates = sorted(next(iter(full.values())).index)
    warmup = 260
    if warmup + args.steps > len(dates):
        raise SystemExit(f"need at least {warmup + args.steps} sessions, have {len(dates)}")

    store = Store(args.db)
    broker = PaperBroker(capital=args.capital, costs=NSECostModel(product="CNC"))
    notifier = TelegramNotifier(bot_token="", chat_id="", transport=FakeTransport())
    config = PaperRunConfig(sectors=NSE_SECTORS)

    market_profile = DEFAULT_PROFILES["Banking"]
    store.set_meta("run_info", {
        "mode": "paper (walk-forward simulation)",
        "universe": f"{len(full)} NSE symbols",
        "capital": args.capital,
        "seed": args.seed,
    })

    for step in range(args.steps):
        cutoff = dates[warmup + step]
        history = {s: df.loc[:cutoff] for s, df in full.items()}
        session = pd.Timestamp(cutoff).strftime("%Y-%m-%d")

        written = MarketWorker(market_profile, store.path,
                               sector_map=nse_sector_of).run_cycle(history, session)
        for sector in NSE_SECTORS:
            symbols = [s for s in history if nse_sector_of(s) == sector]
            if symbols:
                written += SectorWorker(sector, symbols, DEFAULT_PROFILES[sector],
                                        store.path, sector_map=nse_sector_of
                                        ).run_cycle(history, session)

        coordinator = Coordinator(store=store, broker=broker, risk=config.risk,
                                  consensus=config.consensus, notifier=notifier,
                                  sector_of=nse_sector_of)
        report = coordinator.run_cycle(
            session_date=session,
            prices=latest_prices(history),
            atr_values=latest_atr(history, config.risk.atr_window),
            vol_values=latest_vol(history, config.risk.vol_lookback),
        )
        store.set_meta("paper_stats", broker.realised_stats())
        if step % 20 == 0 or step == args.steps - 1:
            print(f"{session}  signals={written:4d} decisions={report.decisions:3d} "
                  f"actionable={report.actionable:2d} entries={len(report.entries):2d} "
                  f"positions={len(broker.positions):2d} "
                  f"equity=Rs {broker.equity(latest_prices(history)):,.2f}")

    stats = broker.realised_stats()
    prices = latest_prices({s: df.loc[:dates[warmup + args.steps - 1]] for s, df in full.items()})
    print("\n" + "=" * 58)
    print(f"  Sessions simulated   {args.steps}")
    print(f"  Final equity         Rs {broker.equity(prices):,.2f}")
    print(f"  Return               {broker.equity(prices) / args.capital - 1:+.2%}")
    print(f"  Closed trades        {stats['n_trades']}")
    print(f"  Win rate             {stats['win_rate']:.1%}")
    print(f"  Realised P&L         Rs {stats['realised_pnl']:+,.2f}")
    print(f"  Charges paid         Rs {stats['total_charges']:,.2f}")
    print(f"  Brier score          {store.brier_score()}")
    print("=" * 58)
    print("\nSynthetic data: this exercises the machinery, not an edge.")
    print(f"Dashboard:  python -m tradingbot.cli dashboard --db {args.db}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
