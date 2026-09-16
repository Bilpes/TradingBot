"""Command-line interface.

    tradingbot backtest  --market nse --sessions 900 --seed 7
    tradingbot sectors   --market nse
    tradingbot paper-step --market nse            # one cycle, writes the store
    tradingbot dashboard                          # serve the decision dashboard
    tradingbot profiles-dump --out profiles.json  # edit, then pass back via --profiles
    tradingbot dhan-check                         # verify Dhan credentials
    tradingbot telegram-check                     # verify Telegram credentials
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, Optional, Sequence

import pandas as pd

from tradingbot.core.engine import Engine, EngineConfig
from tradingbot.core.risk import RiskConfig
from tradingbot.data.providers import CSVProvider, DataProvider, SyntheticProvider
from tradingbot.data.schema import History, closes
from tradingbot.markets.nse import NSE_SECTORS, NSE_UNIVERSE, nse_sector_of

log = logging.getLogger("tradingbot")


# -- data -------------------------------------------------------------------
def _build_provider(args: argparse.Namespace) -> DataProvider:
    if args.provider == "synthetic":
        sector_map = nse_sector_of if args.market == "nse" else None
        return SyntheticProvider(sessions=args.sessions, seed=args.seed,
                                 start=args.start, sector_map=sector_map)
    if args.provider == "csv":
        if not args.csv:
            raise SystemExit("--provider csv requires --csv PATH")
        return CSVProvider(args.csv)
    if args.provider == "dhan":
        from tradingbot.data.dhan import DhanProvider, load_security_map

        if not args.security_map:
            raise SystemExit("--provider dhan requires --security-map MAP.json "
                             "(build it with scripts/make_security_map.py)")
        return DhanProvider(security_map=load_security_map(args.security_map))
    raise SystemExit(f"unknown provider {args.provider!r}")


def _symbols(args: argparse.Namespace) -> list[str]:
    if args.provider == "csv" or args.provider == "dhan":
        return []  # the source decides
    return list(NSE_UNIVERSE) if args.market == "nse" else list(
        __import__("tradingbot.data.universe", fromlist=["UNIVERSE"]).UNIVERSE)


def _load(args: argparse.Namespace) -> History:
    return _build_provider(args).load(_symbols(args), args.start, args.end)


def _sector_of(args: argparse.Namespace):
    if args.market == "nse":
        return nse_sector_of
    from tradingbot.data.universe import sector_of

    return sector_of


def _profiles(args: argparse.Namespace) -> Dict:
    from tradingbot.agents.registry import DEFAULT_PROFILES, load_profiles

    return load_profiles(args.profiles) if args.profiles else dict(DEFAULT_PROFILES)


def _risk(args: argparse.Namespace) -> RiskConfig:
    return RiskConfig(
        risk_per_trade=args.risk_per_trade,
        max_position_weight=args.max_position,
        max_sector_weight=args.max_sector,
        target_vol=args.target_vol,
        min_confidence=args.min_confidence,
        entry_threshold=args.entry_threshold,
        exit_confidence=args.exit_confidence,
        min_holding_days=args.min_holding_days,
    )


# -- commands ---------------------------------------------------------------
def cmd_backtest(args: argparse.Namespace) -> int:
    history = _load(args)
    engine = Engine(config=EngineConfig(risk=_risk(args), initial_cash=args.cash),
                    sector_map=_sector_of(args))

    print(f"Market     {args.market.upper()}")
    print(f"Universe   {len(history)} symbols / {len({_sector_of(args)(s) for s in history})} sectors")
    print(f"Sessions   {len(next(iter(history.values())))}")
    print(f"Agents     {', '.join(a.name for a in engine.agents)}")
    print(f"Warmup     {engine.warmup} sessions\n")

    result = engine.run(history)
    print(result.summary())

    if args.show_decisions and not result.decisions.empty:
        cols = ["date", "symbol", "stage", "action", "shares", "price", "reason"]
        print(f"\nLast {args.show_decisions} decisions:")
        print(result.decisions[cols].tail(args.show_decisions).to_string(index=False))
    return 0


def cmd_sectors(args: argparse.Namespace) -> int:
    from tradingbot.agents.sector import SectorRotationAgent
    from tradingbot.data.schema import row_asof

    history = _load(args)
    sector_map = _sector_of(args)
    agent = SectorRotationAgent(sector_map=sector_map)
    agent.fit(history)

    rank = row_asof(agent._sector_rank, max(df.index[-1] for df in history.values()))
    spread = row_asof(agent._dispersion, max(df.index[-1] for df in history.values()))
    if rank is None:
        print("Not enough history to rank sectors.")
        return 1

    px = closes(history)
    by_sector: Dict[str, list[str]] = {}
    for symbol in px.columns:
        by_sector.setdefault(sector_map(symbol), []).append(symbol)
    sector_px = pd.DataFrame({s: px[c].mean(axis=1) for s, c in by_sector.items()})
    sector_ret = sector_px / sector_px.shift(agent.lookback) - 1.0

    rows = [
        {"sector": s, "score": round(float(rank[s]), 3),
         f"{agent.lookback}d return": round(float(sector_ret[s].iloc[-1]), 4),
         "members": len([x for x in history if sector_map(x) == s])}
        for s in rank.index
    ]
    table = pd.DataFrame(rows).sort_values("score", ascending=False)
    dispersion = float(spread["spread"])
    print(f"Sector rotation as of {pd.Timestamp(max(df.index[-1] for df in history.values())).date()}")
    print(f"Cross-sector dispersion: {dispersion:.4f} "
          f"({'strong rotation signal' if dispersion > 0.08 else 'weak / crowded'})\n")
    print(table.to_string(index=False))
    return 0


def cmd_paper_step(args: argparse.Namespace) -> int:
    """One full in-process cycle: workers -> consensus -> paper fills -> store."""
    from tradingbot.live import PaperRunConfig, run_in_process
    from tradingbot.notify.telegram import TelegramNotifier
    from tradingbot.paper.broker import NSECostModel, PaperBroker
    from tradingbot.store.db import Store

    history = _load(args)
    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    store = Store(args.db)
    broker = PaperBroker(capital=args.capital, costs=NSECostModel(product=args.product))
    notifier = TelegramNotifier()

    result = run_in_process(
        history=history, store=store, broker=broker, profiles=_profiles(args),
        notifier=notifier,
        config=PaperRunConfig(capital=args.capital, product=args.product,
                              risk=_risk(args), sectors=NSE_SECTORS if args.market == "nse" else None),
        sector_of=_sector_of(args),
    )
    report = result["report"]
    print(f"Session        {result['session']}")
    print(f"Signals        {result['signals']}")
    print(f"Decisions      {report.decisions} ({report.actionable} actionable)")
    print(f"Entries        {len(report.entries)}")
    print(f"Equity         Rs {result['equity']:,.2f}  (cash Rs {broker.cash:,.2f})")
    print(f"Exposure       {broker.exposure(_last_prices(history)):.1%}")
    print(f"Telegram       {'configured' if notifier.configured else 'not configured (alerts logged only)'}")
    print(f"Store          {args.db}")
    if report.rejections:
        print(f"\nRejected {len(report.rejections)} idea(s); first few:")
        for item in report.rejections[:5]:
            print(f"  {item['symbol']:<12} {item['rejection']}")
    print("\nDashboard:  tradingbot dashboard --db " + args.db)
    return 0


def _last_prices(history: History) -> Dict[str, float]:
    from tradingbot.live import latest_prices

    return latest_prices(history)


def cmd_dashboard(args: argparse.Namespace) -> int:
    from tradingbot.live import serve_dashboard

    print(f"Serving dashboard for {args.db} on http://{args.host}:{args.port}")
    serve_dashboard(args.db, host=args.host, port=args.port)
    return 0


def cmd_profiles_dump(args: argparse.Namespace) -> int:
    from tradingbot.agents.registry import dump_profiles

    path = dump_profiles(args.out)
    print(f"Wrote {path}")
    print("Edit it, then pass it back with --profiles on any command.")
    return 0


def cmd_dhan_check(args: argparse.Namespace) -> int:
    """Verify Dhan credentials by fetching one real candle. Nothing is traded."""
    from tradingbot.data.dhan import DhanProvider, resolve_security_ids

    symbol = args.symbol
    try:
        sec_map = resolve_security_ids([symbol], args.scrip_master)
        provider = DhanProvider(security_map=sec_map)
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
        start = (pd.Timestamp.today() - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
        history = provider.load([symbol], start, end)
        frame = history[symbol]
        print(f"OK  {symbol} -> security id {sec_map[symbol.upper()]}")
        print(f"    {len(frame)} candles, last close {frame['close'].iloc[-1]:.2f} "
              f"on {frame.index[-1].date()}")
        return 0
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1


def cmd_telegram_check(args: argparse.Namespace) -> int:
    from tradingbot.notify.telegram import TelegramNotifier

    notifier = TelegramNotifier()
    if not notifier.configured:
        print("FAILED: set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        return 1
    result = notifier.send("&#9989; <b>TradingBot</b> test alert. Paper mode; no orders placed.")
    print("OK  message accepted by Telegram" if result.delivered
          else f"FAILED: {result.error}")
    return 0 if result.delivered else 1


# -- parser -----------------------------------------------------------------
def _common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--market", choices=["nse", "us"], default="nse")
    sp.add_argument("--provider", choices=["synthetic", "csv", "dhan"], default="synthetic")
    sp.add_argument("--csv", type=str, default=None)
    sp.add_argument("--security-map", type=str, default=None)
    sp.add_argument("--scrip-master", type=str, default="dhan_scrip_master.csv")
    sp.add_argument("--profiles", type=str, default=None,
                    help="JSON sector profiles (see profiles-dump)")
    sp.add_argument("--start", type=str, default="2019-01-02")
    sp.add_argument("--end", type=str, default=None)
    sp.add_argument("--sessions", type=int, default=900)
    sp.add_argument("--seed", type=int, default=7)


def _risk_flags(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--capital", type=float, default=100_000.0)
    sp.add_argument("--risk-per-trade", type=float, default=0.01)
    sp.add_argument("--max-position", type=float, default=0.12)
    sp.add_argument("--max-sector", type=float, default=0.30)
    sp.add_argument("--target-vol", type=float, default=0.02,
                    help="target portfolio vol contribution per position")
    sp.add_argument("--min-confidence", type=float, default=0.25)
    sp.add_argument("--entry-threshold", type=float, default=0.20)
    sp.add_argument("--exit-confidence", type=float, default=0.12,
                    help="exit gate; keep below --min-confidence for hysteresis")
    sp.add_argument("--min-holding-days", type=int, default=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tradingbot", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    bt = sub.add_parser("backtest", help="run the strategy over a history")
    _common(bt)
    bt.add_argument("--cash", type=float, default=100_000.0)
    _risk_flags(bt)
    bt.add_argument("--show-decisions", type=int, default=10)
    bt.set_defaults(func=cmd_backtest)

    sc = sub.add_parser("sectors", help="sector rotation leaderboard")
    _common(sc)
    sc.set_defaults(func=cmd_sectors)

    ps = sub.add_parser("paper-step", help="one paper cycle: workers -> consensus -> fills -> store")
    _common(ps)
    _risk_flags(ps)
    ps.add_argument("--db", type=str, default="runs/paper.sqlite3")
    ps.add_argument("--product", choices=["CNC", "INTRADAY"], default="CNC")
    ps.set_defaults(func=cmd_paper_step)

    db = sub.add_parser("dashboard", help="serve the decision dashboard")
    db.add_argument("--db", type=str, default="runs/paper.sqlite3")
    db.add_argument("--host", type=str, default="0.0.0.0")
    db.add_argument("--port", type=int, default=8000)
    db.set_defaults(func=cmd_dashboard)

    pd_ = sub.add_parser("profiles-dump", help="write the default sector profiles to JSON")
    pd_.add_argument("--out", type=str, default="profiles.json")
    pd_.set_defaults(func=cmd_profiles_dump)

    dc = sub.add_parser("dhan-check", help="verify Dhan credentials with one real candle")
    dc.add_argument("--symbol", type=str, default="RELIANCE")
    dc.add_argument("--scrip-master", type=str, default="dhan_scrip_master.csv")
    dc.set_defaults(func=cmd_dhan_check)

    tc = sub.add_parser("telegram-check", help="send a test Telegram alert")
    tc.set_defaults(func=cmd_telegram_check)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "WARNING"),
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
