"""Quick event-style backtest of the swing (CNC) logic on real history.

Not a tick-level simulator — it answers one question honestly: "does the
strategy edge survive costs?" It replays the EMA-pullback / 20D-breakout
entry, ATR stops, 2R partial + EMA20 trailing exit, with the same charge
model the paper broker uses.

Usage:
    python scripts/backtest.py --symbols RELIANCE,INFY --years 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from tradingbot import indicators as ta
from tradingbot.broker.paper import estimate_charges
from tradingbot.config import SETTINGS, PROJECT_ROOT
from tradingbot.data.market_data import MarketData
from tradingbot.logger import get_logger

log = get_logger("backtest")


def backtest_symbol(df: pd.DataFrame, symbol: str, capital: float) -> dict:
    close = df["close"]
    e21 = ta.ema(close, 21)
    e50 = ta.ema(close, 50)
    e200 = ta.ema(close, 200)
    atr = ta.atr(df, 14)
    rsi = ta.rsi(close, 14)
    hi20 = close.rolling(20).max()
    vol_avg = df["volume"].rolling(20).mean()

    equity = capital
    trades = []
    pos = None
    start = 260

    for i in range(start, len(df) - 1):
        px = close.iloc[i]

        # ------- manage open position (daily granularity) --------------
        if pos is not None:
            pos["days"] += 1
            exited = False
            if px <= pos["stop"]:
                reason = "STOP"; exited = True
            elif px >= pos["t2"] and pos["partial"]:
                reason = "TARGET2"; exited = True
            elif px < e21.iloc[i] and pos["partial"]:
                reason = "TRAIL"; exited = True
            elif pos["days"] >= 12:
                reason = "TIME_EXIT"; exited = True

            if exited:
                gross = (px - pos["entry"]) * pos["qty"]
                ch = (pos["entry_ch"] * pos["qty"] / pos["qty0"]) + \
                    estimate_charges(symbol, "SELL", pos["qty"], px, "CNC")
                net = gross - ch
                equity += net
                trades.append(net)
                pos = None
            continue

        # ------- entries (long only, CNC) -------------------------------
        if not (e50.iloc[i] > e200.iloc[i] and px > e50.iloc[i]):
            continue
        breakout = px > hi20.iloc[i - 1] and df["volume"].iloc[i] > 1.5 * vol_avg.iloc[i]
        pullback = px > e21.iloc[i] and close.iloc[i - 1] <= e21.iloc[i - 1] and 42 <= rsi.iloc[i] <= 70
        if not (breakout or pullback):
            continue

        stop = px - 2 * atr.iloc[i]
        risk = px - stop
        qty = int(np.floor(capital * 0.01 / risk))
        max_notional = equity * SETTINGS.max_notional_per_stock_pct / 100
        while qty > 0 and qty * px > max_notional:
            qty -= 1
        if qty < 1:
            continue
        entry_ch = estimate_charges(symbol, "BUY", qty, px, "CNC")
        pos = {"entry": px, "stop": stop, "t2": px + 3 * risk, "qty": qty, "qty0": qty,
               "entry_ch": entry_ch, "days": 0, "partial": False}
        # partial at 2R marks trailing arm (simplified: full exit via T2/TRAIL)
        if px + 2 * risk <= df["high"].iloc[i + 1]:
            pos["partial"] = True

    return {"symbol": symbol, "trades": len(trades),
            "net_pnl": round(sum(trades), 0), "equity": round(equity, 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="RELIANCE,INFY,TCS,SBIN,TATAMOTORS,HDFCBANK")
    ap.add_argument("--capital", type=float, default=SETTINGS.capital_inr)
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    md = MarketData(symbols, mode=SETTINGS.resolved_data_mode())

    total_trades, total_pnl = 0, 0.0
    print(f"{'symbol':<12} {'trades':>6} {'net ₹':>12}")
    for s in symbols:
        df = md.daily(s, 1200)
        r = backtest_symbol(df, s, args.capital)
        total_trades += r["trades"]
        total_pnl += r["net_pnl"]
        print(f"{r['symbol']:<12} {r['trades']:>6} {r['net_pnl']:>12,.0f}")
    print("-" * 32)
    print(f"{'TOTAL':<12} {total_trades:>6} {total_pnl:>12,.0f}")
    print(f"capital {args.capital:,.0f} → {args.capital + total_pnl:,.0f} "
          f"({total_pnl / args.capital * 100:+.1f}% over sample)")


if __name__ == "__main__":
    main()
