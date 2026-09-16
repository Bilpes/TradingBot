"""Multi-seed sanity scan.

Run this before believing any single backtest number. One seed is an anecdote;
the distribution across seeds is the only thing that tells you whether a result
is a property of the strategy or a property of the random path you happened to
draw.

    python scripts/seed_scan.py --seeds 1 2 3 7 11 23 42 99 --sessions 900

Note the trap this exists to catch: seed 7 is the CLI default and is typically
among the worst draws. Tuning parameters until the default seed looks good is
overfitting to a single random path, not research.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

sys.path.insert(0, ".")

from tradingbot.core.engine import Engine, EngineConfig
from tradingbot.core.risk import RiskConfig
from tradingbot.data.providers import SyntheticProvider


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 7, 11, 23, 42, 99])
    p.add_argument("--sessions", type=int, default=900)
    p.add_argument("--start", type=str, default="2019-01-02")
    args = p.parse_args()

    rows = []
    t0 = time.time()
    for seed in args.seeds:
        hist = SyntheticProvider(sessions=args.sessions, seed=seed).load([], args.start)
        stats = Engine(config=EngineConfig(risk=RiskConfig())).run(hist).stats
        rows.append(
            (
                seed,
                100 * stats["total_return"],
                stats["sharpe"],
                100 * stats["max_drawdown"],
                int(stats["n_trades"]),
                100 * stats["win_rate"],
                100 * stats["avg_exposure"],
            )
        )

    print("seed   totRet   sharpe    maxDD  trades  winRate  avgExp")
    for r in rows:
        print("%4d  %6.2f%%  %6.2f  %7.2f%%  %6d  %6.1f%%  %5.1f%%" % r)

    rets = [r[1] for r in rows]
    sharpes = [r[2] for r in rows]
    print()
    print("mean return %+.2f%%  median %+.2f%%  best %+.2f%%  worst %+.2f%%"
          % (statistics.mean(rets), statistics.median(rets), max(rets), min(rets)))
    print("mean sharpe %+.2f  |  profitable seeds %d/%d  |  %.1fs"
          % (statistics.mean(sharpes), sum(1 for r in rets if r > 0), len(rets), time.time() - t0))
    print("\nSynthetic data: this measures the machinery, not an edge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
