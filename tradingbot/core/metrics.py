"""Performance statistics.

Every figure here is net of commissions and slippage, and drawdown/Sharpe are
computed on the *marked* equity curve, not on realised trade P&L. Gross numbers
are not reported anywhere: they are the easiest way to make a bad strategy look
investable.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def performance_report(
    equity: pd.Series,
    trades: Optional[List[dict]] = None,
    exposure: Optional[pd.Series] = None,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """Standard risk/return summary of a backtest."""
    out: Dict[str, float] = {}
    if equity is None or len(equity) < 2:
        return {"n_periods": 0.0}

    rets = equity.pct_change().dropna()
    n = len(equity)
    years = n / periods_per_year

    out["n_periods"] = float(n)
    out["start"] = float(equity.iloc[0])
    out["end"] = float(equity.iloc[-1])
    out["total_return"] = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    out["cagr"] = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    ann_vol = float(rets.std(ddof=1) * np.sqrt(periods_per_year)) if len(rets) > 1 else 0.0
    out["ann_vol"] = ann_vol
    mean_r = float(rets.mean())
    out["sharpe"] = float(mean_r / rets.std(ddof=1) * np.sqrt(periods_per_year)) if rets.std(ddof=1) > 0 else 0.0

    downside = rets[rets < 0]
    dd_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    out["sortino"] = float(mean_r / dd_std * np.sqrt(periods_per_year)) if dd_std > 0 else float("nan")

    mdd = max_drawdown(equity)
    out["max_drawdown"] = mdd
    out["calmar"] = float(out["cagr"] / abs(mdd)) if mdd < 0 else float("nan")

    if exposure is not None and len(exposure):
        out["avg_exposure"] = float(exposure.mean())
        out["max_exposure"] = float(exposure.max())

    trades = trades or []
    out["n_trades"] = float(len(trades))
    if trades:
        pnls = np.array([t["pnl"] for t in trades], dtype="float64")
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        out["win_rate"] = float(len(wins) / len(pnls))
        out["avg_win"] = float(wins.mean()) if len(wins) else 0.0
        out["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
        gross_win = float(wins.sum())
        gross_loss = float(-losses.sum())
        out["profit_factor"] = float(gross_win / gross_loss) if gross_loss > 0 else float("inf")
        out["expectancy"] = float(pnls.mean())
        out["total_pnl"] = float(pnls.sum())
    return out


_PCT = ("total_return", "cagr", "ann_vol", "max_drawdown", "win_rate", "avg_exposure", "max_exposure")


def format_report(stats: Dict[str, float]) -> str:
    """Human-readable block for the CLI."""
    if not stats or stats.get("n_periods", 0) == 0:
        return "No results: equity curve is empty."

    lines = [
        "=" * 56,
        f"  Periods            {stats['n_periods']:.0f}",
        f"  Start equity       {stats['start']:>14,.2f}",
        f"  End equity         {stats['end']:>14,.2f}",
        f"  Total return       {stats['total_return']:>13.2%}",
        f"  CAGR               {stats['cagr']:>13.2%}",
        f"  Annualised vol     {stats['ann_vol']:>13.2%}",
        f"  Sharpe             {stats['sharpe']:>13.2f}",
        f"  Sortino            {stats['sortino']:>13.2f}",
        f"  Max drawdown       {stats['max_drawdown']:>13.2%}",
        f"  Calmar             {stats['calmar']:>13.2f}",
    ]
    if "avg_exposure" in stats:
        lines.append(f"  Avg / max exposure {stats['avg_exposure']:>7.1%} /{stats['max_exposure']:.1%}")
    if stats.get("n_trades"):
        lines += [
            "-" * 56,
            f"  Closed trades      {stats['n_trades']:>13.0f}",
            f"  Win rate           {stats['win_rate']:>13.2%}",
            f"  Profit factor      {stats['profit_factor']:>13.2f}",
            f"  Avg win / loss     {stats['avg_win']:>10,.0f} /{stats['avg_loss']:,.0f}",
            f"  Expectancy / trade {stats['expectancy']:>13,.2f}",
        ]
    lines.append("=" * 56)
    return "\n".join(lines)
