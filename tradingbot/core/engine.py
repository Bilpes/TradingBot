"""The orchestrator: data -> agents -> consensus -> risk -> portfolio.

Timing discipline (this is where most backtests quietly lie):

* Agents are fitted once over the full history, but every indicator they use is
  rolling/backward-looking, and reads go through ``row_asof(date)`` which is a
  ``<=`` lookup. ``tests/test_no_lookahead.py`` re-fits on truncated data and
  asserts identical answers.
* A decision is made on the close of session *t* and executed at the open of
  session *t+1*. Nothing ever trades on the bar that produced its own signal.
* Stops are evaluated against the session's low and filled at the stop price,
  or at the open when the market gaps through it -- never at a better price
  than the market actually offered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from tradingbot.agents import build_default_agents
from tradingbot.agents.base import Agent, Signal
from tradingbot.core.confidence import Consensus, aggregate
from tradingbot.core.metrics import performance_report
from tradingbot.core.portfolio import Fill, Order, Portfolio, Side
from tradingbot.core.risk import RiskConfig, atr, realised_vol, size_position
from tradingbot.data.schema import History, validate_history
from tradingbot.data.universe import sector_of


@dataclass
class EngineConfig:
    risk: RiskConfig = field(default_factory=RiskConfig)
    tau: float = 0.5              # dispersion scale in the consensus layer
    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    initial_cash: float = 100_000.0
    trailing_stop: bool = True
    log_decisions: bool = True


@dataclass
class BacktestResult:
    equity: pd.Series
    exposure: pd.Series
    fills: List[Fill]
    trades: List[dict]
    decisions: pd.DataFrame
    stats: Dict[str, float]

    def summary(self) -> str:
        from tradingbot.core.metrics import format_report

        return format_report(self.stats)


class Engine:
    """Runs a set of agents over a history and produces a traded equity curve."""

    def __init__(
        self,
        agents: Optional[Sequence[Agent]] = None,
        config: Optional[EngineConfig] = None,
        sector_map: Callable[[str], str] = sector_of,
    ) -> None:
        self.agents: List[Agent] = list(agents) if agents is not None else build_default_agents()
        if not self.agents:
            raise ValueError("Engine needs at least one agent")
        self.config = config or EngineConfig()
        self.sector_map = sector_map
        self.agent_weights = {a.name: a.weight for a in self.agents}

    @property
    def warmup(self) -> int:
        return max(a.min_history for a in self.agents)

    # -- internals ----------------------------------------------------------
    def _wide(self, hist: History, column: str) -> pd.DataFrame:
        return pd.DataFrame({s: df[column] for s, df in hist.items()}).sort_index()

    def consensus_on(self, date: pd.Timestamp, hist: Optional[History] = None) -> Dict[str, Consensus]:
        """Public helper: consensus for one session (fits agents if needed)."""
        if hist is not None:
            for agent in self.agents:
                agent.fit(hist)
        return self._signals_to_consensus([s for a in self.agents for s in a.signals_on(date)])

    def _signals_to_consensus(self, signals: Sequence[Signal]) -> Dict[str, Consensus]:
        return aggregate(signals, self.agent_weights, self.config.tau)

    # -- main loop ----------------------------------------------------------
    def run(self, history: History) -> BacktestResult:
        cfg = self.config
        risk = cfg.risk
        hist = validate_history(history)

        opens = self._wide(hist, "open")
        lows = self._wide(hist, "low")
        px_close = self._wide(hist, "close")
        atr_df = pd.DataFrame({s: atr(df, risk.atr_window) for s, df in hist.items()}).sort_index()
        vol_df = pd.DataFrame(
            {s: realised_vol(df["close"], risk.vol_lookback) for s, df in hist.items()}
        ).sort_index()

        for agent in self.agents:
            agent.fit(hist)

        portfolio = Portfolio(
            cash=cfg.initial_cash,
            commission_bps=cfg.commission_bps,
            slippage_bps=cfg.slippage_bps,
            sector_of=self.sector_map,
        )

        dates = px_close.index
        pending: List[Order] = []
        decisions: List[dict] = []
        exposure_pts: Dict[pd.Timestamp, float] = {}
        entry_idx: Dict[str, int] = {}   # symbol -> index of the session it was filled
        start_i = min(self.warmup, len(dates) - 1)

        for i, date in enumerate(dates):
            o_row = opens.loc[date]
            l_row = lows.loc[date]
            c_row = px_close.loc[date]

            def px(row: pd.Series, symbol: str) -> float:
                v = row.get(symbol, np.nan)
                return float(v) if v is not None and np.isfinite(v) else float("nan")

            # 1) Fill yesterday's decisions at today's open.
            for order in pending:
                price = px(o_row, order.symbol)
                if not np.isfinite(price):
                    continue  # no print today -> the idea is stale, drop it
                fill = portfolio.execute(order, price, date)
                if fill is not None:
                    if fill.side is Side.BUY:
                        entry_idx[fill.symbol] = i
                    if cfg.log_decisions:
                        decisions.append(
                            {
                                "date": date, "symbol": order.symbol, "action": fill.side.value,
                                "stage": "fill", "shares": fill.shares, "price": fill.price,
                                "reason": fill.reason,
                            }
                        )
            pending = []
            for symbol in list(entry_idx):
                if symbol not in portfolio.positions:
                    entry_idx.pop(symbol, None)

            # 2) Intraday stops.
            for symbol in list(portfolio.positions):
                stop = portfolio.stops.get(symbol)
                if stop is None:
                    continue
                lo, op = px(l_row, symbol), px(o_row, symbol)
                if not np.isfinite(lo) or lo > stop:
                    continue
                exit_price = min(op, stop) if np.isfinite(op) else stop
                qty = portfolio.positions[symbol]
                fill = portfolio.execute(Order(symbol, Side.SELL, qty, "stop"), exit_price, date)
                if fill is not None and cfg.log_decisions:
                    decisions.append(
                        {"date": date, "symbol": symbol, "action": "sell", "stage": "stop",
                         "shares": fill.shares, "price": fill.price, "reason": f"stop {stop:.2f}"}
                    )

            # 3) Decide on today's close; execute tomorrow.
            close_prices = {s: px(c_row, s) for s in c_row.index}
            close_prices = {s: p for s, p in close_prices.items() if np.isfinite(p)}

            if i >= start_i and close_prices:
                consensus = self._signals_to_consensus(
                    [s for a in self.agents for s in a.signals_on(date)]
                )

                # 3a) Exits: conviction lost, or the vote flipped against us.
                #     The exit gate is lower than the entry gate (hysteresis) and
                #     a fresh position is given min_holding_days before a signal
                #     exit is allowed -- otherwise the book churns on noise.
                for symbol in list(portfolio.positions):
                    cons = consensus.get(symbol)
                    lost = (
                        cons is None
                        or cons.score < risk.exit_threshold
                        or cons.confidence < risk.exit_confidence
                    )
                    held = i - entry_idx.get(symbol, i)
                    if not lost or held < risk.min_holding_days:
                        continue
                    pending.append(
                        Order(symbol, Side.SELL, portfolio.positions[symbol],
                              "conviction lost", date)
                    )
                    if cfg.log_decisions:
                        decisions.append(
                            {"date": date, "symbol": symbol, "action": "exit_signal",
                             "stage": "decide", "shares": portfolio.positions[symbol],
                             "price": close_prices.get(symbol, float("nan")),
                             "reason": "no consensus" if cons is None
                                       else f"score {cons.score:+.2f} conf {cons.confidence:.2f}"}
                        )

                # 3b) Ratchet trailing stops on what we keep.
                if cfg.trailing_stop:
                    for symbol in portfolio.positions:
                        a = px(atr_df.loc[date], symbol) if symbol in atr_df.columns else float("nan")
                        p = close_prices.get(symbol, float("nan"))
                        if not np.isfinite(a) or not np.isfinite(p):
                            continue
                        candidate = p - risk.stop_atr_mult * a
                        if candidate > portfolio.stops.get(symbol, -np.inf):
                            portfolio.stops[symbol] = candidate

                # 3c) Entries, best conviction first.
                candidates = sorted(
                    (
                        (c.confidence * abs(c.score), sym, c)
                        for sym, c in consensus.items()
                        if sym != "*"
                        and sym not in portfolio.positions
                        and c.score >= risk.entry_threshold
                        and c.confidence >= risk.min_confidence
                    ),
                    key=lambda t: t[0],
                    reverse=True,
                )

                # Slots are counted conservatively: the exits queued in 3a only
                # free capacity at tomorrow's open, and that fill is not
                # guaranteed (a name with no print is dropped). Crediting them
                # here would let the book exceed max_positions whenever an exit
                # failed to fill -- a limit that only holds on average is not a
                # limit, so queued exits are not counted as freed slots.
                open_slots = risk.max_positions - len(portfolio.positions)

                for _rank, symbol, cons in candidates:
                    if open_slots <= 0:
                        break
                    price = close_prices.get(symbol, float("nan"))
                    if not np.isfinite(price):
                        continue

                    equity = portfolio.equity(close_prices)
                    if portfolio.gross_exposure(close_prices) >= risk.max_gross_exposure:
                        break
                    sector = self.sector_map(symbol)
                    if portfolio.sector_weights(close_prices).get(sector, 0.0) >= risk.max_sector_weight:
                        continue

                    a = px(atr_df.loc[date], symbol) if symbol in atr_df.columns else float("nan")
                    v = px(vol_df.loc[date], symbol) if symbol in vol_df.columns else float("nan")
                    sizing = size_position(equity, price, a, cons.confidence, v, risk)
                    if sizing.shares <= 0:
                        if cfg.log_decisions:
                            decisions.append(
                                {"date": date, "symbol": symbol, "action": "rejected",
                                 "stage": "risk", "shares": 0, "price": price,
                                 "reason": f"{sizing.binding}: {sizing.reason}"}
                            )
                        continue

                    pending.append(
                        Order(symbol, Side.BUY, sizing.shares,
                              f"consensus {cons.score:+.2f} @ conf {cons.confidence:.2f}",
                              date, sizing.stop)
                    )
                    open_slots -= 1
                    if cfg.log_decisions:
                        decisions.append(
                            {"date": date, "symbol": symbol, "action": "entry_signal",
                             "stage": "decide", "shares": sizing.shares, "price": price,
                             "reason": f"{sizing.binding} conf {cons.confidence:.2f} "
                                       f"agree {cons.agreement:.2f} votes {cons.n_votes}"}
                        )

            # 4) Mark to market.
            portfolio.mark(date, close_prices)
            exposure_pts[date] = portfolio.gross_exposure(close_prices)

        equity = portfolio.equity_curve
        trades = portfolio.round_trips()
        exposure = pd.Series(exposure_pts, dtype="float64")
        stats = performance_report(equity, trades, exposure)

        return BacktestResult(
            equity=equity,
            exposure=exposure,
            fills=portfolio.fills,
            trades=trades,
            decisions=pd.DataFrame(decisions),
            stats=stats,
        )
