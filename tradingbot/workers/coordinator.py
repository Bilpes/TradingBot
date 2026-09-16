"""Coordinator: the thin main application.

It does no indicator math. Its whole job, per cycle:

1. Read the signals the sector workers wrote to SQLite.
2. Fold them into a consensus per symbol and apply the quorum rules.
3. Ask the risk layer how big a position that consensus deserves.
4. Fill against the paper broker, applying real NSE charges.
5. Persist the decision with its full reasoning and send a Telegram alert.

Everything expensive happens in a worker process; this loop is a few
milliseconds of arithmetic, which is the design goal.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

import pandas as pd

from tradingbot.agents.base import Signal
from tradingbot.core.consensus import ConsensusConfig, ConsensusDecision, decide_all, explain
from tradingbot.core.risk import RiskConfig, size_position
from tradingbot.markets.nse import nse_sector_of
from tradingbot.notify.telegram import (
    TelegramNotifier,
    entry_alert,
    exit_alert,
    risk_alert,
)
from tradingbot.paper.broker import PaperBroker
from tradingbot.store.db import Store

log = logging.getLogger(__name__)


@dataclass
class CycleReport:
    session_date: str
    decisions: int = 0
    actionable: int = 0
    entries: List[dict] = field(default_factory=list)
    exits: List[dict] = field(default_factory=list)
    rejections: List[dict] = field(default_factory=list)


class Coordinator:
    def __init__(
        self,
        store: Store,
        broker: PaperBroker,
        risk: RiskConfig | None = None,
        consensus: ConsensusConfig | None = None,
        notifier: TelegramNotifier | None = None,
        sector_of: Callable[[str], str] = nse_sector_of,
        horizon_days: int = 5,
        atr_lookup: Optional[Callable[[str], float]] = None,
        vol_lookup: Optional[Callable[[str], float]] = None,
        trailing_stop: bool = True,
    ) -> None:
        self.store = store
        self.broker = broker
        self.risk = risk or RiskConfig()
        self.consensus = consensus or ConsensusConfig()
        self.notifier = notifier or TelegramNotifier()
        self.sector_of = sector_of
        self.horizon_days = horizon_days
        self.atr_lookup = atr_lookup or (lambda symbol: float("nan"))
        self.vol_lookup = vol_lookup or (lambda symbol: float("nan"))
        self._session_index: dict[str, int] = {}
        self._held_since: dict[str, int] = {}
        self.trailing_stop = trailing_stop

    # -- signal reconstruction ---------------------------------------------
    def _signals_from_rows(self, rows: Sequence) -> List[Signal]:
        out: List[Signal] = []
        for row in rows:
            try:
                out.append(
                    Signal(
                        symbol=row["symbol"],
                        score=float(row["score"]),
                        confidence=float(row["confidence"]),
                        agent=row["agent"],
                        role=row["role"],
                        reason=row["reason"] or "",
                        parts=json.loads(row["parts_json"]) if row["parts_json"] else {},
                    )
                )
            except (ValueError, KeyError, TypeError) as exc:
                # A malformed row from one worker must not poison the cycle.
                log.warning("skipping bad signal row id=%s: %s", row["id"], exc)
        return out

    def _agent_weights(self) -> Dict[str, float]:
        """Weights come from the sector profiles, flattened by agent name."""
        from tradingbot.agents.registry import DEFAULT_PROFILES

        weights: Dict[str, float] = {}
        for profile in DEFAULT_PROFILES.values():
            for spec in profile.agents:
                weights[spec.agent] = max(weights.get(spec.agent, 0.0), spec.weight)
        return weights

    # -- main cycle ---------------------------------------------------------
    def run_cycle(self, session_date: str, prices: Dict[str, float],
                  atr_values: Optional[Dict[str, float]] = None,
                  vol_values: Optional[Dict[str, float]] = None) -> CycleReport:
        report = CycleReport(session_date=session_date)
        tick = len(self._session_index)
        self._session_index.setdefault(session_date, tick)
        tick = self._session_index[session_date]

        rows = self.store.latest_signals(session_date)
        if not rows:
            log.warning("no signals for session %s -- standing down", session_date)
            self._alert(risk_alert("no signals", f"no worker output for session {session_date}"),
                        "risk", None)
            return report

        atr_values = atr_values or {}
        vol_values = vol_values or {}

        # 1) Exits first: risk reduction never waits behind new ideas.
        for symbol, position in list(self.broker.positions.items()):
            price = prices.get(symbol)
            if price is None:
                continue
            reason = None
            if position.stop is not None and price <= position.stop:
                reason = f"stop hit ({price:,.2f} <= {position.stop:,.2f})"
            if reason:
                fill = self.broker.sell(symbol, price, pd.Timestamp.now(tz='UTC').isoformat(), reason)
                if fill:
                    self.store.write_fill(fill)
                    report.exits.append(fill)
                    self._alert(exit_alert(symbol, fill["quantity"], fill["price"],
                                           fill["pnl"], reason), "exit", symbol)

        # 2) Consensus + quorum.
        decisions = decide_all(self._signals_from_rows(rows), self._agent_weights(),
                               self.consensus)

        # 3) Signal-driven exits.
        #
        #    Two rules the original version was missing, both of which caused a
        #    851-trade, 19.6%-of-capital-in-charges churn:
        #      * A symbol that emits no signals this session is *silent*, not
        #        bearish. Agents stay quiet when conviction is low, so treating
        #        silence as an exit signal closed positions on noise.
        #      * The exit gate sits below the entry gate (hysteresis) and a
        #        position gets min_holding_days before a signal exit is allowed.
        #    Stops bypass both -- risk exits are never delayed by a hold timer.
        for symbol, position in list(self.broker.positions.items()):
            decision = decisions.get(symbol)
            price = prices.get(symbol)
            if price is None or decision is None:
                continue
            held = tick - self._held_since.get(symbol, tick)
            if held < self.risk.min_holding_days:
                continue
            lost = (decision.score < self.risk.exit_threshold
                    or decision.confidence < self.risk.exit_confidence)
            if lost:
                reason = (f"conviction lost (score {decision.score:+.2f}, "
                          f"conf {decision.confidence:.2f}) after {held} sessions")
                fill = self.broker.sell(symbol, price, pd.Timestamp.now(tz='UTC').isoformat(), reason)
                if fill:
                    self.store.write_fill(fill)
                    report.exits.append(fill)
                    self._alert(exit_alert(symbol, fill["quantity"], fill["price"],
                                           fill["pnl"], reason), "exit", symbol)

        # 4) Entries, best conviction first.
        open_slots = self.risk.max_positions - len(self.broker.positions)
        ranked = sorted(decisions.values(), key=lambda d: (d.confidence, abs(d.score)),
                        reverse=True)

        for decision in ranked:
            self._persist(decision, report, prices.get(decision.symbol))
            if not decision.actionable:
                continue
            if open_slots <= 0:
                self.store.write_decision(self._decision_row(decision, report,
                                                             action="skipped",
                                                             reason="max_positions reached"))
                continue

            # The quorum rules are direction-agnostic: a strongly agreed SHORT
            # is just as "actionable" as a strongly agreed LONG. This book is
            # long-only, so a negative score must never become a buy. Skipping
            # this check produced a 0% win rate -- the system was reliably
            # buying exactly what its own agents expected to fall.
            if decision.score < self.risk.entry_threshold:
                self.store.write_decision(self._decision_row(
                    decision, report, action="no long entry",
                    reason=f"long-only book: score {decision.score:+.2f} is below the "
                           f"{self.risk.entry_threshold:.2f} entry threshold"))
                continue

            symbol = decision.symbol
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue
            if self.broker.exposure(prices) >= self.risk.max_gross_exposure:
                break
            sector = self.sector_of(symbol)
            if self.broker.sector_exposure(prices).get(sector, 0.0) >= self.risk.max_sector_weight:
                self.store.write_decision(self._decision_row(
                    decision, report, action="rejected", reason="sector limit reached"))
                continue

            sizing = size_position(
                equity=self.broker.equity(prices),
                price=price,
                atr_value=atr_values.get(symbol, float("nan")),
                confidence=decision.confidence,
                vol=vol_values.get(symbol, float("nan")),
                cfg=self.risk,
            )
            if sizing.shares <= 0:
                self.store.write_decision(self._decision_row(
                    decision, report, action="rejected",
                    reason=f"sizing: {sizing.binding} ({sizing.reason})"))
                continue

            ts = pd.Timestamp.now(tz='UTC').isoformat()
            fill = self.broker.buy(symbol, sector, sizing.shares, price, ts,
                                   stop=sizing.stop, reason=explain(decision))
            if not fill:
                continue
            self.store.write_fill(fill)
            self.store.write_order({"symbol": symbol, "side": "BUY", "quantity": sizing.shares,
                                    "price": price, "status": "filled", "broker_ref": "paper",
                                    "reason": explain(decision)})
            open_slots -= 1
            self._held_since[symbol] = tick
            report.entries.append(fill)
            self._alert(entry_alert(symbol, sector, sizing.shares, fill["price"],
                                    decision.confidence, decision.score, decision.votes,
                                    sizing.stop), "entry", symbol)

        # 5) Ratchet trailing stops.
        #
        #    Without this the book cannot book a win: the entry gate needs a
        #    score of +0.20 while the signal exit needs it below 0, so the
        #    static stop (2.5 ATR, ~7%) is always the first thing to trigger.
        #    Every one of 22 simulated trades exited on the stop for exactly
        #    this reason. A trailing stop lets a winner leave at a profit
        #    instead of round-tripping back through its entry.
        if self.trailing_stop:
            for symbol, position in self.broker.positions.items():
                price = prices.get(symbol)
                atr_value = atr_values.get(symbol, float("nan"))
                if price is None or not np.isfinite(atr_value) or atr_value <= 0:
                    continue
                candidate = price - self.risk.stop_atr_mult * atr_value
                if position.stop is None or candidate > position.stop:
                    position.stop = candidate

        for symbol in list(self._held_since):
            if symbol not in self.broker.positions:
                self._held_since.pop(symbol, None)

        self._mark(report, prices)
        return report

    # -- helpers ------------------------------------------------------------
    def _decision_row(self, decision: ConsensusDecision, report: CycleReport,
                      action: Optional[str] = None, quantity: Optional[int] = None,
                      price: Optional[float] = None, reason: str = "") -> dict:
        return {
            "session_date": report.session_date,
            "symbol": decision.symbol,
            "sector": self.sector_of(decision.symbol),
            "score": decision.score,
            "confidence": decision.confidence,
            "agreement": decision.agreement,
            "dispersion": decision.dispersion,
            "agreeing": decision.agreeing,
            "dissenting": decision.dissenting,
            "neutral": decision.neutral,
            "vetoed": decision.vetoed,
            "veto_reason": decision.veto_reason,
            "actionable": decision.actionable,
            "rejection": decision.rejection,
            "direction": decision.direction,
            "votes": decision.votes,
            "action": action,
            "quantity": quantity,
            "price": price,
            "reason": reason,
        }

    def _persist(self, decision: ConsensusDecision, report: CycleReport,
                 price: Optional[float] = None) -> None:
        report.decisions += 1
        if decision.actionable:
            report.actionable += 1
        else:
            report.rejections.append({"symbol": decision.symbol,
                                      "rejection": decision.rejection})
        self.store.write_decision(self._decision_row(decision, report, price=price))
        if decision.actionable and price:
            # The prediction is recorded against the price it was made at.
            # Without a real entry price the calibration ledger would divide by
            # zero and every confidence claim would go unmeasured.
            self.store.write_prediction({
                "session_date": report.session_date,
                "symbol": decision.symbol,
                "direction": decision.direction,
                "confidence": decision.confidence,
                "horizon_days": self.horizon_days,
                "entry_price": price,
            })

    def _mark(self, report: CycleReport, prices: Dict[str, float]) -> None:
        ts = pd.Timestamp.now(tz='UTC').isoformat()
        self.store.snapshot_positions(ts, self.broker.snapshot(prices, ts))
        self.store.write_equity(
            ts=ts,
            cash=self.broker.cash,
            invested=self.broker.invested(prices),
            total=self.broker.equity(prices),
            drawdown=self.broker.drawdown(prices),
            exposure=self.broker.exposure(prices),
        )
        self.store.settle_predictions(prices)

    def _alert(self, text: str, kind: str, symbol: Optional[str]) -> None:
        def on_result(result) -> None:
            self.store.write_alert("telegram", kind, text, symbol,
                                   delivered=result.delivered, error=result.error or None)

        self.notifier.send(text, kind=kind, symbol=symbol, on_result=on_result)
