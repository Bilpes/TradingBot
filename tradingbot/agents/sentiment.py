"""Earnings / news sentiment agent.

**Read this before trusting it.** There is no live news or sentiment feed
reachable from this environment (``nseindia.com`` and every vendor API return
connection failures), so this agent ships with the *plumbing* correct and the
*data* pluggable. It is honest about which parts are real:

``SentimentSource``
    Protocol: ``score(symbol, date) -> float | None`` in [-1, 1]. Bring your own
    provider -- news API, broker research feed, or a hand-maintained file.
``StaticSentimentSource``
    Reads a CSV/JSON you maintain. Works today, zero network.
``EarningsCalendar``
    Real and genuinely useful on its own: it applies a **confidence haircut**
    around known earnings dates. Event risk is not directional information, and
    sizing a full position into an earnings print is how a system with a good
    average return still blows up. This needs no sentiment data at all, only
    dates.

What is *not* implemented, and should not be pretended otherwise: LLM-scored
news, social sentiment, or any "earnings sentiment" derived from nothing. An
agent that reports sentiment it never observed is worse than no agent, because
its output looks like evidence.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Protocol, Sequence

import numpy as np
import pandas as pd

from tradingbot.agents.base import Agent, Signal, clip
from tradingbot.data.schema import History


class SentimentSource(Protocol):
    def score(self, symbol: str, date: pd.Timestamp) -> Optional[float]:
        """Return a sentiment score in [-1, 1], or ``None`` for "no opinion"."""
        ...


class StaticSentimentSource:
    """Sentiment from a file you maintain.

    CSV columns: ``symbol, date, score, note``. Dates are inclusive-forward:
    a score applies from its date until the next entry for that symbol, then
    decays to zero over ``decay_days``.
    """

    def __init__(self, path: str | Path, decay_days: int = 20) -> None:
        self.decay_days = max(decay_days, 1)
        self._by_symbol: Dict[str, pd.DataFrame] = {}
        fp = Path(path)
        if not fp.exists():
            raise FileNotFoundError(fp)

        if fp.suffix.lower() == ".json":
            raw = json.loads(fp.read_text())
            rows = [{"symbol": r["symbol"], "date": r["date"], "score": float(r["score"]),
                     "note": r.get("note", "")} for r in raw]
        else:
            with fp.open(newline="") as fh:
                rows = [
                    {"symbol": r["symbol"].upper(), "date": r["date"],
                     "score": float(r["score"]), "note": r.get("note", "")}
                    for r in csv.DictReader(fh)
                ]

        frame = pd.DataFrame(rows)
        if frame.empty:
            return
        frame["date"] = pd.to_datetime(frame["date"])
        frame["score"] = frame["score"].clip(-1.0, 1.0)
        for symbol, group in frame.groupby("symbol"):
            self._by_symbol[str(symbol).upper()] = group.sort_values("date").reset_index(drop=True)

    def score(self, symbol: str, date: pd.Timestamp) -> Optional[float]:
        frame = self._by_symbol.get(symbol.upper())
        if frame is None or frame.empty:
            return None
        prior = frame[frame["date"] <= date]
        if prior.empty:
            return None
        last = prior.iloc[-1]
        age = (date - last["date"]).days
        if age > self.decay_days:
            return None
        decay = 1.0 - (age / self.decay_days)
        return float(last["score"]) * decay


class EarningsCalendar:
    """Known earnings dates -> a confidence haircut, not a direction.

    ``window_days`` before and after a print, conviction is scaled by
    ``min_multiplier`` at the event date, ramping back to 1.0 at the edge of the
    window. Position size falls into known events instead of riding them.
    """

    def __init__(self, dates: Mapping[str, Sequence[str]], window_days: int = 3,
                 min_multiplier: float = 0.35) -> None:
        if not 0.0 < min_multiplier <= 1.0:
            raise ValueError("min_multiplier must be in (0, 1]")
        self.window_days = max(window_days, 1)
        self.min_multiplier = min_multiplier
        self._dates = {k.upper(): [pd.Timestamp(d) for d in v] for k, v in dates.items()}

    def multiplier(self, symbol: str, date: pd.Timestamp) -> float:
        for event in self._dates.get(symbol.upper(), []):
            distance = abs((date - event).days)
            if distance <= self.window_days:
                ramp = distance / self.window_days
                return self.min_multiplier + (1.0 - self.min_multiplier) * ramp
        return 1.0


class SentimentAgent(Agent):
    name = "sentiment"
    weight = 0.8
    min_history = 5

    def __init__(
        self,
        source: Optional[SentimentSource] = None,
        calendar: Optional[EarningsCalendar] = None,
    ) -> None:
        self.source = source
        self.calendar = calendar
        self._history: History | None = None

    def fit(self, history: History) -> None:
        self._history = history

    def signals_on(self, date: pd.Timestamp) -> Sequence[Signal]:
        if self._history is None:
            raise RuntimeError("fit() must be called before signals_on()")
        if self.source is None and self.calendar is None:
            return []  # nothing observed -> nothing claimed

        out = []
        for symbol in self._history:
            raw = self.source.score(symbol, date) if self.source is not None else None
            mult = self.calendar.multiplier(symbol, date) if self.calendar is not None else 1.0

            if raw is None:
                # No sentiment observed. Still emit an event-risk *gate-style*
                # haircut as a directional vote of zero so the calendar can
                # shrink size on names about to report.
                if mult < 1.0:
                    out.append(
                        Signal(symbol=symbol, score=0.0, confidence=1.0 - mult,
                               agent=self.name, role="gate",
                               reason=f"earnings within window (x{mult:.2f} conviction)")
                    )
                continue

            score = clip(float(raw))
            confidence = float(np.clip(abs(score), 0.0, 0.85)) * mult
            if confidence < 0.05:
                continue
            out.append(
                Signal(
                    symbol=symbol,
                    score=score,
                    confidence=confidence,
                    agent=self.name,
                    reason=f"sentiment {score:+.2f}" + (f", earnings haircut x{mult:.2f}" if mult < 1.0 else ""),
                    parts={"sentiment": score, "event_multiplier": mult},
                )
            )
        return out
