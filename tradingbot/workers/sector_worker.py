"""Workers: one process per sector, plus one market-wide process.

The main application does not compute indicators. Each sector runs in its own
process, owns only its own symbols, runs its own agent stack, and writes its
signals to SQLite. The coordinator's entire job is to read those rows, apply the
consensus rules, and manage the book -- which is why the main process stays
responsive while twelve sectors are crunching in parallel.

**Why there is a separate market worker.** Two agents are cross-sectional:
``sector_rotation`` ranks sectors against each other, and ``vol_regime`` measures
the volatility of the whole universe. Hand either of them a single sector's
slice and they compute a cross-section of one -- a standard deviation of a
single value, i.e. NaN. So ``SectorWorker`` builds only the per-name agents and
``MarketWorker`` builds the cross-sectional ones over the full universe. Both
write to the same session in the same store.

Two other properties that matter and are both tested:

* **Isolation.** A worker crashing takes its sector down, not the system. The
  coordinator notices a sector has gone quiet (``stale_sectors``) and reports it
  rather than silently trading on stale signals.
* **No shared memory.** Workers get a ``store_path``, never a connection. SQLite
  WAL handles the concurrency; nothing is pickled across process boundaries.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from contextlib import closing
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd

from tradingbot.agents.base import Agent
from tradingbot.agents.registry import SectorProfile, profile_for
from tradingbot.data.schema import History
from tradingbot.markets.nse import NSE_SECTORS, nse_symbols_in_sector
from tradingbot.store.db import Store

log = logging.getLogger(__name__)

#: Agents whose signal is defined *relative to the rest of the universe* and
#: therefore cannot run on a single sector's slice. sector_rotation ranks
#: sectors against each other; vol_regime measures the whole book; momentum
#: z-scores each name against every other name -- inside a four-name sector that
#: degenerates to "best of four", which would call the strongest bank bullish
#: while the entire banking sector falls.
MARKET_WIDE_AGENTS = frozenset({"sector_rotation", "vol_regime", "momentum"})


def _store(path: str):
    """Context manager yielding a Store and closing it."""
    return closing(Store(path))


@dataclass
class SectorAssignment:
    sector: str
    symbols: List[str]
    profile: SectorProfile = field(default_factory=lambda: profile_for("default"))

    @staticmethod
    def from_universe(sector: str, profile: Optional[SectorProfile] = None) -> "SectorAssignment":
        return SectorAssignment(
            sector=sector,
            symbols=nse_symbols_in_sector(sector),
            profile=profile or profile_for(sector),
        )


def _rows(signals, session_date: str, sector: str) -> List[dict]:
    import math

    kept = []
    for signal in signals:
        if not (math.isfinite(signal.score) and math.isfinite(signal.confidence)):
            log.error("dropping non-finite signal from %s for %s (score=%r conf=%r)",
                      signal.agent, signal.symbol, signal.score, signal.confidence)
            continue
        kept.append(signal)
    signals = kept
    return [
        {
            "session_date": session_date,
            "sector": sector,
            "symbol": signal.symbol,
            "agent": signal.agent,
            "role": signal.role,
            "score": signal.score,
            "confidence": signal.confidence,
            "reason": signal.reason,
            "parts": dict(signal.parts),
        }
        for signal in signals
    ]


class SectorWorker:
    """Computes per-name signals for one sector and writes them to the store."""

    def __init__(self, sector: str, symbols: Sequence[str], profile: SectorProfile,
                 store_path: str, sector_map: Optional[Callable[[str], str]] = None) -> None:
        self.sector = sector
        self.symbols = list(symbols)
        self.profile = profile
        self.store_path = store_path
        self._agents: List[Agent] = [
            a for a in profile.build_agents(sector_map)
            if a.name not in MARKET_WIDE_AGENTS
        ]
        self._fitted = False

    def slice_history(self, history: History) -> History:
        """The worker only ever sees its own symbols -- this is the isolation."""
        missing = [s for s in self.symbols if s not in history]
        if missing:
            log.warning("sector %s missing symbols: %s", self.sector, missing)
        return {s: history[s] for s in self.symbols if s in history}

    def run_cycle(self, history: History, session_date: str) -> int:
        """Fit agents on this sector's slice and persist the signals. Returns row count."""
        local = self.slice_history(history)
        if not local:
            return 0

        for agent in self._agents:
            agent.fit(local)

        date = max(df.index[-1] for df in local.values())
        rows: List[dict] = []
        for agent in self._agents:
            for signal in agent.signals_on(date):
                # A per-name agent should only ever speak about names it owns;
                # anything else indicates a mis-scoped agent.
                if signal.symbol not in local:
                    log.warning("agent %s spoke about %s outside sector %s",
                                agent.name, signal.symbol, self.sector)
                    continue
                rows.extend(_rows([signal], session_date, self.sector))

        with _store(self.store_path) as store:
            store.write_signals(rows)
        return len(rows)


class MarketWorker:
    """Computes the cross-sectional signals over the full universe."""

    sector_tag = "__market__"

    def __init__(self, profile: SectorProfile, store_path: str,
                 sector_map: Optional[Callable[[str], str]] = None) -> None:
        self.profile = profile
        self.store_path = store_path
        self._agents: List[Agent] = [
            a for a in profile.build_agents(sector_map) if a.name in MARKET_WIDE_AGENTS
        ]

    def run_cycle(self, history: History, session_date: str) -> int:
        if not history or not self._agents:
            return 0
        for agent in self._agents:
            agent.fit(history)

        date = max(df.index[-1] for df in history.values())
        rows: List[dict] = []
        for agent in self._agents:
            rows.extend(_rows(agent.signals_on(date), session_date, self.sector_tag))

        with _store(self.store_path) as store:
            store.write_signals(rows)
        return len(rows)


def worker_main(sector: str, symbols: Sequence[str], profile: SectorProfile,
                store_path: str, load_history: Callable[[], History],
                poll_seconds: float, stop_event: Optional["mp.synchronize.Event"] = None,
                max_cycles: Optional[int] = None, market: bool = False) -> int:
    """Process entry point. Loops until ``stop_event`` is set or ``max_cycles`` runs."""
    worker: MarketWorker | SectorWorker = (
        MarketWorker(profile, store_path) if market
        else SectorWorker(sector, symbols, profile, store_path)
    )
    cycles = 0
    while stop_event is None or not stop_event.is_set():
        try:
            history = load_history()
            session = pd.Timestamp(max(df.index[-1] for df in history.values())).strftime("%Y-%m-%d")
            written = worker.run_cycle(history, session)
            log.info("worker %s cycle %d wrote %d signals", sector, cycles, written)
        except Exception as exc:  # one worker failing must not take the system down
            log.exception("worker %s cycle failed: %s", sector, exc)
        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            break
        if stop_event is not None:
            stop_event.wait(poll_seconds)
        else:
            time.sleep(poll_seconds)
    return cycles


def spawn_workers(
    store_path: str,
    load_history: Callable[[], History],
    sectors: Optional[Sequence[str]] = None,
    poll_seconds: float = 60.0,
    profiles: Optional[Dict[str, SectorProfile]] = None,
    market_profile: Optional[SectorProfile] = None,
    start: bool = True,
) -> tuple[List[mp.Process], "mp.synchronize.Event"]:
    """Start one process per sector plus the market-wide process."""
    stop = mp.Event()
    procs: List[mp.Process] = []
    profiles = profiles or {}

    for sector in (sectors or NSE_SECTORS):
        assignment = SectorAssignment.from_universe(sector, profiles.get(sector))
        if not assignment.symbols:
            continue
        proc = mp.Process(
            target=worker_main,
            args=(sector, assignment.symbols, assignment.profile, store_path,
                  load_history, poll_seconds, stop),
            kwargs={"market": False},
            name=f"sector-{sector}",
            daemon=True,
        )
        if start:
            proc.start()
        procs.append(proc)

    market = mp.Process(
        target=worker_main,
        args=("__market__", [], market_profile or profile_for("Banking"), store_path,
              load_history, poll_seconds, stop),
        kwargs={"market": True},
        name="market-wide",
        daemon=True,
    )
    if start:
        market.start()
    procs.append(market)
    return procs, stop


def stale_sectors(store_path: str, session_date: str, expected: Sequence[str],
                  stale_after: float = 300.0) -> List[str]:
    """Sectors that should have signalled this session but did not."""
    with _store(store_path) as store:
        rows = store.latest_signals(session_date)
    seen = {r["sector"] for r in rows}
    return [s for s in expected if s not in seen]
