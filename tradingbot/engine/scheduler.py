"""Session scheduler — IST-aware market clock that drives the engine.

Phases through the NSE day:
  PREP   08:45–09:15  refresh shared state, pre-open notes (no entries)
  OPEN   09:15–09:30  opening range forming; swing decisions late only
  SCAN   09:30–15:10  intraday scan cycles + position management
  SWING  14:30        swing (CNC) entry window
  EXIT   15:10        force square-off all MIS
  DIGEST 15:25        daily P&L digest to Telegram
Weekly digest: Saturday 10:00.

sim mode ignores the clock (always "OPEN") so the demo never looks dead.
"""

from __future__ import annotations

import threading
import time as _time
from datetime import datetime

from ..logger import get_logger
from ..notify.telegram import Telegram
from .orchestrator import TradingEngine

log = get_logger("scheduler")


class Scheduler(threading.Thread):
    def __init__(self, engine: TradingEngine):
        super().__init__(daemon=True, name="scheduler")
        self.engine = engine
        self.scan_interval = 120 if engine.sim_mode else 300   # seconds
        self.last_digest_day: str = ""
        self.last_weekly_week: str = ""
        self.swing_done_hour: int = -1

    # ------------------------------------------------------------------
    def _ist_phase(self, now: datetime) -> str:
        if self.engine.sim_mode:
            return "SCAN"
        hm = now.hour * 60 + now.minute
        if hm < 8 * 60 + 45:
            return "IDLE"
        if hm < 9 * 60 + 15:
            return "PREP"
        if hm < 9 * 60 + 30:
            return "OPEN"
        if hm >= 15 * 60 + 30:
            return "CLOSED"
        if hm >= 15 * 60 + 25:
            return "DIGEST"
        return "SCAN"

    # ------------------------------------------------------------------
    def run(self) -> None:
        log.info("scheduler started (interval %ds, sim=%s)", self.scan_interval,
                 self.engine.sim_mode)
        tg = Telegram()
        while True:
            try:
                now = self.engine.now_ist()
                phase = self._ist_phase(now)
                self.engine.phase = phase

                if phase == "PREP":
                    self.engine.refresh_shared()
                elif phase in ("OPEN", "SCAN"):
                    self.engine.run_cycle(do_entries=True)
                    self.engine.phase = "SCAN"
                elif phase == "DIGEST":
                    self.engine.manage_positions()
                    if self.last_digest_day != now.strftime("%Y-%m-%d"):
                        tg.daily_digest(self.engine)
                        self.last_digest_day = now.strftime("%Y-%m-%d")
                elif phase == "CLOSED":
                    self.engine.manage_positions()
                    # weekly digest Saturday
                    if now.weekday() == 5 and self.last_weekly_week != now.strftime("%G-W%V"):
                        tg.weekly_digest(self.engine)
                        self.last_weekly_week = now.strftime("%G-W%V")
                elif phase == "IDLE":
                    pass

                _time.sleep(min(self.scan_interval, 30) if phase != "SCAN" else self.scan_interval)
            except Exception:
                log.exception("scheduler loop error")
                _time.sleep(30)
