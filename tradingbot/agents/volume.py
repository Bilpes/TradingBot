"""Volume-spike agent: a confirmation vote, never a direction of its own.

Volume has no sign. A 5x volume day is bullish or bearish depending entirely on
what price did, so this agent multiplies spike intensity by the sign of the
short-horizon return. A spike on a down day is a distribution signal and votes
negative -- which is the case a naive "high volume = interest = buy" agent gets
exactly backwards.

It carries a deliberately low weight. Volume confirms; it does not predict.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from tradingbot.agents.base import Agent, Signal, clip
from tradingbot.data.schema import History, closes, row_asof
from tradingbot.indicators import volume_zscore


class VolumeSpikeAgent(Agent):
    name = "volume_spike"
    weight = 0.6
    min_history = 30

    def __init__(self, window: int = 20, z_threshold: float = 1.5, return_window: int = 3) -> None:
        self.window = window
        self.z_threshold = z_threshold
        self.return_window = return_window
        self.min_history = max(self.min_history, window + return_window)
        self._score: pd.DataFrame | None = None
        self._conf: pd.DataFrame | None = None
        self._z: pd.DataFrame | None = None

    def fit(self, history: History) -> None:
        px = closes(history)

        z = pd.DataFrame(
            {s: volume_zscore(history[s]["volume"], self.window) for s in history}
        ).sort_index()
        self._z = z

        ret = px / px.shift(self.return_window) - 1.0
        direction = np.sign(ret)

        # Intensity above the threshold, saturating at 3 z-scores.
        excess = (z - self.z_threshold).clip(lower=0.0) / 1.5
        intensity = excess.clip(0.0, 1.0)

        self._score = (direction * intensity).clip(-1.0, 1.0)
        # A volume agent is only worth listening to when volume actually spiked.
        self._conf = (intensity * 0.75).clip(0.0, 0.7)

    def signals_on(self, date: pd.Timestamp) -> Sequence[Signal]:
        if self._score is None:
            raise RuntimeError("fit() must be called before signals_on()")
        srow = row_asof(self._score, date)
        crow = row_asof(self._conf, date)
        zrow = row_asof(self._z, date)
        if srow is None or crow is None:
            return []

        out = []
        for symbol in srow.index:
            score, conf = float(srow[symbol]), float(crow[symbol])
            if not np.isfinite(score) or not np.isfinite(conf) or conf < 0.05:
                continue
            z = float(zrow[symbol]) if zrow is not None else float("nan")
            out.append(
                Signal(
                    symbol=symbol,
                    score=clip(score),
                    confidence=conf,
                    agent=self.name,
                    reason=f"volume {z:+.1f} sigma vs {self.window}d mean",
                    parts={"volume_z": z},
                )
            )
        return out
