"""MLPrediction agent — the probability engine the whole book leans on.

Consumes engineered features (see ml/features.py) for TWO horizons:
  h=1 → next-session directional edge (gates intraday entries)
  h=5 → 5-session forward edge (gates swing entries)

Ensemble priority: LightGBM+LSTM → LightGBM → cold-start heuristic.
Hard output: P(up), model mode, confidence, feature snapshot for the UI.
"""

from __future__ import annotations

import numpy as np

from ..ml.predictor import Predictor, HORIZON_INTRADAY, HORIZON_SWING
from ..models import AgentReport
from .base import BaseAgent, MarketContext


class MLPredictionAgent(BaseAgent):
    name = "MLPrediction"
    description = "LightGBM(+LSTM) forward-return probability at h=1 & h=5"

    def __init__(self, pred_intraday: Predictor, pred_swing: Predictor):
        super().__init__()
        self.pred_intraday = pred_intraday
        self.pred_swing = pred_swing

    def run(self, ctx: MarketContext) -> AgentReport:
        # --- swing probability (h=5) --------------------------------------
        p_swing, conf_swing, mode = None, 0.3, "coldstart"
        if ctx.features_swing is not None and len(ctx.features_swing):
            row = ctx.features_swing.iloc[-1]
            p_swing, conf_swing = self.pred_swing.predict(row, ctx.features_swing)
            mode = self.pred_swing.mode

        # --- intraday probability (h=1) ------------------------------------
        p_intra, conf_intra = None, 0.3
        if ctx.features_intraday is not None and len(ctx.features_intraday):
            row = ctx.features_intraday.iloc[-1]
            p_intra, conf_intra = self.pred_intraday.predict(row, ctx.features_intraday)

        if p_swing is None and p_intra is None:
            return AgentReport(agent=self.name, status="SKIP", note="no features available")

        # fused probability: swing-first, intraday as tiebreak context
        if p_swing is not None and p_intra is not None:
            p = 0.65 * p_swing + 0.35 * p_intra
            conf = 0.6 * conf_swing + 0.4 * conf_intra
        else:
            p = p_swing if p_swing is not None else p_intra
            conf = conf_swing if p_swing is not None else conf_intra

        # map probability → score centered at 0.5
        score = float(np.clip((p - 0.5) * 400, -100, 100))
        note = (f"P(up) {p:.2f} [{mode}] swing {p_swing if p_swing is not None else '—'} / "
                f"intra {p_intra if p_intra is not None else '—'}")

        return AgentReport(
            agent=self.name, score=score, confidence=conf,
            note=note,
            data={
                "p_up": round(p, 3), "p_swing": round(p_swing, 3) if p_swing is not None else None,
                "p_intraday": round(p_intra, 3) if p_intra is not None else None,
                "mode": mode, "model_info": self.pred_swing.info(),
                "top_features": self._top_features(ctx),
            })

    @staticmethod
    def _top_features(ctx: MarketContext) -> dict:
        try:
            if ctx.features_swing is None:
                return {}
            row = ctx.features_swing.iloc[-1]
            keys = ["ret5", "ret20", "rsi14", "dist_ema50", "vol_z", "rel_strength_20", "sentiment"]
            out = {}
            for k in keys:
                if k in row.index:
                    v = row[k]
                    out[k] = round(float(v), 3) if v is not None and not np.isnan(v) else None
            return out
        except Exception:
            return {}
