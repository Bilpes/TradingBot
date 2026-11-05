"""Ensemble prediction manager.

Priority of brains:
  1. trained LightGBM + LSTM ensemble (artifacts in data/models/)
  2. LightGBM only (no torch)
  3. cold-start heuristic (momentum/trend composite) — so the bot is never
     dumb on day one; run scripts/train_models.py to upgrade it.

The ensemble:  p = w_gbm · P_gbm + w_lstm · P_lstm  with a disagreement
penalty — when the two models disagree by more than 0.25 we shave confidence
instead of gambling on the louder voice.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import data_dir
from ..logger import get_logger
from . import gbdt, lstm

log = get_logger("predictor")

MODELS_DIR = data_dir() / "models"
W_GBM, W_LSTM = 0.6, 0.4


class Predictor:
    def __init__(self, horizon: int = 5):
        self.horizon = horizon                    # 1 = intraday-ish, 5 = swing
        self.mode = "coldstart"
        self.gbm = None
        self.lstm_model = None
        self.meta: Dict = {}
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        gbm_path = MODELS_DIR / f"gbm_h{self.horizon}.joblib"
        meta_path = MODELS_DIR / f"meta_h{self.horizon}.json"
        if gbm_path.exists():
            self.gbm = gbdt.load(gbm_path)
            if self.gbm is not None:
                self.mode = "gbm"
                if meta_path.exists():
                    try:
                        self.meta = json.loads(meta_path.read_text())
                    except Exception:
                        self.meta = {}
        if lstm.torch_available():
            pt_path = MODELS_DIR / f"lstm_h{self.horizon}.pt"
            if pt_path.exists():
                try:
                    import torch  # noqa: F401
                    m = lstm.LSTMModel(len(lstm.SEQ_FEATURES))
                    m.net.load_state_dict(torch.load(pt_path, map_location="cpu", weights_only=True))
                    # head weights saved alongside with suffix
                    head_path = MODELS_DIR / f"lstm_head_h{self.horizon}.pt"
                    if head_path.exists():
                        m.head.load_state_dict(torch.load(head_path, map_location="cpu", weights_only=True))
                    self.lstm_model = m
                    self.mode = "ensemble"
                except Exception as e:
                    log.warning("lstm load failed: %s", e)
        log.info("predictor h=%d mode=%s", self.horizon, self.mode)

    @property
    def is_trained(self) -> bool:
        return self.gbm is not None

    # ------------------------------------------------------------------
    def _coldstart_prob(self, features: pd.Series, df: pd.DataFrame) -> float:
        """Transparent heuristic P(up over horizon): momentum + trend + pullback
        structure, squashed through a logistic. Documented, bounded, boring."""
        score = 0.0
        r5 = features.get("ret5", 0.0) or 0.0
        r20 = features.get("ret20", 0.0) or 0.0
        score += np.tanh(r5 * 30) * 0.35
        score += np.tanh(r20 * 15) * 0.25
        if features.get("dist_ema50", 0) is not None and not math.isnan(features.get("dist_ema50", 0.0) or 0.0):
            score += np.tanh((features.get("dist_ema50") or 0.0) * 25) * 0.15
        rsi = features.get("rsi14", 50.0)
        if rsi is not None and not math.isnan(rsi):
            # mild mean-reversion at extremes in the heuristic
            if rsi > 75:
                score -= 0.10
            elif rsi < 30:
                score += 0.10
        vol_z = features.get("vol_z", 0.0)
        if vol_z is not None and not math.isnan(vol_z) and vol_z > 1.5:
            score += 0.08                      # institutional participation
        p = 1 / (1 + math.exp(-2.2 * score))
        return float(min(0.85, max(0.15, p)))

    # ------------------------------------------------------------------
    def predict(self, feature_row: pd.Series, features_df: pd.DataFrame) -> Tuple[float, float]:
        """→ (P(up over horizon), model_confidence 0..1)"""
        X = pd.DataFrame([feature_row]).fillna(0.0).astype(float)

        p_gbm, conf_gbm = None, 0.5
        if self.gbm is not None:
            try:
                p_gbm = float(gbdt.predict_proba(self.gbm, X[list(self.gbm.feature_name_)] if hasattr(self.gbm, "feature_name_") else X)[0])
            except Exception as e:
                log.debug("gbm predict error: %s", e)
                p_gbm = None

        p_lstm = None
        if self.lstm_model is not None:
            p_lstm = lstm.predict_proba(self.lstm_model, features_df)

        if p_gbm is not None and p_lstm is not None:
            p = W_GBM * p_gbm + W_LSTM * p_lstm
            disagreement = abs(p_gbm - p_lstm)
            conf = 0.8 - min(0.35, disagreement * 1.2)
        elif p_gbm is not None:
            p, conf = p_gbm, 0.65
        elif p_lstm is not None:
            p, conf = p_lstm, 0.55
        else:
            p = self._coldstart_prob(feature_row, features_df)
            conf = 0.35

        return float(min(0.95, max(0.05, p))), float(conf)

    # ------------------------------------------------------------------
    def info(self) -> dict:
        return {
            "horizon": self.horizon,
            "mode": self.mode,
            "val_auc": self.meta.get("val_auc"),
            "train_rows": self.meta.get("train_rows"),
            "torch": lstm.torch_available(),
        }


HORIZON_INTRADAY = 1
HORIZON_SWING = 5
