"""LightGBM gradient-boosted probability model with time-ordered validation."""

from __future__ import annotations

import joblib
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..logger import get_logger

log = get_logger("gbdt")


def train(X: pd.DataFrame, y: pd.Series) -> Tuple[object, Dict]:
    """Train LGBMClassifier with a chronologically-held-out validation tail."""
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score

    split = int(len(X) * 0.8)
    Xtr, Xva = X.iloc[:split], X.iloc[split:]
    ytr, yva = y.iloc[:split], y.iloc[split:]

    model = LGBMClassifier(
        n_estimators=400, learning_rate=0.03, max_depth=4,
        num_leaves=15, min_child_samples=25,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.5,
        class_weight="balanced", random_state=42, verbose=-1,
    )
    model.fit(Xtr, ytr)

    auc = None
    if len(Xva) and yva.nunique() == 2:
        p = model.predict_proba(Xva)[:, 1]
        auc = float(roc_auc_score(yva, p))

    meta = {
        "train_rows": int(len(Xtr)),
        "val_rows": int(len(Xva)),
        "val_auc": auc,
        "pos_rate": float(y.mean()),
        "features": list(X.columns),
    }
    log.info("GBM trained: rows=%d auc=%s pos_rate=%.2f", len(Xtr), auc, meta["pos_rate"])
    return model, meta


def predict_proba(model: object, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def save(model: object, path: Path) -> None:
    joblib.dump(model, path)


def load(path: Path) -> Optional[object]:
    try:
        return joblib.load(path)
    except Exception as e:
        log.warning("gbm load failed: %s", e)
        return None
