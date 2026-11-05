"""Walk-forward model training — the honest way.

For each symbol: features are built leak-free (labels = FUTURE returns),
LightGBM (+ optional LSTM) is trained on a rolling window with an embargo
gap, and out-of-fold probabilities are aggregated. Artifacts land in
data/models/ and are picked up by the MLPrediction agent automatically.

Usage:
    python scripts/train_models.py --symbols RELIANCE,TCS --horizons 1,5
    python scripts/train_models.py --all          # whole Nifty-50 (slow first run)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from tradingbot.config import SETTINGS, PROJECT_ROOT, data_dir
from tradingbot.data.market_data import MarketData
from tradingbot.logger import get_logger
from tradingbot.ml import aligned_dataset, FEATURE_NAMES
from tradingbot.ml import gbdt as gbdt_mod
from tradingbot.ml import lstm as lstm_mod
from tradingbot.ml.predictor import MODELS_DIR

log = get_logger("train")


def walk_forward_auc(X: pd.DataFrame, y: pd.Series, n_folds: int = 3,
                     embargo: int = 5) -> list:
    """Chronological K-fold with embargo — never let future leak into past."""
    from sklearn.metrics import roc_auc_score
    from lightgbm import LGBMClassifier

    fold_size = len(X) // (n_folds + 1)
    aucs = []
    for k in range(n_folds):
        tr_end = fold_size * (k + 1)
        va_start = tr_end + embargo
        va_end = min(va_start + fold_size, len(X))
        if va_end - va_start < 30:
            continue
        Xtr, ytr = X.iloc[:tr_end], y.iloc[:tr_end]
        Xva, yva = X.iloc[va_start:va_end], y.iloc[va_start:va_end]
        if ytr.nunique() < 2 or yva.nunique() < 2:
            continue
        m = LGBMClassifier(n_estimators=300, learning_rate=0.04, max_depth=4,
                           num_leaves=15, min_child_samples=25,
                           class_weight="balanced", random_state=42, verbose=-1)
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xva)[:, 1]
        aucs.append(float(roc_auc_score(yva, p)))
    return aucs


def train_one(md: MarketData, symbol: str, horizon: int, index_df: pd.DataFrame,
              vix: float, with_lstm: bool) -> dict:
    df = md.daily(symbol, 1200)
    if len(df) < 400:
        return {"symbol": symbol, "h": horizon, "status": "skip (short history)"}

    X, y, _ = aligned_dataset(df, index_df, horizon=horizon,
                              vix=vix, threshold=0.005 if horizon == 5 else 0.003)
    if len(X) < 300:
        return {"symbol": symbol, "h": horizon, "status": "skip (few features)"}

    aucs = walk_forward_auc(X, y)
    auc_mean = float(np.mean(aucs)) if aucs else None

    model, meta = gbdt_mod.train(X, y)
    meta["walk_forward_auc"] = aucs
    meta["walk_forward_auc_mean"] = auc_mean
    meta["symbol"] = symbol
    meta["horizon"] = horizon
    gbdt_mod.save(model, MODELS_DIR / f"gbm_h{horizon}.joblib")
    (MODELS_DIR / f"meta_h{horizon}.json").write_text(json.dumps(meta, indent=2))

    lstm_out = None
    if with_lstm and lstm_mod.torch_available():
        lm, lmeta = lstm_mod.train(X, y, epochs=12)
        if lm is not None:
            lstm_mod.save(lm, MODELS_DIR / f"lstm_h{horizon}.pt")
            try:
                import torch
                torch.save(lm.head.state_dict(), MODELS_DIR / f"lstm_head_h{horizon}.pt")
            except Exception:
                pass
            lstm_out = lmeta

    return {"symbol": symbol, "h": horizon, "rows": int(len(X)),
            "wf_auc": [round(a, 3) for a in aucs],
            "wf_auc_mean": round(auc_mean, 3) if auc_mean else None,
            "lstm": (lstm_out or {}).get("val_auc")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="RELIANCE,INFY,TCS,SBIN,TATAMOTORS")
    ap.add_argument("--all", action="store_true", help="train on the whole universe")
    ap.add_argument("--horizons", default="1,5")
    ap.add_argument("--with-lstm", action="store_true", default=None)
    args = ap.parse_args()

    symbols = SETTINGS.load_universe() if args.all else \
        [s.strip().upper() for s in args.symbols.split(",")]
    horizons = [int(h) for h in args.horizons.split(",")]
    with_lstm = args.with_lstm if args.with_lstm is not None else lstm_mod.torch_available()

    md = MarketData(symbols, mode=SETTINGS.resolved_data_mode())
    index_df = md.index_daily()
    vix = md.india_vix()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for sym in symbols:
        for h in horizons:
            try:
                r = train_one(md, sym, h, index_df, vix, with_lstm)
            except Exception as e:
                r = {"symbol": sym, "h": h, "status": f"error: {e}"}
            results.append(r)
            print(f"  {r}")

    ok = [r for r in results if "wf_auc_mean" in r and r["wf_auc_mean"]]
    print("\n================ SUMMARY ================")
    if ok:
        mean_auc = np.mean([r["wf_auc_mean"] for r in ok])
        print(f"trained {len(ok)} models · mean walk-forward AUC {mean_auc:.3f}")
        print("AUC ≈ 0.52-0.55 is a usable edge for daily direction; 0.57+ is strong.")
    else:
        print("no models trained — check data feed (try MARKET_DATA_MODE=yfinance)")
    print(f"artifacts → {MODELS_DIR}")


if __name__ == "__main__":
    main()
