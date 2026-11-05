"""PyTorch LSTM leg of the ensemble (OPTIONAL — gracefully skipped if torch
is not installed). Input: 30-session sequence of selected normalised features;
output: P(next 5-session move > +0.5%).

The bot never requires torch: MLPrediction falls back to LightGBM-only, and
further back to a cold-start heuristic, so the product runs everywhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..logger import get_logger

log = get_logger("lstm")

SEQ_LEN = 30
SEQ_FEATURES = ["ret1", "rsi14", "atr_pct", "vol_z", "bench_ret1", "sentiment"]


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def build_sequences(X: pd.DataFrame) -> Optional[np.ndarray]:
    cols = [c for c in SEQ_FEATURES if c in X.columns]
    if len(cols) < len(SEQ_FEATURES):
        return None
    arr = X[cols].fillna(0.0).values.astype(np.float32)
    # z-normalise per feature over the sample
    mu, sd = arr.mean(0), arr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    arr = (arr - mu) / sd
    if len(arr) < SEQ_LEN + 10:
        return None
    windows = np.stack([arr[i - SEQ_LEN:i] for i in range(SEQ_LEN, len(arr) + 1)])
    return windows


class LSTMModel:
    def __init__(self, n_features: int):
        import torch.nn as nn
        self.net = nn.LSTM(input_size=n_features, hidden_size=32, num_layers=1,
                           batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.net(x)
        return self.head(out[:, -1, :])


def train(X: pd.DataFrame, y: pd.Series, epochs: int = 15) -> Tuple[Optional[object], Optional[dict]]:
    if not torch_available():
        log.info("torch not installed — LSTM leg disabled")
        return None, None
    import torch
    import torch.nn as nn

    seq = build_sequences(X)
    if seq is None:
        return None, None
    labels = y.values[SEQ_LEN - 1:]  # window i ends at row SEQ_LEN-1+i
    n = min(len(seq), len(labels))
    seq_t = torch.tensor(seq[:n])
    y_t = torch.tensor(labels[:n], dtype=torch.float32).unsqueeze(1)

    split = int(n * 0.8)
    model = LSTMModel(seq.shape[2])
    opt = torch.optim.Adam(
        list(model.net.parameters()) + list(model.head.parameters()), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model.forward(seq_t[:split])
        loss = loss_fn(logits, y_t[:split])
        loss.backward()
        opt.step()

    # quick validation AUC
    auc = None
    if n - split > 10:
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model.forward(seq_t[split:])).numpy().ravel()
        from sklearn.metrics import roc_auc_score
        try:
            auc = float(roc_auc_score(labels[split:n], p))
        except ValueError:
            auc = None
    log.info("LSTM trained: n=%d auc=%s", n, auc)
    return model, {"n": n, "val_auc": auc, "seq_features": SEQ_FEATURES}


def predict_proba(model, X: pd.DataFrame) -> Optional[np.ndarray]:
    """Probability for the LAST window only (what the agent needs)."""
    if not torch_available() or model is None:
        return None
    import torch
    seq = build_sequences(X)
    if seq is None:
        return None
    model.eval()
    with torch.no_grad():
        x = torch.tensor(seq[-1:]).float()
        return float(torch.sigmoid(model.forward(x)).item())


def save(model, path: Path) -> None:
    try:
        import torch
        torch.save(model.state_dict(), path)
    except Exception as e:
        log.warning("lstm save failed: %s", e)
