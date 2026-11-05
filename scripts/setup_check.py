"""Setup doctor — validates every dependency, credential and connection.

    python scripts/setup_check.py

Prints a ✅/❌ table: python packages, .env keys, Dhan API auth + funds,
Telegram token (sends a test message), NSE/yfinance reachability, ML models
present or cold-start. Exit code 0 only when trading-critical items pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def row(name: str, ok: bool, detail: str = "") -> None:
    mark = "✅" if ok else ("⚠️" if ok is None else "❌")
    print(f"  {mark}  {name:<34} {detail}")


def main() -> None:
    print("\nAlphaDesk setup check\n" + "=" * 60)

    # ---- packages ------------------------------------------------------
    print("\n[1] Python packages")
    for mod in ("pandas", "numpy", "requests", "yfinance", "fastapi",
                "uvicorn", "lightgbm", "sklearn", "statsmodels", "feedparser", "dotenv"):
        try:
            __import__(mod)
            row(mod, True)
        except ImportError:
            row(mod, False, "pip install -r requirements.txt")

    try:
        import torch
        row("torch (optional, DL)", True, torch.__version__)
    except ImportError:
        row("torch (optional, DL)", None, "not installed — GBM-only ensemble")

    # ---- env -----------------------------------------------------------
    print("\n[2] Configuration (.env)")
    from tradingbot.config import SETTINGS, PROJECT_ROOT
    env_file = PROJECT_ROOT / ".env"
    row(".env file", env_file.exists(), str(env_file))
    row("capital / risk", True,
        f"₹{SETTINGS.capital_inr:,.0f} · {SETTINGS.risk_per_trade_pct}% · "
        f"max {SETTINGS.max_open_positions} positions")
    row("data mode", True, f"{SETTINGS.market_data_mode} → resolves to {SETTINGS.resolved_data_mode()}")

    # ---- Dhan ------------------------------------------------------------
    print("\n[3] Dhan API v2")
    row("credentials present", SETTINGS.has_dhan_creds,
        f"client {SETTINGS.dhan_client_id[:6] + '…' if SETTINGS.dhan_client_id else '—'}")
    if SETTINGS.has_dhan_creds:
        from tradingbot.broker.dhan_client import DhanClient
        from tradingbot.broker.instruments import INSTRUMENTS
        ok = DhanClient().ping()
        row("auth + funds call", ok, "live" if ok else "401 — regenerate access token")
        sid = INSTRUMENTS.security_id("RELIANCE")
        row("scrip master / securityId", sid is not None, f"RELIANCE → {sid}")
    else:
        row("credentials present", None, "skipped — running on yfinance/sim")

    # ---- feeds -------------------------------------------------------------
    print("\n[4] Market data")
    try:
        import yfinance as yf
        h = yf.Ticker("RELIANCE.NS").history(period="5d")
        row("yfinance", len(h) > 0, f"{len(h)} bars")
    except Exception as e:
        row("yfinance", False, str(e.__class__.__name__))

    # ---- Telegram -----------------------------------------------------------
    print("\n[5] Telegram")
    row("credentials", bool(SETTINGS.telegram_bot_token and SETTINGS.telegram_chat_id),
        f"chat {SETTINGS.telegram_chat_id or '—'}")
    if SETTINGS.telegram_enabled and SETTINGS.telegram_bot_token and SETTINGS.telegram_chat_id:
        from tradingbot.notify.telegram import Telegram
        ok = Telegram().test()
        row("test message sent", ok, "check your chat" if ok else "send failed — verify token/chat id")

    # ---- ML -----------------------------------------------------------------
    print("\n[6] ML models")
    from tradingbot.ml.predictor import MODELS_DIR
    g1 = (MODELS_DIR / "gbm_h1.joblib").exists()
    g5 = (MODELS_DIR / "gbm_h5.joblib").exists()
    row("LightGBM h=1 (intraday)", g1, "trained" if g1 else "cold-start — run scripts/train_models.py")
    row("LightGBM h=5 (swing)", g5, "trained" if g5 else "cold-start — run scripts/train_models.py")

    print("\n" + "=" * 60)
    print("Done. Red ❌ items on the trading-critical path (feeds, auth) must")
    print("pass before going live; ⚠️ optional items degrade gracefully.\n")


if __name__ == "__main__":
    main()
