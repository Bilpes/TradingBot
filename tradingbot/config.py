"""Central configuration: environment loading, strategy constants, NSE session rules.

Everything money-related lives here so risk can be audited in one place.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(key: str, default: str) -> str:
    return os.getenv(key, default).strip()


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


# ----------------------------------------------------------------------------
# NSE session (IST). Intraday force-square-off is before the 15:20 broker cut-off.
# ----------------------------------------------------------------------------
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"
INTRADAY_SQUAREOFF = "15:10"     # force-exit all MIS positions
SWING_DECISION_TIMES = ["14:30"]  # swing entries evaluated once, late in the day
ORB_MINUTES = 15                 # opening-range breakout window
WEEKLY_DIGEST_DAY = "SAT"
WEEKLY_DIGEST_TIME = "10:00"
DAILY_DIGEST_TIME = "15:25"

AGENT_WEIGHTS: Dict[str, float] = {
    "Technical": 0.25,
    "MLPrediction": 0.25,
    "TimeSeries": 0.15,
    "Sentiment": 0.10,
    "AltData": 0.10,
    "MarketStructure": 0.10,
    "Macro": 0.05,
}

# Realistic India brokerage/levy model (approximations — editable, per trade leg).
# Dhan pricing: equity delivery ₹0 brokerage; intraday ₹20 or 0.03% whichever is lower.
CHARGES = {
    "stt_cnc": 0.0010,           # 0.10% both legs (delivery)
    "stt_mis_sell": 0.00025,     # 0.025% sell leg only (intraday)
    "exchange_txn": 0.0000297,   # ~0.00297% NSE equity
    "sebi": 0.000001,            # ₹10/crore
    "gst": 0.18,                 # on brokerage + txn + SEBI
    "stamp_cnc_buy": 0.00015,    # 0.015% buy leg (delivery)
    "stamp_mis_buy": 0.00003,    # 0.003% buy leg (intraday)
    "intraday_brokerage_pct": 0.0003,
    "intraday_brokerage_cap": 20.0,
}


@dataclass
class Settings:
    """Runtime settings loaded once at boot."""

    # broker
    dhan_client_id: str = field(default_factory=lambda: _env("DHAN_CLIENT_ID", ""))
    dhan_access_token: str = field(default_factory=lambda: _env("DHAN_ACCESS_TOKEN", ""))

    # mode
    market_data_mode: str = field(default_factory=lambda: _env("MARKET_DATA_MODE", "auto"))
    live_trading: bool = field(default_factory=lambda: _env("LIVE_TRADING", "false").lower() == "true")

    # capital & risk
    capital_inr: float = field(default_factory=lambda: _env_float("CAPITAL_INR", 100_000.0))
    risk_per_trade_pct: float = field(default_factory=lambda: _env_float("RISK_PER_TRADE_PCT", 1.0))
    max_open_positions: int = field(default_factory=lambda: _env_int("MAX_OPEN_POSITIONS", 3))
    daily_loss_limit_pct: float = field(default_factory=lambda: _env_float("DAILY_LOSS_LIMIT_PCT", 2.0))
    max_trades_per_day: int = field(default_factory=lambda: _env_int("MAX_TRADES_PER_DAY", 6))

    # telegram
    telegram_enabled: bool = field(default_factory=lambda: _env("TELEGRAM_ENABLED", "true").lower() == "true")
    telegram_bot_token: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: _env("TELEGRAM_CHAT_ID", ""))

    # strategy
    ml_prob_gate_intraday: float = field(default_factory=lambda: _env_float("ML_PROB_GATE_INTRADAY", 0.56))
    ml_prob_gate_swing: float = field(default_factory=lambda: _env_float("ML_PROB_GATE_SWING", 0.58))
    max_notional_per_stock_pct: float = field(default_factory=lambda: _env_float("MAX_NOTIONAL_PER_STOCK_PCT", 25.0))
    sector_concentration_pct: float = field(default_factory=lambda: _env_float("SECTOR_CONCENTRATION_PCT", 40.0))
    slippage_bps: float = field(default_factory=lambda: _env_float("SLIPPAGE_BPS", 5.0))

    # housekeeping
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    universe_file: str = field(default_factory=lambda: _env("UNIVERSE_FILE", "config/nifty50.json"))
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))

    # ------------------------------------------------------------------
    @property
    def has_dhan_creds(self) -> bool:
        return bool(self.dhan_client_id) and bool(self.dhan_access_token)

    @property
    def risk_amount_inr(self) -> float:
        """₹ at risk per trade = risk% × capital. For ₹1L @1% → ₹1,000."""
        return round(self.capital_inr * self.risk_per_trade_pct / 100.0, 0)

    def resolved_data_mode(self) -> str:
        """Resolve 'auto' to the best available feed."""
        if self.market_data_mode == "auto":
            if self.has_dhan_creds:
                return "dhan"
            return "yfinance"
        return self.market_data_mode

    def load_universe(self) -> List[str]:
        """Nifty-50 symbol list from config/nifty50.json (editable)."""
        path = PROJECT_ROOT / self.universe_file
        try:
            data = json.loads(path.read_text())
            return [s.upper() for s in data["symbols"]]
        except Exception:
            # absolute last-resort minimal set so the bot never dies
            return ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "TATAMOTORS",
                    "SBIN", "ITC", "LT", "BHARTIARTL"]


SETTINGS = Settings()


def data_dir() -> Path:
    d = PROJECT_ROOT / "data"
    d.mkdir(exist_ok=True)
    (d / "models").mkdir(exist_ok=True)
    (d / "logs").mkdir(exist_ok=True)
    (d / "cache").mkdir(exist_ok=True)
    return d
