"""Dhan API v2 REST client — market data, funds, positions, orders.

Docs: https://dhanhq.com/docs/v2/
Auth: headers `access-token` (JWT) + `client-id`. Tokens expire (typically
daily) — regenerate from web.dhan.co and paste into .env.

The client is deliberately defensive: Dhan's JSON shapes have evolved over
time, so every parser checks several field spellings and logs raw bodies on
failure instead of crashing the engine.
"""

from __future__ import annotations

import requests
from datetime import date, timedelta
from typing import Dict, List, Optional

from ..config import SETTINGS
from ..logger import get_logger
from .instruments import INSTRUMENTS

log = get_logger("dhan")

BASE = "https://api.dhan.co"


class DhanClient:
    def __init__(self, client_id: str = "", access_token: str = ""):
        self.client_id = client_id or SETTINGS.dhan_client_id
        self.token = access_token or SETTINGS.dhan_access_token
        self.session = requests.Session()
        self.session.headers.update({
            "access-token": self.token,
            "client-id": self.client_id,
            "Accept": "application/json",
            "Content-Type": "application/json",
        })

    # ------------------------------------------------------------- util ---
    def _req(self, method: str, path: str, json_body: Optional[dict] = None) -> Optional[dict]:
        if not SETTINGS.has_dhan_creds:
            return None
        try:
            r = self.session.request(method, f"{BASE}{path}", json=json_body, timeout=15)
            if r.status_code in (401, 403):
                log.error("Dhan auth rejected (%s) — regenerate access token.", r.status_code)
                return None
            if r.status_code == 429:
                log.warning("Dhan rate limit hit on %s", path)
                return None
            r.raise_for_status()
            return r.json() if r.text else {}
        except requests.RequestException as e:
            log.warning("Dhan %s %s failed: %s", method, path, e)
            return None
        except ValueError:
            log.warning("Dhan returned non-JSON on %s: %.200s", path, r.text)
            return None

    # ------------------------------------------------------ historical ----
    def historical(self, security_id: str, interval: str = "1D",
                   days: int = 400, instrument: str = "EQUITY",
                   exchange_segment: str = "NSE_EQ") -> Optional[List[dict]]:
        """interval: 1D | 1H | 15 | 5 | 1  → list of {ts, open, high, low, close, volume}."""
        to_d = date.today()
        from_d = to_d - timedelta(days=days)
        body = {
            "securityId": str(security_id),
            "exchangeSegment": exchange_segment,
            "instrument": instrument,
            "interval": interval,
            "fromDate": from_d.isoformat(),
            "toDate": to_d.isoformat(),
        }
        data = self._req("POST", "/charts/2.0/historical", body)
        if not isinstance(data, dict):
            return None
        keys = ("open", "high", "low", "close", "volume", "timestamp")
        if any(k not in data for k in keys):
            log.warning("Dhan historical unexpected keys: %s", list(data.keys())[:12])
            return None
        try:
            out = []
            for i in range(len(data["close"])):
                out.append({
                    "ts": data["timestamp"][i],
                    "open": float(data["open"][i]),
                    "high": float(data["high"][i]),
                    "low": float(data["low"][i]),
                    "close": float(data["close"][i]),
                    "volume": float(data.get("volume", [0] * len(data["close"]))[i] or 0),
                })
            return out
        except Exception as e:
            log.warning("Dhan historical parse error: %s", e)
            return None

    # ------------------------------------------------------------- ltp ----
    def ltp(self, security_ids: List[str], segment: str = "NSE_EQ") -> Dict[str, float]:
        """POST /marketfeed/ltp  body {"NSE_EQ": [11536, ...]} → {securityId: ltp}."""
        if not security_ids:
            return {}
        data = self._req("POST", "/marketfeed/ltp", {segment: [int(s) for s in security_ids]})
        out: Dict[str, float] = {}
        if isinstance(data, dict):
            payload = data.get("data", data)
            seg = payload.get(segment, {}) if isinstance(payload, dict) else {}
            if isinstance(seg, dict):
                for sid, v in seg.items():
                    lp = (v or {}).get("last_price") if isinstance(v, dict) else v
                    try:
                        out[str(sid)] = float(lp)
                    except (TypeError, ValueError):
                        pass
        return out

    # ------------------------------------------------------------ funds ---
    def available_funds(self) -> Optional[float]:
        data = self._req("GET", "/funds?limit=1")
        if not isinstance(data, dict):
            return None
        for k in ("availabelBalance", "availableBalance", "available_balance"):
            v = data.get(k)
            if v is None and isinstance(data.get("data"), dict):
                v = data["data"].get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return None

    # -------------------------------------------------------- positions ---
    def positions(self) -> Optional[List[dict]]:
        data = self._req("GET", "/positions")
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]
        return None

    # ----------------------------------------------------------- orders ---
    def place_order(self, symbol: str, side: str, qty: int, product: str = "CNC",
                    order_type: str = "MARKET", price: Optional[float] = None,
                    trigger_price: Optional[float] = None) -> Optional[str]:
        """LIVE path (gated by engine — never called in paper mode).
        side: BUY|SELL  product: CNC|MIS  order_type: MARKET|LIMIT|SL|SLM"""
        sec_id = INSTRUMENTS.security_id(symbol)
        if sec_id is None:
            log.error("cannot place order for %s — no securityId", symbol)
            return None
        body: Dict[str, object] = {
            "transactionType": side.upper(),
            "exchangeSegment": "NSE_EQ",
            "productType": product,
            "orderType": order_type,
            "validity": "DAY",
            "securityId": sec_id,
            "quantity": int(qty),
        }
        if price is not None:
            body["price"] = round(float(price), 2)
        if trigger_price is not None:
            body["triggerPrice"] = round(float(trigger_price), 2)
        data = self._req("POST", "/orders", body)
        if isinstance(data, dict):
            oid = data.get("orderId") or (data.get("data") or {}).get("orderId")
            if data.get("orderStatus") not in ("TRANSIT", "PENDING", None) and not oid:
                log.warning("Dhan order rejected: %s", data)
                return None
            return str(oid) if oid else None
        return None

    def ping(self) -> bool:
        """Connectivity + auth check for scripts/setup_check.py."""
        return self.available_funds() is not None
