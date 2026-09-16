"""Dhan data and order access (NSE).

Verification status, stated plainly:

* The request construction and response parsing below are unit-tested against
  fixtures shaped like the documented Dhan v2 responses.
* ``api.dhan.co`` and ``dhan-api.dhan.co`` are **not reachable from the sandbox
  this was built in**, so none of this has ever executed against the live
  endpoint. Run ``tradingbot dhan-check`` with your own credentials before
  trusting it.

Security IDs are resolved from Dhan's official scrip master rather than a
hardcoded table, and resolution failure is a hard error: a wrong security id
means buying the wrong stock, which is worse than refusing to start.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

import pandas as pd

from tradingbot.data.providers import DataProvider
from tradingbot.data.schema import History, validate_history

HistoricalFactory = Callable[[], object]

EXCHANGE_NSE = "NSE_EQ"
INSTRUMENT_EQUITY = "EQUITY"


class SecurityResolutionError(RuntimeError):
    """Raised when a symbol cannot be mapped to a Dhan security id."""


def resolve_security_ids(
    symbols: Sequence[str],
    scrip_master: str | Path,
    exchange: str = "NSE",
    instrument: str = "EQUITY",
) -> Dict[str, str]:
    """Map trading symbols to Dhan security ids using the official scrip master.

    The master is the CSV Dhan publishes (``dhan_scrip_master.csv``). We filter to
    equities on the requested exchange and index by trading symbol.
    """
    fp = Path(scrip_master)
    if not fp.exists():
        raise SecurityResolutionError(
            f"scrip master not found at {fp}. Download it from Dhan and pass --scrip-master."
        )

    index: Dict[str, str] = {}
    with fp.open(newline="", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            if row.get("SEM_EXM_EXCH_ID") != exchange:
                continue
            if row.get("SEM_INSTRUMENT_NAME") != instrument:
                continue
            symbol = (row.get("SEM_TRADING_SYMBOL") or "").strip().upper()
            sec_id = (row.get("SEM_SMST_SECURITY_ID") or "").strip()
            if symbol and sec_id:
                index[symbol] = sec_id

    wanted = {s.upper() for s in symbols}
    missing = sorted(wanted - set(index))
    if missing:
        raise SecurityResolutionError(
            f"no security id for {len(missing)} symbol(s): {', '.join(missing[:10])}"
            + ("..." if len(missing) > 10 else "")
        )
    return {s.upper(): index[s.upper()] for s in wanted}


def parse_historical(response: dict) -> pd.DataFrame:
    """Convert a Dhan ``/charts/historical`` or ``/charts/intraday`` body to OHLCV.

    Dhan returns column arrays plus ``start_time`` as epoch seconds. Anything
    else is treated as an error rather than silently producing an empty frame.
    """
    status = response.get("status")
    if status is not None and status != "success":
        raise RuntimeError(f"dhan returned status={status}: {response.get('remarks')}")

    stamps = response.get("start_time") or response.get("timestamp")
    if stamps is None:
        raise RuntimeError(f"unrecognised dhan payload keys: {sorted(response)}")

    frame = pd.DataFrame(
        {
            "open": [float(v) for v in response.get("open", [])],
            "high": [float(v) for v in response.get("high", [])],
            "low": [float(v) for v in response.get("low", [])],
            "close": [float(v) for v in response.get("close", [])],
            "volume": [float(v) for v in response.get("volume", [])],
        },
        index=pd.to_datetime([int(t) for t in stamps], unit="s", utc=True),
    )
    frame.index = frame.index.tz_convert("Asia/Kolkata").tz_localize(None).normalize()
    frame.index.name = "date"
    return frame


class DhanProvider(DataProvider):
    """Historical daily bars from Dhan.

    ``historical_factory`` returns an object exposing
    ``historical_daily_data(...)``; the default builds a real
    ``dhanhq.HistoricalData`` from your credentials. Inject a stub in tests.
    """

    def __init__(
        self,
        security_map: Dict[str, str],
        client_id: Optional[str] = None,
        access_token: Optional[str] = None,
        historical_factory: Optional[HistoricalFactory] = None,
    ) -> None:
        self.security_map = {k.upper(): v for k, v in security_map.items()}
        self.client_id = client_id or os.environ.get("DHAN_CLIENT_ID", "")
        self.access_token = access_token or os.environ.get("DHAN_ACCESS_TOKEN", "")
        self._historical_factory = historical_factory

    def _historical(self) -> object:
        if self._historical_factory is not None:
            return self._historical_factory()
        if not self.client_id or not self.access_token:
            raise RuntimeError(
                "Dhan credentials missing: set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN"
            )
        from dhanhq import DhanContext, HistoricalData  # imported lazily

        return HistoricalData(DhanContext(self.client_id, self.access_token))

    def load(self, symbols: Sequence[str], start: str, end: Optional[str] = None) -> History:
        symbols = [s.upper() for s in symbols]
        missing = [s for s in symbols if s not in self.security_map]
        if missing:
            raise SecurityResolutionError(f"no security id for: {', '.join(missing)}")

        end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        api = self._historical()
        history: History = {}
        for symbol in symbols:
            payload = api.historical_daily_data(  # type: ignore[attr-defined]
                security_id=self.security_map[symbol],
                exchange_segment=EXCHANGE_NSE,
                instrument_type=INSTRUMENT_EQUITY,
                from_date=pd.Timestamp(start).strftime("%Y-%m-%d"),
                to_date=pd.Timestamp(end).strftime("%Y-%m-%d"),
            )
            frame = parse_historical(payload)
            if frame.empty:
                continue
            history[symbol] = frame
        if not history:
            raise RuntimeError("Dhan returned no candles for any requested symbol")
        return validate_history(history)


class DhanPaperOrders:
    """Order gateway shaped like Dhan's ``Order.place_order``.

    Paper mode never calls this. It exists so the switch to live trading is a
    one-line constructor change and not a rewrite of the execution path -- and
    so the argument mapping gets written and reviewed now, while it is cheap.
    """

    def __init__(self, order_api: object) -> None:
        self._api = order_api

    def buy(self, security_id: str, quantity: int, price: float = 0.0,
            product: str = "CNC", correlation_id: str = "") -> dict:
        return self._api.place_order(  # type: ignore[attr-defined]
            security_id=security_id,
            exchange_segment=EXCHANGE_NSE,
            transaction_type="BUY",
            quantity=quantity,
            order_type="MARKET" if price <= 0 else "LIMIT",
            product_type=product,
            price=price,
            trigger_price=0,
            disclosed_quantity=0,
            after_market_order=False,
            validity="DAY",
            amo_time="OPEN",
            tag=correlation_id or None,
        )

    def sell(self, security_id: str, quantity: int, price: float = 0.0,
             product: str = "CNC", correlation_id: str = "") -> dict:
        return self._api.place_order(  # type: ignore[attr-defined]
            security_id=security_id,
            exchange_segment=EXCHANGE_NSE,
            transaction_type="SELL",
            quantity=quantity,
            order_type="MARKET" if price <= 0 else "LIMIT",
            product_type=product,
            price=price,
            trigger_price=0,
            disclosed_quantity=0,
            after_market_order=False,
            validity="DAY",
            amo_time="OPEN",
            tag=correlation_id or None,
        )


def load_security_map(path: str | Path) -> Dict[str, str]:
    """Load a ``symbol -> security_id`` JSON map produced by scripts/make_security_map.py."""
    import json

    return {k.upper(): str(v) for k, v in json.loads(Path(path).read_text()).items()}
