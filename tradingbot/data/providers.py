"""Data providers.

Four of them, with deliberately different trust levels:

``SyntheticProvider``
    Factor-model price paths, seeded and reproducible. Used for tests and for
    developing anywhere without a data subscription. **Its output is not market
    data** and a backtest run on it says nothing about future returns.

``CSVProvider``
    Your own OHLCV files. Use this for any number you intend to believe.

``AlpacaProvider``
    A real client for the Alpaca markets API (US), with an injectable
    ``http_get`` so parsing and pagination are unit-tested offline.

``DhanProvider``
    Lives in :mod:`tradingbot.data.dhan` -- NSE, via the official ``dhanhq`` SDK.

None of the network providers has ever reached its live endpoint from this
sandbox; egress to market-data hosts is blocked.
"""

from __future__ import annotations

import json
import os
import zlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from tradingbot.data.schema import OHLCV, History, validate_history
from tradingbot.data.universe import SECTORS, UNIVERSE, sector_of

HttpGet = Callable[[str, Mapping[str, str]], str]


def _stable_hash(text: str) -> int:
    """Process-independent 32-bit hash.

    Never use the builtin ``hash()`` for anything that must reproduce: CPython
    randomises str hashing per process, so a "seeded" generator built on it
    yields different data on every invocation.
    """
    return zlib.crc32(text.encode("utf-8"))


class DataProvider(ABC):
    @abstractmethod
    def load(self, symbols: Sequence[str], start: str, end: Optional[str] = None) -> History: ...


# --------------------------------------------------------------------------
# Synthetic
# --------------------------------------------------------------------------
class SyntheticProvider(DataProvider):
    """Seeded factor-model generator with genuine sector structure.

    Each session's return is ``market + sector + idiosyncratic``. The market
    factor switches between calm/trending/stressed regimes so a volatility gate
    has something real to react to, and each sector gets its own drift, beta and
    volatility so rotation has something real to rotate between.
    """

    def __init__(
        self,
        sessions: int = 750,
        seed: int = 7,
        start: str = "2019-01-02",
        sector_map: Callable[[str], str] = sector_of,
        universe: Optional[Sequence[str]] = None,
    ) -> None:
        self.sessions = sessions
        self.seed = seed
        self.start = start
        self.sector_map = sector_map
        # Explicit default universe. ``load([])`` used to mean "use the US
        # universe", which silently produced US symbols under an NSE sector map
        # -- every name classified as "Other" and every worker left idle.
        self.universe = list(universe) if universe is not None else list(UNIVERSE)

    def load(self, symbols: Sequence[str], start: Optional[str] = None, end: Optional[str] = None) -> History:
        rng = np.random.default_rng(self.seed)
        dates = pd.bdate_range(start=start or self.start, periods=self.sessions)
        n = self.sessions
        symbols = list(symbols) or self.universe

        sectors = sorted({self.sector_map(s) for s in symbols})
        sec_beta = {s: 0.7 + 0.09 * i for i, s in enumerate(sectors)}
        sec_drift = {s: (0.0009 if i % 3 == 0 else 0.0002) for i, s in enumerate(sectors)}
        sec_vol = {s: 0.11 + 0.02 * (i % 5) for i, s in enumerate(sectors)}

        # Regime-switching market factor: calm / trending / stressed.
        regime = np.zeros(n, dtype=int)
        cur, t = 0, 0
        while t < n:
            length = int(rng.integers(40, 140))
            regime[t:t + length] = cur
            t += length
            cur = int(rng.integers(0, 3))
        regime_mu = np.array([0.0004, 0.0012, -0.0016])
        regime_sd = np.array([0.007, 0.009, 0.020])
        market = rng.normal(regime_mu[regime], regime_sd[regime])

        sector_shock = {s: rng.normal(0.0, sec_vol[s] / np.sqrt(252), n) for s in sectors}

        history: History = {}
        for symbol in symbols:
            sector = self.sector_map(symbol)
            idio_vol = sec_vol[sector] * 0.9 / np.sqrt(252)
            r = (sec_drift[sector] + sec_beta[sector] * market
                 + sector_shock[sector] + rng.normal(0.0, idio_vol, n))
            # Deterministic per-name start price -- see _stable_hash.
            p0 = 20.0 + 13.0 * ((_stable_hash(symbol) % 97) / 97.0)
            close = p0 * np.exp(np.cumsum(r))

            prev = np.concatenate([[p0], close[:-1]])
            open_ = prev * (1.0 + rng.normal(0.0, 0.002, n))
            span = np.abs(rng.normal(0.0, 0.008, n)) + 0.001
            high = np.maximum(open_, close) * (1.0 + span)
            low = np.minimum(open_, close) * (1.0 - span)
            base_vol = 5e5 + 5e6 * ((_stable_hash(symbol + "v") % 53) / 53.0)
            volume = np.maximum(base_vol * np.exp(rng.normal(0.0, 0.35, n)), 1e3)

            history[symbol] = pd.DataFrame(
                {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
                index=dates,
            )
        return validate_history(history)


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------
class CSVProvider(DataProvider):
    """Load OHLCV from CSV.

    Accepts either a directory of ``<SYMBOL>.csv`` files or one long-format CSV
    with a ``symbol`` column. Required columns: ``date`` (or the index) plus
    ``open, high, low, close, volume``.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def _read(self, fp: Path) -> pd.DataFrame:
        cols = {c.lower(): c for c in pd.read_csv(fp, nrows=0).columns}
        idx_col = cols.get("date") or cols.get("timestamp")
        if idx_col is None:
            df = pd.read_csv(fp, index_col=0)
        else:
            df = pd.read_csv(fp, parse_dates=[idx_col], index_col=idx_col)
        df.columns = [c.lower() for c in df.columns]
        return df[list(OHLCV)]

    def load(self, symbols: Sequence[str], start: Optional[str] = None, end: Optional[str] = None) -> History:
        history: History = {}
        wanted = {s.upper() for s in symbols} if symbols else None

        if self.path.is_dir():
            files = sorted(self.path.glob("*.csv"))
            if not files:
                raise FileNotFoundError(f"no CSV files in {self.path}")
            for fp in files:
                symbol = fp.stem.upper()
                if wanted and symbol not in wanted:
                    continue
                history[symbol] = self._read(fp)
        else:
            df = pd.read_csv(self.path)
            df.columns = [c.lower() for c in df.columns]
            if "symbol" not in df.columns:
                raise ValueError(f"{self.path.name}: missing 'symbol' column")
            idx_col = "date" if "date" in df.columns else "timestamp"
            for symbol, group in df.groupby("symbol"):
                if wanted and str(symbol).upper() not in wanted:
                    continue
                g = group.set_index(pd.to_datetime(group[idx_col])).sort_index()
                history[str(symbol).upper()] = g[list(OHLCV)]

        history = validate_history(history)
        if start:
            history = {s: df.loc[df.index >= pd.Timestamp(start)] for s, df in history.items()}
        if end:
            history = {s: df.loc[df.index <= pd.Timestamp(end)] for s, df in history.items()}
        return validate_history(history)


# --------------------------------------------------------------------------
# Alpaca (US)
# --------------------------------------------------------------------------
def _default_http_get(url: str, headers: Mapping[str, str]) -> str:
    import requests  # imported lazily: only live runs need it

    resp = requests.get(url, headers=dict(headers), timeout=30)
    resp.raise_for_status()
    return resp.text


class AlpacaProvider(DataProvider):
    """Alpaca Markets historical bars. ``http_get`` is injectable for tests."""

    BASE = "https://data.alpaca.markets"

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        http_get: HttpGet = _default_http_get,
        feed: str = "iex",
    ) -> None:
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")
        self.http_get = http_get
        self.feed = feed

    def _headers(self) -> Dict[str, str]:
        return {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.secret_key}

    def load(self, symbols: Sequence[str], start: str, end: Optional[str] = None) -> History:
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Alpaca credentials missing (set ALPACA_API_KEY / ALPACA_SECRET_KEY)")

        symbols = list(symbols)
        history: History = {}
        next_token: Optional[str] = None
        while True:
            params = [f"start={pd.Timestamp(start).isoformat()}", "timeframe=1Day",
                      "adjustment=split", f"feed={self.feed}"]
            if end:
                params.append(f"end={pd.Timestamp(end).isoformat()}")
            if next_token:
                params.append(f"page_token={next_token}")
            url = f"{self.BASE}/v2/stocks/{','.join(symbols)}/bars?{'&'.join(params)}"

            payload = json.loads(self.http_get(url, self._headers()))
            for symbol, bars in (payload.get("bars") or {}).items():
                rows = [
                    {"date": pd.Timestamp(b["t"]), "open": float(b["o"]), "high": float(b["h"]),
                     "low": float(b["l"]), "close": float(b["c"]), "volume": float(b["v"])}
                    for b in bars
                ]
                if not rows:
                    continue
                frame = pd.DataFrame(rows).set_index("date").sort_index()
                history[symbol] = pd.concat([history[symbol], frame]) if symbol in history else frame

            next_token = payload.get("next_page_token")
            if not next_token:
                break

        if not history:
            raise RuntimeError("Alpaca returned no bars")
        return validate_history(history)


def available_symbols() -> List[str]:
    return list(UNIVERSE)


__all__ = [
    "DataProvider", "SyntheticProvider", "CSVProvider", "AlpacaProvider",
    "SECTORS", "available_symbols",
]
