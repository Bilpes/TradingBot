"""Instrument master: symbol → Dhan securityId mapping.

Dhan publishes a public scrip-master CSV (no auth needed):
    https://images.dhan.co/api-data/api-scrip-master.csv
We download it, cache it locally (~1 day TTL) and build a NSE-EQUITY map.
This is the *only* reliable way to get securityIds — never hardcode them.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

from ..config import data_dir
from ..logger import get_logger

log = get_logger("instruments")

SCRIP_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
CACHE_TTL_SEC = 24 * 3600


class InstrumentMaster:
    def __init__(self):
        self._df: Optional[pd.DataFrame] = None
        self._symbol_map: Dict[str, str] = {}      # SYMBOL -> securityId (NSE_EQ)
        self._loaded_at: float = 0.0
        self._cache_file = data_dir() / "cache" / "scrip_master.csv"

    # ------------------------------------------------------------------
    def _download(self) -> bool:
        try:
            r = requests.get(SCRIP_MASTER_URL, timeout=30)
            r.raise_for_status()
            self._cache_file.write_bytes(r.content)
            return True
        except Exception as e:  # network blocked etc.
            log.warning("scrip master download failed: %s", e)
            return False

    def _load(self, force: bool = False) -> bool:
        now = time.time()
        if self._df is not None and not force and now - self._loaded_at < CACHE_TTL_SEC:
            return True

        fresh = self._cache_file.exists() and (now - self._cache_file.stat().st_mtime < CACHE_TTL_SEC)
        if force or not fresh:
            self._download()

        if not self._cache_file.exists():
            log.error("no scrip master available — securityId lookups will fail")
            return False

        try:
            df = pd.read_csv(self._cache_file, low_memory=False)
        except Exception:
            # cache corrupted → re-download once
            if self._download():
                df = pd.read_csv(self._cache_file, low_memory=False)
            else:
                return False

        # normalise column names defensively (master format has varied over time)
        cols = {c.upper().strip(): c for c in df.columns}
        exch = cols.get("SEM_EXM_EXCH_ID")
        seg = cols.get("SEM_SEGMENT")
        secid = cols.get("SEM_SMST_SECURITY_ID") or cols.get("SEM_SECURITY_ID")
        sym = cols.get("SEM_TRADING_SYMBOL") or cols.get("SEM_SM_SYMBOL_NAME")
        inst = cols.get("SEM_INSTRUMENT_NAME")
        if not all([exch, secid, sym]):
            log.error("scrip master columns unexpected: %s", list(df.columns)[:10])
            return False

        df = df[df[exch].astype(str).str.upper() == "NSE"]
        if inst is not None:
            df = df[df[inst].astype(str).str.upper().str.contains("EQUITY")]
        # drop expired/duplicate symbols: keep the row with the most recent expiry column
        df = df.dropna(subset=[secid])
        df[sym] = df[sym].astype(str).str.upper().str.strip()
        df = df.sort_values(cols.get("SEM_EXPIRY_DATE", sym)).drop_duplicates(subset=[sym], keep="last")

        self._df = df
        self._symbol_map = dict(zip(df[sym], df[secid].astype(str)))
        self._loaded_at = now
        log.info("instrument master loaded: %d NSE equity symbols", len(self._symbol_map))
        return True

    # ------------------------------------------------------------------
    def security_id(self, symbol: str) -> Optional[str]:
        """symbol like 'RELIANCE', 'M&M', 'BAJAJ-AUTO' → Dhan securityId."""
        if not self._load():
            return None
        s = symbol.upper().strip()
        if s in self._symbol_map:
            return self._symbol_map[s]
        # small normalisation attempts (ampersand encodings differ across feeds)
        for cand in (s.replace("&", "%26"), s.replace("&", "_"), s.replace(" ", "")):
            if cand in self._symbol_map:
                return self._symbol_map[cand]
        return None

    # index securityIds (stable constants from Dhan docs)
    INDEX_IDS = {"NIFTY": "13", "BANKNIFTY": "25", "FINNIFTY": "27", "SENSEX": "51"}


INSTRUMENTS = InstrumentMaster()
