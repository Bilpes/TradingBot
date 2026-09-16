"""A static, network-free GICS-style universe (US).

The NSE universe used for Dhan paper trading lives in ``tradingbot/markets/nse.py``.
This one backs the original US-flavoured tests and the ``--market us`` CLI path.

Deliberately hardcoded: a universe that needs a network call cannot be tested in
an environment with no market-data egress. Swap in a live constituent list via
:func:`from_records` when running against a real data provider.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

from tradingbot.data.schema import SectorInfo

_RAW: List[tuple[str, str, str]] = [
    # Information Technology
    ("AAPL", "Information Technology", "Apple Inc."),
    ("MSFT", "Information Technology", "Microsoft Corp."),
    ("NVDA", "Information Technology", "NVIDIA Corp."),
    ("CRM", "Information Technology", "Salesforce Inc."),
    # Communication Services
    ("GOOGL", "Communication Services", "Alphabet Inc."),
    ("META", "Communication Services", "Meta Platforms Inc."),
    ("NFLX", "Communication Services", "Netflix Inc."),
    ("DIS", "Communication Services", "Walt Disney Co."),
    # Consumer Discretionary
    ("AMZN", "Consumer Discretionary", "Amazon.com Inc."),
    ("TSLA", "Consumer Discretionary", "Tesla Inc."),
    ("HD", "Consumer Discretionary", "Home Depot Inc."),
    ("NKE", "Consumer Discretionary", "Nike Inc."),
    # Consumer Staples
    ("PG", "Consumer Staples", "Procter & Gamble Co."),
    ("KO", "Consumer Staples", "Coca-Cola Co."),
    ("PEP", "Consumer Staples", "PepsiCo Inc."),
    ("WMT", "Consumer Staples", "Walmart Inc."),
    # Health Care
    ("UNH", "Health Care", "UnitedHealth Group Inc."),
    ("JNJ", "Health Care", "Johnson & Johnson"),
    ("LLY", "Health Care", "Eli Lilly and Co."),
    ("PFE", "Health Care", "Pfizer Inc."),
    # Financials
    ("JPM", "Financials", "JPMorgan Chase & Co."),
    ("BAC", "Financials", "Bank of America Corp."),
    ("BRK.B", "Financials", "Berkshire Hathaway Inc."),
    ("GS", "Financials", "Goldman Sachs Group Inc."),
    # Energy
    ("XOM", "Energy", "Exxon Mobil Corp."),
    ("CVX", "Energy", "Chevron Corp."),
    ("COP", "Energy", "ConocoPhillips"),
    ("SLB", "Energy", "Schlumberger Ltd."),
    # Industrials
    ("CAT", "Industrials", "Caterpillar Inc."),
    ("BA", "Industrials", "Boeing Co."),
    ("UNP", "Industrials", "Union Pacific Corp."),
    ("HON", "Industrials", "Honeywell International Inc."),
    # Utilities
    ("NEE", "Utilities", "NextEra Energy Inc."),
    ("DUK", "Utilities", "Duke Energy Corp."),
    ("SO", "Utilities", "Southern Co."),
    ("AEP", "Utilities", "American Electric Power Co."),
    # Real Estate
    ("PLD", "Real Estate", "Prologis Inc."),
    ("AMT", "Real Estate", "American Tower Corp."),
    ("EQIX", "Real Estate", "Equinix Inc."),
    ("SPG", "Real Estate", "Simon Property Group Inc."),
    # Materials
    ("LIN", "Materials", "Linde plc"),
    ("SHW", "Materials", "Sherwin-Williams Co."),
    ("FCX", "Materials", "Freeport-McMoRan Inc."),
    ("NEM", "Materials", "Newmont Corp."),
]

UNIVERSE: Dict[str, SectorInfo] = {s: SectorInfo(s, sec, n) for s, sec, n in _RAW}

SECTORS: List[str] = sorted({i.sector for i in UNIVERSE.values()})


def sector_of(symbol: str) -> str:
    """Return the sector for ``symbol``, or ``"Unknown"`` if unclassified."""
    info = UNIVERSE.get(symbol)
    return info.sector if info is not None else "Unknown"


def symbols_in_sector(sector: str) -> List[str]:
    return [s for s, i in UNIVERSE.items() if i.sector == sector]


def from_records(records: Iterable[tuple[str, str, str]]) -> Dict[str, SectorInfo]:
    """Build a universe from ``(symbol, sector, name)`` triples."""
    return {s: SectorInfo(s, sec, n) for s, sec, n in records}
