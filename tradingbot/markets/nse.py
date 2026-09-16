"""NSE universe: symbols grouped by NSE sectoral index.

Deliberately contains **no Dhan security IDs**. A wrong security id means
trading the wrong stock, which is a far worse failure than a startup error, so
IDs are resolved at runtime from Dhan's official scrip master
(``dhan_scrip_master.csv``) via :func:`tradingbot.data.dhan.resolve_security_ids`
and the system refuses to run if any symbol fails to resolve.

Generate the map with:

    python scripts/make_security_map.py --scrip-master dhan_scrip_master.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class NSEInstrument:
    symbol: str          # NSE trading symbol, as used by Dhan
    sector: str          # NSE sectoral index name
    name: str


_RAW: List[tuple[str, str, str]] = [
    # NIFTY Bank / Financial Services
    ("HDFCBANK", "Banking", "HDFC Bank Ltd."),
    ("ICICIBANK", "Banking", "ICICI Bank Ltd."),
    ("SBIN", "Banking", "State Bank of India"),
    ("KOTAKBANK", "Banking", "Kotak Mahindra Bank Ltd."),
    ("AXISBANK", "Banking", "Axis Bank Ltd."),
    ("BAJFINANCE", "Financial Services", "Bajaj Finance Ltd."),
    ("HDFCLIFE", "Financial Services", "HDFC Life Insurance"),
    ("CHOLAFIN", "Financial Services", "Cholamandalam Investment"),
    # NIFTY IT
    ("TCS", "IT", "Tata Consultancy Services"),
    ("INFY", "IT", "Infosys Ltd."),
    ("HCLTECH", "IT", "HCL Technologies Ltd."),
    ("WIPRO", "IT", "Wipro Ltd."),
    ("TECHM", "IT", "Tech Mahindra Ltd."),
    # NIFTY Auto
    ("MARUTI", "Auto", "Maruti Suzuki India Ltd."),
    ("TATAMOTORS", "Auto", "Tata Motors Ltd."),
    ("M&M", "Auto", "Mahindra & Mahindra Ltd."),
    ("BAJAJ-AUTO", "Auto", "Bajaj Auto Ltd."),
    ("EICHERMOT", "Auto", "Eicher Motors Ltd."),
    # NIFTY Pharma
    ("SUNPHARMA", "Pharma", "Sun Pharmaceutical Industries"),
    ("DRREDDY", "Pharma", "Dr. Reddy's Laboratories"),
    ("CIPLA", "Pharma", "Cipla Ltd."),
    ("DIVISLAB", "Pharma", "Divi's Laboratories Ltd."),
    # NIFTY FMCG
    ("HINDUNILVR", "FMCG", "Hindustan Unilever Ltd."),
    ("ITC", "FMCG", "ITC Ltd."),
    ("NESTLEIND", "FMCG", "Nestle India Ltd."),
    ("BRITANNIA", "FMCG", "Britannia Industries Ltd."),
    ("DABUR", "FMCG", "Dabur India Ltd."),
    # NIFTY Energy / Oil & Gas
    ("RELIANCE", "Energy", "Reliance Industries Ltd."),
    ("ONGC", "Energy", "Oil and Natural Gas Corp."),
    ("COALINDIA", "Energy", "Coal India Ltd."),
    ("BPCL", "Energy", "Bharat Petroleum Corp."),
    # NIFTY Metal
    ("TATASTEEL", "Metal", "Tata Steel Ltd."),
    ("HINDALCO", "Metal", "Hindalco Industries Ltd."),
    ("JSWSTEEL", "Metal", "JSW Steel Ltd."),
    ("VEDL", "Metal", "Vedanta Ltd."),
    # NIFTY Realty / Infra
    ("DLF", "Realty", "DLF Ltd."),
    ("GODREJPROP", "Realty", "Godrej Properties Ltd."),
    ("LT", "Infra", "Larsen & Toubro Ltd."),
    ("ULTRACEMCO", "Infra", "UltraTech Cement Ltd."),
    ("GRASIM", "Infra", "Grasim Industries Ltd."),
    # NIFTY Consumer Durables / Telecom
    ("TITAN", "Consumer Durables", "Titan Company Ltd."),
    ("ASIANPAINT", "Consumer Durables", "Asian Paints Ltd."),
    ("HAVELLS", "Consumer Durables", "Havells India Ltd."),
    ("BHARTIARTL", "Telecom", "Bharti Airtel Ltd."),
    ("IDEA", "Telecom", "Vodafone Idea Ltd."),
]

NSE_UNIVERSE: Dict[str, NSEInstrument] = {s: NSEInstrument(s, sec, n) for s, sec, n in _RAW}

NSE_SECTORS: List[str] = sorted({i.sector for i in NSE_UNIVERSE.values()})


def nse_sector_of(symbol: str) -> str:
    inst = NSE_UNIVERSE.get(symbol.upper())
    return inst.sector if inst is not None else "Other"


def nse_symbols_in_sector(sector: str) -> List[str]:
    return [s for s, i in NSE_UNIVERSE.items() if i.sector == sector]


def sector_symbol_counts() -> Dict[str, int]:
    out: Dict[str, int] = {}
    for inst in NSE_UNIVERSE.values():
        out[inst.sector] = out.get(inst.sector, 0) + 1
    return out
