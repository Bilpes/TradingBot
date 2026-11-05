"""News ingestion + sentiment scoring via free RSS feeds.

Feeds (no API key required):
  - Google News RSS per symbol:  <symbol> stock — en-IN edition
  - Google News RSS market-wide: Nifty / Sensex / Indian stock market

Every headline gets: lexicon score × recency weight (≤24h:1.0, ≤48h:0.6,
else 0.3). The Sentiment agent converts these into a per-symbol score, a
market score, and hard event flags (results today, SEBI action, raids...).
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple
from urllib.parse import quote

import feedparser

from ..config import data_dir
from ..logger import get_logger
from .lexicon import score_headline, aggregate

log = get_logger("news")
IST = timezone(timedelta(hours=5, minutes=30))

GOOGLE_NEWS = "https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"

# cache TTL: don't hammer RSS more than every 10 min per query
_ttl: Dict[str, Tuple[float, list]] = {}


def _fetch(query: str, max_items: int = 12) -> List[dict]:
    key = query.lower()
    ts, cached = _ttl.get(key, (0.0, []))
    if cached and time.time() - ts < 600:
        return cached
    url = GOOGLE_NEWS.format(q=quote(query))
    try:
        d = feedparser.parse(url)
        items = []
        for e in d.entries[:max_items]:
            title = re.sub(r"\s+", " ", getattr(e, "title", "")).strip()
            # struct_time is the reliable public API (newer feedparser versions
            # removed module-level _parse_date)
            published = getattr(e, "published_parsed", None)
            items.append({"title": title, "published_parsed": published})
        _ttl[key] = (time.time(), items)
        return items
    except Exception as e:
        log.warning("rss fetch failed for %r: %s", query, e)
        return []


def _age_hours(published_parsed) -> float:
    """Age of a headline in hours; unknown age → assume stale (24h)."""
    if published_parsed is None:
        return 24.0
    try:
        dt = datetime(*published_parsed[:6], tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)
    except Exception:
        return 24.0


def _weighted_sentiment(items: List[dict]) -> Tuple[float, int, List[str]]:
    """Returns (score ∈ [-1,1], n_items, negative_headlines)."""
    if not items:
        return 0.0, 0, []
    weighted: List[float] = []
    negatives: List[str] = []
    for it in items:
        s, _ = score_headline(it["title"])
        age = _age_hours(it.get("published_parsed"))
        w = 1.0 if age <= 24 else (0.6 if age <= 48 else 0.3)
        weighted.append(s * w)
        if s < -0.8 and len(negatives) < 5:
            negatives.append(it["title"])
    return aggregate(weighted), len(items), negatives


EVENT_PATTERNS = {
    "results_today": r"(q[1-4] results|quarterly results|earnings (today|tomorrow)|board meeting)",
    "regulatory": r"(sebi|enforcement directorate|raid|probe|penalt|fine imposed|investigation)",
    "m_a": r"(acquisition|acquire|merger|stake (sale|buy)|block deal)",
    "guidance": r"(guidance|outlook|forecast)",
}


def symbol_news(symbol: str) -> dict:
    """News + sentiment for one NSE symbol."""
    items = _fetch(f"{symbol} stock India", max_items=10)
    score, n, negatives = _weighted_sentiment(items)
    text = " ".join(i["title"].lower() for i in items)
    events = {k: bool(re.search(p, text)) for k, p in EVENT_PATTERNS.items()}
    return {"symbol": symbol, "score": round(score, 3), "items": n,
            "negatives": negatives, "events": events}


def market_news() -> dict:
    items = _fetch("Nifty Sensex Indian stock market", max_items=14)
    score, n, negatives = _weighted_sentiment(items)
    return {"score": round(score, 3), "items": n, "negatives": negatives}
