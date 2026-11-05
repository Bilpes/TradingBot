"""Sentiment agent (NLP) — news-flow vote per symbol + market-wide context.

Hard veto material: severe negative flow (score < −0.6 with ≥3 negative
headlines) or regulatory events → the Strategy layer blocks fresh entries.
"""

from __future__ import annotations

import numpy as np

from ..models import AgentReport
from .base import BaseAgent, MarketContext


class SentimentAgent(BaseAgent):
    name = "Sentiment"
    description = "RSS headline NLP: lexicon + negation + recency weighting"

    def run(self, ctx: MarketContext) -> AgentReport:
        s_news = ctx.sentiment or {}
        m_news = ctx.market_sentiment or {}

        sym_score = s_news.get("score", 0.0)
        mkt_score = m_news.get("score", 0.0)
        items = s_news.get("items", 0)
        negatives = s_news.get("negatives", [])
        events = s_news.get("events", {})

        score = float(np.clip(sym_score * 70 + mkt_score * 25, -100, 100))

        # event flags
        if events.get("regulatory"):
            score -= 35
        if events.get("results_today"):
            score -= 10            # event risk → keep flat until results print

        veto = bool(sym_score < -0.6 and len(negatives) >= 3)
        if veto:
            score = min(score, -80)

        note_bits = [f"sym {sym_score:+.2f} ({items} hl)", f"mkt {mkt_score:+.2f}"]
        if negatives:
            note_bits.append(f"{len(negatives)} negative")
        note = " | ".join(note_bits)

        return AgentReport(
            agent=self.name, score=score,
            confidence=min(0.8, 0.3 + items * 0.05),
            note=note,
            data={
                "symbol_score": sym_score, "market_score": mkt_score,
                "items": items, "veto": veto,
                "negatives": negatives[:2],
                "events": events,
            })
