"""Finance-tuned sentiment lexicon with negation, intensifiers and direction verbs.

Designed for Indian financial news headlines (ET / Moneycontrol / Reuters India
style English). Fast, deterministic, zero-dependency. Optionally upgraded to
FinBERT when `transformers` is installed (see README -> Upgrades).
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

# ---- direction words: weight × word --------------------------------------
POSITIVE = {
    "surge": 2.0, "surges": 2.0, "soar": 2.5, "soars": 2.5, "jump": 1.8, "jumps": 1.8,
    "rally": 2.0, "rallies": 2.0, "gain": 1.0, "gains": 1.0, "rise": 1.2, "rises": 1.2,
    "climb": 1.2, "climbs": 1.2, "beat": 1.5, "beats": 1.5, "upgrade": 2.2, "upgraded": 2.2,
    "outperform": 2.0, "record": 1.2, "high": 0.8, "profit": 1.5, "profits": 1.5,
    "growth": 1.2, "strong": 1.3, "robust": 1.5, "buy": 1.5, "accumulate": 1.3,
    "bullish": 2.2, "order win": 2.5, "orderwin": 2.5, "bagged": 1.8, "wins": 1.5,
    "dividend": 0.8, "bonus": 0.8, "expansion": 1.0, "recovery": 1.3, "boost": 1.4,
    "boosts": 1.4, "stellar": 1.8, "beats estimate": 2.5, "raises": 1.2, "guidance raise": 2.0,
    "affordable": 0.6, "reform": 1.0, "rate cut": 2.2, "cuts rates": 2.2,
}
NEGATIVE = {
    "fall": -1.2, "falls": -1.2, "drop": -1.5, "drops": -1.5, "plunge": -2.5,
    "plunges": -2.5, "slide": -1.5, "slides": -1.5, "slump": -2.2, "slumps": -2.2,
    "sink": -2.0, "sinks": -2.0, "crash": -2.8, "crashes": -2.8, "miss": -1.5,
    "misses": -1.5, "downgrade": -2.2, "downgraded": -2.2, "underperform": -1.8,
    "loss": -1.8, "losses": -1.8, "weak": -1.3, "poor": -1.3, "bearish": -2.2,
    "fraud": -3.0, "probe": -1.8, "investigation": -1.8, "sebi action": -2.8,
    "penalty": -1.8, "raid": -2.2, "stake sale": -1.2, "block deal": -0.8,
    "resign": -1.5, "resigns": -1.5, "layoff": -1.8, "layoffs": -1.8,
    "recall": -1.5, "litigation": -1.5, "default": -2.2, "guidance cut": -2.2,
    "cuts guidance": -2.2, "rate hike": -1.5, "hikes rates": -1.5, "decline": -1.0,
    "declines": -1.0, "low": -0.6, "warns": -1.6, "warning": -1.6, "sell": -1.2,
    "reduce": -1.0, "cut": -1.0, "cuts": -1.0, "delay": -1.0, "delays": -1.0,
}
INTENSIFIERS = {"sharply": 1.5, "steeply": 1.5, "massive": 1.6, "huge": 1.5,
                "record": 1.3, "all-time": 1.4, "surprise": 1.3, "unexpected": 1.3}
NEGATORS = {"not", "no", "never", "miss", "misses", "fails", "fail", "without", "despite"}

TOKEN_RE = re.compile(r"[a-zA-Z&\-]+")

# context overrides: crude/oil RISING is bad for Indian equities (import bill,
# inflation) even though "jumps/surges" are otherwise positive words.
# These run as multi-word phrases and consume their tokens.
CRUDE_UP = {
    "crude jump": -2.0, "crude jumps": -2.0, "crude surge": -2.0, "crude surges": -2.2,
    "crude spike": -2.2, "crude spikes": -2.2, "crude climb": -1.5, "crude climbs": -1.5,
    "crude rise": -1.5, "crude rises": -1.5, "oil jump": -2.0, "oil jumps": -2.0,
    "oil surge": -2.0, "oil surges": -2.2, "oil spike": -2.2, "oil spikes": -2.2,
    "oil rise": -1.5, "oil rises": -1.5, "crude gain": -1.2, "oil gain": -1.2,
}

# multi-word phrases must run before tokenisation
PHRASES = list(POSITIVE) + list(NEGATIVE) + list(CRUDE_UP)


def score_headline(headline: str) -> Tuple[float, Dict[str, float]]:
    """Return (score ∈ [-3, 3], breakdown) for one headline.

    Two passes:
      1. multi-word phrases with word boundaries ("order win", "rate cut",
         "beats estimate") — substring matching here would double-count
         ("fall" inside "falls"), so boundaries are mandatory
      2. single tokens with exact dict hits, negation ±2 tokens before and
         intensifiers on both sides ("falls sharply", "sharply falls")
    """
    text = headline.lower()
    breakdown: Dict[str, float] = {}

    # ---- phrase pass: multi-word only, word-boundary matched -------------
    phrase_score = 0.0
    consumed = set()
    for ph in PHRASES:
        if " " not in ph:
            continue                     # single words handled by token pass
        if re.search(r"\b" + re.escape(ph) + r"\b", text):
            # context overrides (crude/oil up = bearish) take priority,
            # then negatives, then positives
            w = CRUDE_UP.get(ph, NEGATIVE.get(ph, POSITIVE.get(ph, 0.0)))
            phrase_score += w
            breakdown[ph] = w
            consumed.update(ph.split())

    # ---- token pass --------------------------------------------------------
    tokens = TOKEN_RE.findall(text)
    score = 0.0
    for i, tok in enumerate(tokens):
        if tok in consumed:
            continue                     # already counted as part of a phrase
        w = POSITIVE.get(tok, NEGATIVE.get(tok, 0.0))
        if w == 0.0:
            continue
        # negation within 2 tokens before ("not up", "never beat")
        window_before = tokens[max(0, i - 2):i]
        if any(n in window_before for n in NEGATORS) and tok not in ("miss", "misses", "fails", "fail"):
            w *= -0.8
        # intensifier on either side ("sharply falls" / "falls sharply")
        window_near = tokens[max(0, i - 2):i] + tokens[i + 1:i + 3]
        if any(t in window_near for t in INTENSIFIERS):
            w *= 1.4
        score += w
        breakdown[tok] = w

    total = max(-3.0, min(3.0, phrase_score * 0.9 + score * 0.8))
    return total, breakdown


def aggregate(scores: list) -> float:
    """Weighted mean already applied by caller; here we just squash to [-1, 1]."""
    if not scores:
        return 0.0
    return max(-1.0, min(1.0, sum(scores) / len(scores) / 1.5))
