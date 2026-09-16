"""Consensus: turning several disagreeing opinions into one number.

The important design decision here is what happens when agents disagree. The
obvious approach -- average the scores -- is wrong, because a unanimous weak
signal and a violently split strong signal both land near the same mean while
deserving completely different treatment.

So confidence is built from two separate quantities:

* **Level** (``score``) -- the confidence-weighted mean direction.
* **Cohesion** (``agreement``) -- ``exp(-dispersion / tau)``, where dispersion
  is the confidence-weighted spread of the votes around that mean.

Confidence is ``level-independent``: it is the mean confidence of the voters,
multiplied by cohesion, multiplied by any risk gates. Disagreement therefore
cannot be hidden by a large average -- it directly deflates conviction, which
the risk layer converts into a smaller position.

What confidence is *not*: a probability of profit. It is a bounded, calibrated
input to position sizing. Nothing downstream is allowed to read it as a
guarantee, and ``RiskConfig`` caps exposure regardless of how high it gets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, sqrt
from typing import Dict, Mapping, Sequence

from tradingbot.agents.base import Signal


@dataclass(frozen=True)
class Consensus:
    symbol: str
    score: float
    confidence: float
    agreement: float
    dispersion: float
    n_votes: int
    gate: float = 1.0
    breakdown: Mapping[str, tuple[float, float]] = field(default_factory=dict)


def aggregate(
    signals: Sequence[Signal],
    weights: Mapping[str, float] | None = None,
    tau: float = 0.5,
) -> Dict[str, Consensus]:
    """Fold one session's signals into a per-symbol consensus.

    ``weights`` maps agent name -> voting weight (defaults to 1.0). ``tau`` is
    the dispersion scale: smaller ``tau`` punishes disagreement more severely.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")

    weights = weights or {}
    directional: Dict[str, list[Signal]] = {}
    gate_product = 1.0
    gates: list[Signal] = []

    for sig in signals:
        if sig.role == "gate":
            gates.append(sig)
            gate_product *= sig.confidence
        else:
            directional.setdefault(sig.symbol, []).append(sig)

    out: Dict[str, Consensus] = {}
    for symbol, votes in directional.items():
        w = [max(float(weights.get(v.agent, 1.0)), 0.0) * max(v.confidence, 0.0) for v in votes]
        raw_w = [max(float(weights.get(v.agent, 1.0)), 0.0) for v in votes]
        s = [v.score for v in votes]

        w_sum = sum(w)
        raw_sum = sum(raw_w)
        if w_sum <= 1e-12 or raw_sum <= 1e-12:
            continue

        score = sum(wi * si for wi, si in zip(w, s)) / w_sum
        variance = sum(wi * (si - score) ** 2 for wi, si in zip(w, s)) / w_sum
        dispersion = sqrt(max(variance, 0.0))
        agreement = exp(-dispersion / tau)

        base_conf = sum(rw * v.confidence for rw, v in zip(raw_w, votes)) / raw_sum
        confidence = min(max(base_conf * agreement * gate_product, 0.0), 1.0)

        out[symbol] = Consensus(
            symbol=symbol,
            score=min(max(score, -1.0), 1.0),
            confidence=confidence,
            agreement=agreement,
            dispersion=dispersion,
            n_votes=len(votes),
            gate=gate_product,
            breakdown={v.agent: (v.score, v.confidence) for v in votes},
        )

    # Expose the gate even when no directional votes survived, so callers can
    # log why the system stood down.
    if not out and gates:
        out["*"] = Consensus(
            symbol="*", score=0.0, confidence=gate_product, agreement=1.0,
            dispersion=0.0, n_votes=0, gate=gate_product,
        )
    return out
