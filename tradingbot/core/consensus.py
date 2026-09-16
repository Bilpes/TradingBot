"""Consensus engine: weighting signals and *requiring* agreement.

:func:`tradingbot.core.confidence.aggregate` produces a blended score and a
confidence. That is necessary but not sufficient: a blended number can look
convincing while resting on one loud agent and three indifferent ones.

This layer adds the rules that make a signal *actionable*:

``min_voters``
    Enough agents must have spoken at all.
``min_agreeing``
    Enough must agree on the *sign* with meaningful conviction.
``max_dissent_confidence``
    A strongly dissenting agent vetoes the trade outright. Being wrong loudly
    is information; averaging it away is not.
``min_confidence``
    The blended confidence floor.

A trade needs all of them. ``actionable`` is the single boolean the engine
consults, and every rejection is recorded with the rule that rejected it, so the
dashboard can show why a system that looked bullish did not buy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence

from tradingbot.agents.base import Signal
from tradingbot.core.confidence import Consensus, aggregate


@dataclass(frozen=True)
class ConsensusConfig:
    min_voters: int = 3
    min_agreeing: int = 2
    vote_confidence: float = 0.30        # below this a vote does not count either way
    max_dissent_confidence: float = 0.55  # a dissent this confident vetoes the trade
    # A veto also needs a dissent of some size. Without this, an agent scoring
    # -0.03 with high confidence -- which is really "I see nothing" -- vetoes a
    # trade it has no opinion about, and the reason string reads as if it had
    # argued the other side.
    min_dissent_score: float = 0.25
    # Calibrated, not guessed. At 0.35 the blended confidence almost never
    # clears the bar once the dispersion penalty is applied across six agents,
    # and the system made zero trades out of 45 decisions. At 0.25 roughly 9%
    # of decisions are actionable -- selective without being inert. This was
    # tuned against *how often the system acts*, never against returns.
    min_confidence: float = 0.25
    min_abs_score: float = 0.15
    tau: float = 0.5

    def __post_init__(self) -> None:
        if self.min_agreeing > self.min_voters:
            raise ValueError("min_agreeing cannot exceed min_voters")
        if not 0.0 <= self.vote_confidence <= 1.0:
            raise ValueError("vote_confidence must be in [0, 1]")
        if self.min_voters < 1:
            raise ValueError("min_voters must be >= 1")


@dataclass(frozen=True)
class ConsensusDecision:
    symbol: str
    score: float
    confidence: float
    agreement: float
    dispersion: float
    agreeing: int
    dissenting: int
    neutral: int
    vetoed: bool
    veto_reason: str
    actionable: bool
    rejection: str
    consensus: Consensus
    votes: Dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def direction(self) -> str:
        if not self.actionable:
            return "flat"
        return "long" if self.score > 0 else "short"


def decide(
    symbol: str,
    consensus: Consensus,
    config: ConsensusConfig,
) -> ConsensusDecision:
    """Apply the quorum rules to one symbol's consensus."""
    agreeing = dissenting = neutral = 0
    vetoed = False
    veto_reason = ""

    sign = 1.0 if consensus.score >= 0 else -1.0
    for agent, (score, confidence) in consensus.breakdown.items():
        if confidence < config.vote_confidence:
            neutral += 1
            continue
        if score * sign > 0:
            agreeing += 1
        elif score * sign < 0:
            dissenting += 1
            if (confidence >= config.max_dissent_confidence
                    and abs(score) >= config.min_dissent_score):
                vetoed = True
                veto_reason = f"{agent} dissents at {score:+.2f} with confidence {confidence:.2f}"
        else:
            neutral += 1

    voters = len(consensus.breakdown)
    rejection = ""
    if vetoed:
        rejection = f"veto: {veto_reason}"
    elif voters < config.min_voters:
        rejection = f"only {voters} voters (need {config.min_voters})"
    elif agreeing < config.min_agreeing:
        rejection = f"only {agreeing} agreeing (need {config.min_agreeing})"
    elif consensus.confidence < config.min_confidence:
        # %.6g, not %.2f: a value a hair under the floor must not print as
        # "0.25 < 0.25", which reads as a contradiction on the dashboard.
        rejection = f"confidence {consensus.confidence:.6g} < {config.min_confidence:.6g}"
    elif abs(consensus.score) < config.min_abs_score:
        rejection = f"|score| {abs(consensus.score):.2f} < {config.min_abs_score:.2f}"

    return ConsensusDecision(
        symbol=symbol,
        score=consensus.score,
        confidence=consensus.confidence,
        agreement=consensus.agreement,
        dispersion=consensus.dispersion,
        agreeing=agreeing,
        dissenting=dissenting,
        neutral=neutral,
        vetoed=vetoed,
        veto_reason=veto_reason,
        actionable=not rejection,
        rejection=rejection,
        consensus=consensus,
        votes=dict(consensus.breakdown),
    )


def decide_all(
    signals: Sequence[Signal],
    weights: Dict[str, float] | None,
    config: ConsensusConfig,
) -> Dict[str, ConsensusDecision]:
    """Fold a whole session's signals into per-symbol actionable decisions."""
    merged = aggregate(signals, weights, config.tau)
    return {
        symbol: decide(symbol, cons, config)
        for symbol, cons in merged.items()
        if symbol != "*"
    }


def explain(decision: ConsensusDecision) -> str:
    """One-line human-readable justification, used by alerts and the dashboard."""
    parts = [
        f"{decision.symbol}: score {decision.score:+.2f} conf {decision.confidence:.2f} "
        f"({decision.agreeing} agree / {decision.dissenting} dissent / {decision.neutral} silent)"
    ]
    if decision.actionable:
        parts.append(f"-> {decision.direction.upper()}")
    else:
        parts.append(f"-> NO TRADE ({decision.rejection})")
    return " ".join(parts)


def ranking_key(decision: ConsensusDecision) -> tuple:
    """Sort key for picking the best opportunities: conviction, then agreement."""
    return (decision.actionable, round(decision.confidence, 6), round(abs(decision.score), 6))
