import pytest

from tradingbot.agents.base import Signal
from tradingbot.core.confidence import aggregate


def _sig(agent, score, conf, role="directional", symbol="AAPL", weight=None):
    return Signal(symbol=symbol, score=score, confidence=conf, agent=agent, role=role)


def test_unanimous_strong_votes_keep_confidence_high():
    out = aggregate([_sig("a", 0.9, 0.8), _sig("b", 0.85, 0.75), _sig("c", 0.95, 0.85)])
    cons = out["AAPL"]
    assert cons.score > 0.8
    assert cons.dispersion < 0.06
    assert cons.agreement > 0.9
    assert cons.confidence > 0.6


def test_disagreement_deflates_confidence_even_at_high_individual_conviction():
    agreeing = aggregate([_sig("a", 0.8, 0.9), _sig("b", 0.8, 0.9)])["AAPL"]
    split = aggregate([_sig("a", 0.8, 0.9), _sig("b", -0.8, 0.9)])["AAPL"]

    # A 50/50 split cancels the direction entirely...
    assert abs(split.score) < 1e-9
    # ...and conviction collapses to a fraction of the agreeing case.
    assert split.confidence < 0.25 * agreeing.confidence
    assert split.agreement < 0.25  # exp(-dispersion / tau) = exp(-1.6) ~= 0.20


def test_partial_disagreement_lands_between_the_extremes():
    aligned = aggregate([_sig("a", 0.8, 0.8), _sig("b", 0.6, 0.8), _sig("c", 0.7, 0.8)])["AAPL"]
    mixed = aggregate([_sig("a", 0.8, 0.8), _sig("b", -0.3, 0.8), _sig("c", 0.7, 0.8)])["AAPL"]
    assert mixed.confidence < aligned.confidence
    assert mixed.score > 0
    assert 0.2 < mixed.agreement < 0.95


def test_gate_multiplies_confidence_without_touching_direction():
    ungated = aggregate([_sig("a", 0.7, 0.8)])["AAPL"]
    gated = aggregate([_sig("a", 0.7, 0.8), _sig("vol", 0.0, 0.3, role="gate", symbol="*")])["AAPL"]
    assert gated.score == pytest.approx(ungated.score)
    assert gated.confidence == pytest.approx(ungated.confidence * 0.3)
    assert gated.gate == pytest.approx(0.3)


def test_multiple_gates_compound():
    out = aggregate(
        [
            _sig("a", 0.7, 0.8),
            _sig("vol", 0.0, 0.5, role="gate", symbol="*"),
            _sig("liq", 0.0, 0.4, role="gate", symbol="*"),
        ]
    )["AAPL"]
    assert out.gate == pytest.approx(0.2)


def test_voting_weights_shift_the_consensus():
    equal = aggregate([_sig("strong", 1.0, 0.9), _sig("weak", -1.0, 0.9)], weights={"strong": 1, "weak": 1})
    weighted = aggregate([_sig("strong", 1.0, 0.9), _sig("weak", -1.0, 0.9)], weights={"strong": 9, "weak": 1})
    assert abs(equal["AAPL"].score) < 1e-9
    assert weighted["AAPL"].score > 0.7


def test_confidence_is_bounded_zero_to_one():
    out = aggregate([_sig("a", 1.0, 1.0), _sig("b", 1.0, 1.0)])["AAPL"]
    assert 0.0 <= out.confidence <= 1.0
    assert -1.0 <= out.score <= 1.0


def test_breakdown_exposes_every_vote_for_audit():
    out = aggregate([_sig("a", 0.5, 0.6), _sig("b", -0.2, 0.4)])["AAPL"]
    assert set(out.breakdown) == {"a", "b"}
    assert out.breakdown["a"] == (0.5, 0.6)


def test_signal_rejects_out_of_range_values():
    with pytest.raises(ValueError):
        Signal(symbol="X", score=1.5, confidence=0.5, agent="a")
    with pytest.raises(ValueError):
        Signal(symbol="X", score=0.5, confidence=1.5, agent="a")
