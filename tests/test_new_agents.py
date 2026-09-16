import numpy as np
import pandas as pd
import pytest

from tradingbot.agents.macd import MACDAgent
from tradingbot.agents.moving_average import MovingAverageAgent
from tradingbot.agents.registry import (
    AGENT_TYPES, DEFAULT_PROFILES, AgentSpec, dump_profiles, load_profiles,
    profile_for, profile_for_symbol,
)
from tradingbot.agents.rsi import RSIAgent
from tradingbot.agents.sentiment import (
    EarningsCalendar, SentimentAgent, StaticSentimentSource,
)
from tradingbot.agents.volume import VolumeSpikeAgent
from tradingbot.core.consensus import ConsensusConfig, decide, decide_all
from tradingbot.markets.nse import NSE_SECTORS
from tradingbot.core.confidence import aggregate


def _frame(drift=0.0, vol=0.012, n=260, p0=100.0, seed=0, vol_base=1e6, vol_last=None):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    close = p0 * np.exp(np.cumsum(r))
    volume = np.full(n, vol_base) * np.exp(rng.normal(0, 0.15, n))
    if vol_last:
        volume[-1] = vol_last
    dates = pd.bdate_range("2020-01-02", periods=n)
    prev = np.concatenate([[p0], close[:-1]])
    return pd.DataFrame({
        "open": prev, "high": np.maximum(prev, close) * 1.004,
        "low": np.minimum(prev, close) * 0.996, "close": close, "volume": volume,
    }, index=dates)


@pytest.fixture
def hist():
    return {"AAA": _frame(drift=0.002, seed=1), "BBB": _frame(drift=-0.001, seed=2)}


def _agent_signals_at(agent, history, date=None):
    agent.fit(history)
    date = date if date is not None else max(df.index[-1] for df in history.values())
    return {s.symbol: s for s in agent.signals_on(date)}


def _last_agent_signals(agent, history):
    return _agent_signals_at(agent, history)


# -- new agents -------------------------------------------------------------
@pytest.mark.parametrize("cls", [RSIAgent, MACDAgent, MovingAverageAgent])
def test_new_agents_emit_bounded_signals(cls, hist):
    signals = _last_agent_signals(cls(), hist)
    assert signals, f"{cls.__name__} produced nothing"
    for sig in signals.values():
        assert -1.0 <= sig.score <= 1.0
        assert 0.0 <= sig.confidence <= 1.0


@pytest.mark.parametrize("cls", [RSIAgent, MACDAgent, MovingAverageAgent, VolumeSpikeAgent])
def test_new_agents_are_causal(cls, hist):  # noqa: F811 - parametrised over all four
    """Same question, same date, with and without the future present.

    Checked across a range of dates rather than one bar: an agent that is
    legitimately silent on most bars (the volume agent only speaks on a spike)
    would otherwise make this test pass or fail on which bar was picked.
    """
    cut = hist["AAA"].index[220]
    truncated = {s: df.loc[:cut] for s, df in hist.items()}
    full, short = cls(), cls()
    full.fit(hist)
    short.fit(truncated)

    dates = hist["AAA"].index[150:221]
    spoke = 0
    for date in dates:
        a = {s.symbol: s for s in full.signals_on(date)}
        b = {s.symbol: s for s in short.signals_on(date)}
        assert set(a) == set(b), f"{cls.__name__} changed which symbols it covers on {date}"
        for symbol in a:
            assert a[symbol].score == pytest.approx(b[symbol].score, abs=1e-12)
            assert a[symbol].confidence == pytest.approx(b[symbol].confidence, abs=1e-12)
        spoke += len(a)
    assert spoke > 0, f"{cls.__name__} never spoke across {len(dates)} sessions"


def test_moving_average_is_bullish_in_a_clean_uptrend():
    signals = _last_agent_signals(MovingAverageAgent(fast=20, medium=50, slow=200),
                                  {"UP": _frame(drift=0.004, vol=0.004, n=260, seed=9)})
    assert signals["UP"].score > 0.5


def test_volume_agent_stays_silent_without_a_spike(hist):
    """Silence is a valid answer; a fabricated vote is not."""
    signals = _last_agent_signals(VolumeSpikeAgent(), hist)
    assert all(-1.0 <= s.score <= 1.0 and 0.0 <= s.confidence <= 1.0
               for s in signals.values())


def test_volume_spike_reads_direction_from_price_not_volume():
    """A spike on a down day must vote negative -- the classic naive mistake."""
    falling = {"DOWN": _frame(drift=-0.006, vol=0.005, n=200, seed=5, vol_last=20e6)}
    rising = {"UP": _frame(drift=0.006, vol=0.005, n=200, seed=5, vol_last=20e6)}
    down = _last_agent_signals(VolumeSpikeAgent(), falling)["DOWN"]
    up = _last_agent_signals(VolumeSpikeAgent(), rising)["UP"]
    assert down.score < 0 < up.score


def test_rsi_agent_validates_its_thresholds():
    with pytest.raises(ValueError):
        RSIAgent(overbought=30, oversold=70)


# -- registry ---------------------------------------------------------------
def test_every_sector_profile_builds_real_agents():
    for sector, profile in DEFAULT_PROFILES.items():
        agents = profile.build_agents()
        assert agents, f"{sector} profile builds nothing"
        names = {a.name for a in agents}
        assert "sector_rotation" in names and "vol_regime" in names, sector


def test_all_universe_sectors_have_a_profile():
    missing = [s for s in NSE_SECTORS if s not in DEFAULT_PROFILES]
    assert not missing, f"no profile for {missing}"


def test_profiles_differ_by_sector():
    banking = {s.agent: s.weight for s in DEFAULT_PROFILES["Banking"].agents}
    fmcg = {s.agent: s.weight for s in DEFAULT_PROFILES["FMCG"].agents}
    assert banking != fmcg
    assert banking["macd"] > fmcg["macd"]   # banks trend, so MACD carries more
    assert fmcg["rsi"] > banking["rsi"]     # FMCG fades, so RSI carries more
    assert "mean_reversion" not in banking  # contrarian agents are left out on purpose


def test_profile_for_symbol_resolves_through_the_universe():
    assert profile_for_symbol("HDFCBANK").sector == "Banking"
    assert profile_for_symbol("TCS").sector == "IT"
    assert profile_for("Unknown Sector").sector == "Unknown Sector"


def test_profiles_round_trip_through_json(tmp_path):
    out = tmp_path / "profiles.json"
    dump_profiles(out)
    loaded = load_profiles(out)
    assert set(loaded) == set(DEFAULT_PROFILES)
    assert loaded["Metal"].agents[0].agent == DEFAULT_PROFILES["Metal"].agents[0].agent


def test_load_profiles_rejects_an_unknown_agent(tmp_path):
    fp = tmp_path / "bad.json"
    fp.write_text('{"Banking": {"agents": [{"agent": "not_a_real_agent"}]}}')
    with pytest.raises(ValueError, match="unknown agent"):
        load_profiles(fp)


def test_agent_spec_rejects_unknown_type():
    with pytest.raises(ValueError):
        AgentSpec(agent="nope")


def test_every_registered_agent_type_is_constructible():
    for name, cls in AGENT_TYPES.items():
        if name == "sentiment":
            continue  # needs a source
        assert cls() is not None, name


# -- consensus / quorum -----------------------------------------------------
def _sig(agent, score, conf):
    from tradingbot.agents.base import Signal

    return Signal(symbol="X", score=score, confidence=conf, agent=agent)


def test_quorum_blocks_a_single_loud_agent():
    cons = aggregate([_sig("a", 0.9, 0.95)])["X"]
    decision = decide("X", cons, ConsensusConfig(min_voters=3, min_agreeing=2))
    assert not decision.actionable
    assert "voters" in decision.rejection


def test_quorum_requires_agreement_not_just_votes():
    # "b" dissents below the veto threshold and "c" is too unsure to count as a
    # vote at all, so only one agent actually agrees -- the agreement rule binds.
    cons = aggregate([_sig("a", 0.8, 0.8), _sig("b", -0.5, 0.45), _sig("c", 0.1, 0.2)])["X"]
    decision = decide("X", cons, ConsensusConfig(min_voters=3, min_agreeing=2,
                                                 max_dissent_confidence=0.55))
    assert not decision.vetoed
    assert decision.agreeing == 1 and decision.dissenting == 1 and decision.neutral == 1
    assert not decision.actionable
    assert "agreeing" in decision.rejection


def test_a_strong_dissenter_vetoes_an_otherwise_good_signal():
    cons = aggregate([_sig("a", 0.8, 0.8), _sig("b", 0.7, 0.75),
                      _sig("c", 0.6, 0.7), _sig("bear", -0.9, 0.9)])["X"]
    decision = decide("X", cons, ConsensusConfig(max_dissent_confidence=0.55))
    assert decision.vetoed
    assert not decision.actionable
    assert "bear" in decision.veto_reason


def test_a_weak_dissenter_does_not_veto():
    cons = aggregate([_sig("a", 0.8, 0.8), _sig("b", 0.7, 0.75),
                      _sig("c", 0.6, 0.7), _sig("d", -0.2, 0.35)])["X"]
    decision = decide("X", cons, ConsensusConfig(max_dissent_confidence=0.55))
    assert not decision.vetoed


def test_a_clean_unanimous_signal_is_actionable():
    cons = aggregate([_sig("a", 0.8, 0.8), _sig("b", 0.75, 0.75),
                      _sig("c", 0.85, 0.8)])["X"]
    decision = decide("X", cons, ConsensusConfig())
    assert decision.actionable
    assert decision.direction == "long"
    assert decision.agreeing == 3 and decision.dissenting == 0
    assert decision.rejection == ""


def test_confidence_floor_is_enforced():
    cons = aggregate([_sig("a", 0.3, 0.4), _sig("b", 0.3, 0.4), _sig("c", 0.3, 0.4)])["X"]
    decision = decide("X", cons, ConsensusConfig(min_confidence=0.95))
    assert not decision.actionable
    assert "confidence" in decision.rejection


def test_decide_all_skips_the_market_wide_gate_symbol():
    from tradingbot.agents.base import Signal

    signals = [_sig("a", 0.8, 0.8), _sig("b", 0.7, 0.7), _sig("c", 0.75, 0.75),
               Signal(symbol="*", score=0.0, confidence=0.9, agent="vol_regime", role="gate")]
    decisions = decide_all(signals, None, ConsensusConfig())
    assert "*" not in decisions
    assert "X" in decisions


def test_consensus_config_validation():
    with pytest.raises(ValueError):
        ConsensusConfig(min_voters=2, min_agreeing=5)
    with pytest.raises(ValueError):
        ConsensusConfig(min_voters=0)


# -- sentiment --------------------------------------------------------------
def test_static_sentiment_source_decays_to_nothing(tmp_path):
    fp = tmp_path / "sent.csv"
    fp.write_text("symbol,date,score,note\nHDFCBANK,2022-01-03,0.8,guidance raise\n")
    src = StaticSentimentSource(fp, decay_days=10)
    assert src.score("HDFCBANK", pd.Timestamp("2022-01-03")) == pytest.approx(0.8)
    assert src.score("HDFCBANK", pd.Timestamp("2022-01-08")) == pytest.approx(0.4)
    assert src.score("HDFCBANK", pd.Timestamp("2022-02-01")) is None
    assert src.score("UNKNOWN", pd.Timestamp("2022-01-03")) is None


def test_earnings_calendar_haircuts_conviction_around_the_print():
    cal = EarningsCalendar({"HDFCBANK": ["2022-01-15"]}, window_days=3, min_multiplier=0.3)
    assert cal.multiplier("HDFCBANK", pd.Timestamp("2022-01-15")) == pytest.approx(0.3)
    assert cal.multiplier("HDFCBANK", pd.Timestamp("2022-01-18")) == pytest.approx(1.0)
    assert cal.multiplier("HDFCBANK", pd.Timestamp("2022-03-01")) == pytest.approx(1.0)
    assert cal.multiplier("OTHER", pd.Timestamp("2022-01-15")) == pytest.approx(1.0)


def test_sentiment_agent_stays_silent_without_a_source():
    agent = SentimentAgent()
    agent.fit({"HDFCBANK": _frame(n=60)})
    assert agent.signals_on(pd.Timestamp("2020-06-01")) == []


def test_sentiment_agent_emits_a_gate_when_only_a_calendar_is_given():
    dates = pd.bdate_range("2020-01-02", periods=60)
    agent = SentimentAgent(calendar=EarningsCalendar({"HDFCBANK": [str(dates[40].date())]}))
    agent.fit({"HDFCBANK": _frame(n=60)})
    signals = agent.signals_on(dates[40])
    assert len(signals) == 1
    assert signals[0].role == "gate"
    assert signals[0].confidence > 0


def test_sentiment_agent_uses_the_source_when_present(tmp_path):
    fp = tmp_path / "s.json"
    fp.write_text('[{"symbol": "HDFCBANK", "date": "2020-03-20", "score": 0.9}]')
    agent = SentimentAgent(source=StaticSentimentSource(fp, decay_days=30))
    agent.fit({"HDFCBANK": _frame(n=60)})
    signals = agent.signals_on(pd.Timestamp("2020-03-20"))
    assert signals and signals[0].score == pytest.approx(0.9)
    # confidence is |score| capped at 0.85, so 0.9 saturates.
    assert signals[0].confidence == pytest.approx(0.85)


def test_a_zero_opinion_cannot_veto():
    """'-0.03 with high confidence' is not a dissent; it is an agent seeing nothing.

    Without a magnitude floor this vetoed real trades and the reason string read
    as though the agent had argued the other side.
    """
    cons = aggregate([_sig("a", 0.8, 0.8), _sig("b", 0.7, 0.75),
                      _sig("c", 0.6, 0.7), _sig("quiet", -0.03, 0.95)])["X"]
    decision = decide("X", cons, ConsensusConfig())
    assert not decision.vetoed
    assert decision.dissenting == 1  # still counted as a dissent, just not a veto


def test_a_substantial_dissent_still_vetoes():
    cons = aggregate([_sig("a", 0.8, 0.8), _sig("b", 0.7, 0.75),
                      _sig("c", 0.6, 0.7), _sig("bear", -0.6, 0.9)])["X"]
    decision = decide("X", cons, ConsensusConfig())
    assert decision.vetoed and not decision.actionable


def test_rejection_message_does_not_read_as_a_tautology():
    """A confidence just under the bar must not print as 'X < X'."""
    cons = aggregate([_sig("a", 0.5, 0.4499), _sig("b", 0.5, 0.4499), _sig("c", 0.5, 0.4499)])["X"]
    decision = decide("X", cons, ConsensusConfig(min_confidence=0.45))
    assert not decision.actionable
    assert "confidence" in decision.rejection
    lhs, rhs = decision.rejection.replace("confidence ", "").split(" < ")
    assert float(lhs) < float(rhs)
