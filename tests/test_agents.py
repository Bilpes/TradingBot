import numpy as np
import pandas as pd
import pytest

from tradingbot.agents import build_default_agents
from tradingbot.agents.meanrev import MeanReversionAgent
from tradingbot.agents.momentum import CrossSectionalMomentumAgent
from tradingbot.agents.sector import SectorRotationAgent
from tradingbot.agents.trend import TrendAgent
from tradingbot.agents.volatility import VolatilityRegimeAgent


def _frame(drift=0.0, vol=0.01, n=250, p0=100.0, seed=0, shock_last=None):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    if shock_last:
        k, mult = shock_last
        r[-k:] *= mult
    close = p0 * np.exp(np.cumsum(r))
    dates = pd.bdate_range("2020-01-02", periods=n)
    prev = np.concatenate([[p0], close[:-1]])
    return pd.DataFrame(
        {
            "open": prev,
            "high": np.maximum(prev, close) * 1.005,
            "low": np.minimum(prev, close) * 0.995,
            "close": close,
            "volume": np.full(n, 1e6),
        },
        index=dates,
    )


def _ou_frame(n=250, phi=0.90, noise=0.02, p0=100.0, seed=1):
    """Mean-reverting (Ornstein-Uhlenbeck) log price: choppy by construction."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0.0, noise)
    close = p0 * np.exp(x)
    dates = pd.bdate_range("2020-01-02", periods=n)
    return pd.DataFrame(
        {
            "open": close, "high": close * 1.005, "low": close * 0.995,
            "close": close, "volume": np.full(n, 1e6),
        },
        index=dates,
    )


# -- contract ---------------------------------------------------------------
def test_every_agent_emits_bounded_signals(synthetic_history):
    date = sorted(synthetic_history["AAPL"].index)[-30]
    for agent in build_default_agents():
        agent.fit(synthetic_history)
        signals = agent.signals_on(date)
        assert signals, f"{agent.name} produced nothing"
        for sig in signals:
            assert -1.0 <= sig.score <= 1.0
            assert 0.0 <= sig.confidence <= 1.0
            assert sig.agent == agent.name


def test_agents_refuse_to_answer_before_being_fitted(synthetic_history):
    for agent in build_default_agents():
        with pytest.raises(RuntimeError):
            agent.signals_on(pd.Timestamp("2021-01-04"))


def test_min_history_covers_the_longest_window_used():
    assert CrossSectionalMomentumAgent(lookback=126).min_history >= 126
    assert TrendAgent(slow=50, channel=60).min_history >= 60
    assert SectorRotationAgent(lookback=63, breadth_ma=50).min_history >= 63


# -- momentum ---------------------------------------------------------------
def test_momentum_ranks_the_strongest_name_first():
    hist = {
        "HOT": _frame(drift=0.0030, seed=1),
        "MILD": _frame(drift=0.0008, seed=2),
        "COLD": _frame(drift=-0.0025, seed=3),
    }
    agent = CrossSectionalMomentumAgent(lookback=126, skip=21)
    agent.fit(hist)
    scores = {sig.symbol: sig.score for sig in agent.signals_on(hist["HOT"].index[-1])}
    assert scores["HOT"] > scores["MILD"] > scores["COLD"]
    assert scores["HOT"] > 0 > scores["COLD"]


# -- trend ------------------------------------------------------------------
def test_trend_is_positive_and_confident_in_a_persistent_uptrend():
    hist = {"UP": _frame(drift=0.0035, vol=0.004, seed=5)}
    agent = TrendAgent()
    agent.fit(hist)
    sig = agent.signals_on(hist["UP"].index[-1])[0]
    assert sig.score > 0.5
    assert sig.confidence > 0.5  # a persistent trend has a high efficiency ratio


# -- mean reversion ---------------------------------------------------------
def test_mean_reversion_stays_quiet_in_a_strong_trend():
    hist = {"TREND": _frame(drift=0.004, vol=0.001, seed=6)}
    agent = MeanReversionAgent()
    agent.fit(hist)
    n_spoken = sum(len(agent.signals_on(d)) for d in hist["TREND"].index[-100:])
    assert n_spoken == 0  # it recognises the trend and declines to vote


def test_mean_reversion_speaks_in_a_choppy_market():
    hist = {"CHOP": _ou_frame(phi=0.85, noise=0.025, seed=8)}
    agent = MeanReversionAgent()
    agent.fit(hist)
    all_sigs = [s for d in hist["CHOP"].index[-150:] for s in agent.signals_on(d)]
    assert all_sigs, "expected some contrarian votes in a ranging regime"
    assert all(0.0 < s.confidence <= 0.8 for s in all_sigs)


# -- sector rotation --------------------------------------------------------
def test_sector_agent_prefers_the_stronger_sector():
    sector_map = {"G1": "Growth", "G2": "Growth", "V1": "Value", "V2": "Value"}
    hist = {
        "G1": _frame(drift=0.0025, seed=11),
        "G2": _frame(drift=0.0022, seed=12),
        "V1": _frame(drift=-0.0022, seed=13),
        "V2": _frame(drift=-0.0025, seed=14),
    }
    agent = SectorRotationAgent(sector_map=lambda s: sector_map[s])
    agent.fit(hist)
    scores = {sig.symbol: sig.score for sig in agent.signals_on(hist["G1"].index[-1])}
    assert scores["G1"] > 0 > scores["V1"]
    assert scores["G2"] == pytest.approx(scores["G1"])  # peers share the sector view


# -- volatility gate --------------------------------------------------------
def test_volatility_gate_cuts_conviction_in_a_stressed_regime():
    calm = {"M": _frame(vol=0.005, n=400, seed=21)}
    stressed = {"M": _frame(vol=0.005, n=400, seed=21, shock_last=(40, 5.0))}

    a = VolatilityRegimeAgent()
    a.fit(calm)
    calm_sig = a.signals_on(calm["M"].index[-1])[0]

    b = VolatilityRegimeAgent()
    b.fit(stressed)
    hot_sig = b.signals_on(stressed["M"].index[-1])[0]

    assert calm_sig.role == "gate" and calm_sig.symbol == "*"
    assert calm_sig.confidence == pytest.approx(1.0)
    assert hot_sig.confidence < calm_sig.confidence
    assert hot_sig.confidence >= VolatilityRegimeAgent().floor
