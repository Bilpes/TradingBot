"""The test that matters most: agents must not see the future.

Method: fit an agent on the full history and record its answer at date ``d``.
Then fit a *fresh* agent on the same history truncated at ``d`` -- the future is
physically absent -- and record its answer at ``d`` again. If any indicator
peeked forward, the two answers differ.
"""

import pandas as pd
import pytest

from tradingbot.agents import build_default_agents


def _truncate(history, cutoff):
    return {s: df.loc[df.index <= cutoff] for s, df in history.items()}


@pytest.mark.parametrize("agent_cls", [type(a) for a in build_default_agents()])
def test_signal_is_identical_when_the_future_is_removed(agent_cls, synthetic_history):
    agent_full = agent_cls()
    agent_full.fit(synthetic_history)

    probe = sorted(set().union(*[set(df.index) for df in synthetic_history.values()]))
    date = probe[int(len(probe) * 0.75)]

    before = {s: (sig.score, sig.confidence) for sig in agent_full.signals_on(date) for s in [sig.symbol]}

    truncated = _truncate(synthetic_history, date)
    assert max(df.index[-1] for df in truncated.values()) <= date

    agent_cut = agent_cls()
    agent_cut.fit(truncated)
    after = {s: (sig.score, sig.confidence) for sig in agent_cut.signals_on(date) for s in [sig.symbol]}

    assert before, "agent produced no signals at the probe date"
    assert set(before) == set(after)
    for symbol in before:
        assert before[symbol][0] == pytest.approx(after[symbol][0], abs=1e-12), f"{symbol} score leaked"
        assert before[symbol][1] == pytest.approx(after[symbol][1], abs=1e-12), f"{symbol} confidence leaked"


def test_probe_date_actually_has_history(synthetic_history):
    """Guard against the test above silently probing an empty date."""
    probe = sorted(set().union(*[set(df.index) for df in synthetic_history.values()]))
    date = probe[int(len(probe) * 0.75)]
    assert date > probe[100]
    assert isinstance(date, pd.Timestamp)
