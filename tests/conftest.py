import numpy as np
import pandas as pd
import pytest

from tradingbot.data.providers import SyntheticProvider


@pytest.fixture(scope="session")
def synthetic_history():
    """A small deterministic factor-model history shared across tests."""
    return SyntheticProvider(sessions=400, seed=11).load([], "2019-01-02")


@pytest.fixture(scope="session")
def small_universe_history():
    """Six symbols in two sectors, for tests that need a readable panel."""
    return SyntheticProvider(sessions=300, seed=3).load(
        ["AAPL", "MSFT", "XOM", "CVX", "PG", "KO"], "2019-01-02"
    )


@pytest.fixture
def tiny_history():
    """Hand-built 60-session frame with exact, inspectable numbers."""
    dates = pd.bdate_range("2022-01-03", periods=60)
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.01, 60)))
    df = pd.DataFrame(
        {
            "open": close * (1 - 0.001),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(60, 1_000_000.0),
        },
        index=dates,
    )
    return {"TEST": df}
