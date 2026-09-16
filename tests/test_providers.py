import json

import numpy as np
import pandas as pd
import pytest

from tradingbot.data.providers import AlpacaProvider, CSVProvider, SyntheticProvider
from tradingbot.data.schema import validate_history


# -- synthetic --------------------------------------------------------------
def test_synthetic_is_deterministic_and_ohl_consistent():
    a = SyntheticProvider(sessions=200, seed=5).load([], "2020-01-02")
    b = SyntheticProvider(sessions=200, seed=5).load([], "2020-01-02")
    pd.testing.assert_frame_equal(a["AAPL"], b["AAPL"])

    for symbol, df in a.items():
        assert (df["high"] >= df["low"]).all()
        assert (df[["open", "high", "low", "close"]] > 0).all().all()
        assert (df["volume"] > 0).all()


def test_synthetic_seeds_produce_different_paths():
    a = SyntheticProvider(sessions=200, seed=1).load(["AAPL"], "2020-01-02")["AAPL"]["close"]
    b = SyntheticProvider(sessions=200, seed=2).load(["AAPL"], "2020-01-02")["AAPL"]["close"]
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_is_reproducible_across_processes():
    """Regression test: this used to fail because of per-process str hashing.

    An in-process comparison cannot catch it -- the salt is fixed for the life
    of a process -- so the generator has to be run in a *separate* interpreter
    and the checksums compared.
    """
    import subprocess
    import sys

    snippet = (
        "from tradingbot.data.providers import SyntheticProvider;"
        "h = SyntheticProvider(sessions=60, seed=4).load([], '2020-01-02');"
        "print(round(float(sum(df['close'].sum() for df in h.values())), 6))"
    )
    runs = [
        subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, check=True)
        for _ in range(3)
    ]
    checksums = {r.stdout.strip() for r in runs}
    assert len(checksums) == 1, f"generator is not reproducible across processes: {checksums}"


# -- validation -------------------------------------------------------------
def _good_df():
    dates = pd.bdate_range("2022-01-03", periods=10)
    return pd.DataFrame(
        {
            "open": np.linspace(10, 11, 10),
            "high": np.linspace(11, 12, 10),
            "low": np.linspace(9, 10, 10),
            "close": np.linspace(10, 11, 10),
            "volume": np.full(10, 1000.0),
        },
        index=dates,
    )


def test_validation_rejects_bad_frames():
    with pytest.raises(ValueError, match="empty"):
        validate_history({})

    bad_cols = _good_df().drop(columns=["close"])
    with pytest.raises(ValueError, match="missing columns"):
        validate_history({"X": bad_cols})

    inverted = _good_df()
    inverted.loc[inverted.index[3], "high"] = 1.0
    with pytest.raises(ValueError, match="high < low"):
        validate_history({"X": inverted})

    neg = _good_df()
    neg.loc[neg.index[2], "close"] = -5.0
    with pytest.raises(ValueError, match="non-positive"):
        validate_history({"X": neg})

    nan = _good_df()
    nan.loc[nan.index[1], "close"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_history({"X": nan})


def test_validation_sorts_and_dedupes():
    df = _good_df()
    shuffled = pd.concat([df.iloc[5:], df.iloc[:5]]).sort_index(ascending=False)
    out = validate_history({"X": shuffled})["X"]
    assert out.index.is_monotonic_increasing
    assert len(out) == 10


# -- CSV --------------------------------------------------------------------
def test_csv_provider_reads_a_directory(tmp_path):
    for symbol in ("AAA", "BBB"):
        df = _good_df().reset_index().rename(columns={"index": "date"})
        df.to_csv(tmp_path / f"{symbol}.csv", index=False)
    hist = CSVProvider(tmp_path).load(["AAA", "BBB"])
    assert set(hist) == {"AAA", "BBB"}
    assert list(hist["AAA"].columns) == ["open", "high", "low", "close", "volume"]


def test_csv_provider_reads_long_format(tmp_path):
    frames = []
    for symbol in ("AAA", "BBB"):
        df = _good_df().reset_index().rename(columns={"index": "date"})
        df["symbol"] = symbol
        frames.append(df)
    fp = tmp_path / "all.csv"
    pd.concat(frames).to_csv(fp, index=False)
    hist = CSVProvider(fp).load([])
    assert set(hist) == {"AAA", "BBB"}


def test_csv_provider_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        CSVProvider(tmp_path / "nope")


# -- Alpaca -----------------------------------------------------------------
def _page(bars_by_symbol, token=None):
    return json.dumps({"bars": bars_by_symbol, "next_page_token": token})


def test_alpaca_parses_bars_and_follows_pagination():
    calls = []

    def fake_get(url, headers):
        calls.append(url)
        assert headers["APCA-API-KEY-ID"] == "k"
        if len(calls) == 1:
            return _page({"AAPL": [
                {"t": "2022-01-03T05:00:00Z", "o": 10, "h": 11, "l": 9, "c": 10.5, "v": 100},
                {"t": "2022-01-04T05:00:00Z", "o": 10.5, "h": 12, "l": 10, "c": 11.5, "v": 200},
            ]}, token="PAGE2")
        return _page({"AAPL": [
            {"t": "2022-01-05T05:00:00Z", "o": 11.5, "h": 13, "l": 11, "c": 12.5, "v": 300},
        ]}, token=None)

    provider = AlpacaProvider(api_key="k", secret_key="s", http_get=fake_get)
    hist = provider.load(["AAPL"], "2022-01-03")

    assert len(calls) == 2
    assert "page_token=PAGE2" in calls[1]
    assert "timeframe=1Day" in calls[0]
    assert len(hist["AAPL"]) == 3
    assert hist["AAPL"]["close"].iloc[-1] == 12.5


def test_alpaca_requires_credentials():
    provider = AlpacaProvider(api_key="", secret_key="", http_get=lambda u, h: "{}")
    with pytest.raises(RuntimeError, match="credentials missing"):
        provider.load(["AAPL"], "2022-01-03")


def test_alpaca_raises_when_the_response_has_no_bars():
    provider = AlpacaProvider(api_key="k", secret_key="s", http_get=lambda u, h: '{"bars": {}}')
    with pytest.raises(RuntimeError, match="no bars"):
        provider.load(["AAPL"], "2022-01-03")
