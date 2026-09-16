import csv
import json

import pytest

from tradingbot.data.dhan import (
    SecurityResolutionError, parse_historical, resolve_security_ids,
)


def _scrip_master(tmp_path, rows):
    fp = tmp_path / "dhan_scrip_master.csv"
    with fp.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "SEM_SMST_SECURITY_ID", "SEM_TRADING_SYMBOL", "SEM_EXM_EXCH_ID",
            "SEM_INSTRUMENT_NAME", "SEM_SEGMENT"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return fp


def test_resolves_nse_equities_only(tmp_path):
    fp = _scrip_master(tmp_path, [
        {"SEM_SMST_SECURITY_ID": "1333", "SEM_TRADING_SYMBOL": "RELIANCE",
         "SEM_EXM_EXCH_ID": "NSE", "SEM_INSTRUMENT_NAME": "EQUITY", "SEM_SEGMENT": "E"},
        # Same symbol on BSE must not be picked up.
        {"SEM_SMST_SECURITY_ID": "9999", "SEM_TRADING_SYMBOL": "RELIANCE",
         "SEM_EXM_EXCH_ID": "BSE", "SEM_INSTRUMENT_NAME": "EQUITY", "SEM_SEGMENT": "E"},
        # A futures row for the same name must not be picked up either.
        {"SEM_SMST_SECURITY_ID": "8888", "SEM_TRADING_SYMBOL": "RELIANCE",
         "SEM_EXM_EXCH_ID": "NSE", "SEM_INSTRUMENT_NAME": "FUTSTK", "SEM_SEGMENT": "D"},
    ])
    assert resolve_security_ids(["RELIANCE"], fp) == {"RELIANCE": "1333"}


def test_missing_symbol_is_a_hard_failure_not_a_silent_skip(tmp_path):
    fp = _scrip_master(tmp_path, [
        {"SEM_SMST_SECURITY_ID": "1333", "SEM_TRADING_SYMBOL": "RELIANCE",
         "SEM_EXM_EXCH_ID": "NSE", "SEM_INSTRUMENT_NAME": "EQUITY", "SEM_SEGMENT": "E"},
    ])
    with pytest.raises(SecurityResolutionError, match="NOTREAL"):
        resolve_security_ids(["RELIANCE", "NOTREAL"], fp)


def test_missing_scrip_master_explains_what_to_do(tmp_path):
    with pytest.raises(SecurityResolutionError, match="scrip master"):
        resolve_security_ids(["RELIANCE"], tmp_path / "absent.csv")


def _payload(n=3, start_epoch=1_640_000_000):
    return {
        "status": "success",
        "start_time": [start_epoch + 86_400 * i for i in range(n)],
        "open": [100.0 + i for i in range(n)],
        "high": [101.0 + i for i in range(n)],
        "low": [99.0 + i for i in range(n)],
        "close": [100.5 + i for i in range(n)],
        "volume": [1000.0 * (i + 1) for i in range(n)],
    }


def test_parse_historical_produces_ohlcv_in_ist():
    frame = parse_historical(_payload())
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert len(frame) == 3
    assert frame["close"].iloc[0] == pytest.approx(100.5)
    assert frame.index.tz is None  # naive, IST-localised
    assert frame.index.is_monotonic_increasing


def test_parse_historical_surfaces_a_failure_status():
    with pytest.raises(RuntimeError, match="status=failure"):
        parse_historical({"status": "failure", "remarks": {"code": "DH-901"}})


def test_parse_historical_rejects_an_unrecognised_shape():
    with pytest.raises(RuntimeError, match="unrecognised"):
        parse_historical({"foo": [1, 2, 3]})


def test_provider_uses_the_injected_transport_and_resolves_ids():
    from tradingbot.data.dhan import DhanProvider

    calls = []

    class FakeHistorical:
        def historical_daily_data(self, **kwargs):
            calls.append(kwargs)
            return _payload()

    provider = DhanProvider(security_map={"RELIANCE": "1333"},
                            historical_factory=FakeHistorical)
    history = provider.load(["RELIANCE"], "2022-01-01", "2022-01-10")

    assert "RELIANCE" in history
    assert calls[0]["security_id"] == "1333"
    assert calls[0]["exchange_segment"] == "NSE_EQ"
    assert calls[0]["instrument_type"] == "EQUITY"
    assert calls[0]["from_date"] == "2022-01-01"


def test_provider_refuses_an_unmapped_symbol():
    from tradingbot.data.dhan import DhanProvider

    provider = DhanProvider(security_map={}, historical_factory=lambda: None)
    with pytest.raises(SecurityResolutionError, match="no security id"):
        provider.load(["RELIANCE"], "2022-01-01")


def test_provider_requires_credentials_when_no_factory_is_injected(monkeypatch):
    from tradingbot.data.dhan import DhanProvider

    monkeypatch.delenv("DHAN_CLIENT_ID", raising=False)
    monkeypatch.delenv("DHAN_ACCESS_TOKEN", raising=False)
    provider = DhanProvider(security_map={"RELIANCE": "1333"})
    with pytest.raises(RuntimeError, match="credentials missing"):
        provider.load(["RELIANCE"], "2022-01-01")


def test_load_security_map_round_trip(tmp_path):
    from tradingbot.data.dhan import load_security_map

    fp = tmp_path / "map.json"
    fp.write_text(json.dumps({"reliance": "1333"}))
    assert load_security_map(fp) == {"RELIANCE": "1333"}
