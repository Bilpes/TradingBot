import pytest
from fastapi.testclient import TestClient

from tradingbot.agents.registry import profile_for
from tradingbot.data.providers import SyntheticProvider
from tradingbot.live import PaperRunConfig, run_in_process
from tradingbot.markets.nse import NSE_SECTORS, NSE_UNIVERSE, nse_sector_of
from tradingbot.notify.telegram import FakeTransport, TelegramNotifier
from tradingbot.paper.broker import NSECostModel, PaperBroker
from tradingbot.store.db import Store
from tradingbot.web.app import create_app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db = str(tmp_path_factory.mktemp("web") / "paper.sqlite3")
    history = SyntheticProvider(sessions=400, seed=17, sector_map=nse_sector_of,
                        universe=list(NSE_UNIVERSE)).load([], "2021-01-04")
    store = Store(db)
    broker = PaperBroker(capital=100_000, costs=NSECostModel(product="CNC"))
    run_in_process(
        history=history, store=store, broker=broker,
        profiles={s: profile_for(s) for s in NSE_SECTORS},
        notifier=TelegramNotifier(bot_token="t", chat_id="1",
                                  transport=FakeTransport(), min_seconds_between=0.0),
        config=PaperRunConfig(sectors=NSE_SECTORS), sector_of=nse_sector_of,
    )
    store.close()
    return TestClient(create_app(db))


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok" and body["exists"] == "True"


def test_dashboard_html_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "TradingBot" in resp.text
    assert "/api/state" in resp.text


def test_state_reports_the_run(client):
    body = client.get("/api/state").json()
    assert body["equity"] is not None
    assert body["n_decisions"] > 0
    assert body["n_signals"] > 0
    assert body["curve"], "equity curve missing"
    assert body["meta"]["universe"]


def test_decisions_expose_their_votes(client):
    rows = client.get("/api/decisions?limit=20").json()
    assert rows
    first = rows[0]
    assert isinstance(first["votes"], dict) and first["votes"]
    assert "votes_json" not in first  # parsed, not leaked raw
    for key in ("score", "confidence", "agreeing", "dissenting", "actionable"):
        assert key in first


def test_single_decision_endpoint(client):
    rows = client.get("/api/decisions?limit=1").json()
    detail = client.get(f"/api/decisions/{rows[0]['id']}").json()
    assert detail["id"] == rows[0]["id"]
    assert detail["votes"]


def test_unknown_decision_is_a_404(client):
    assert client.get("/api/decisions/999999").status_code == 404


def test_rejections_endpoint_explains_every_decline(client):
    rows = client.get("/api/rejections?limit=50").json()
    assert rows
    assert all(r["rejection"] for r in rows)


def test_signals_endpoint_filters_by_symbol(client):
    all_rows = client.get("/api/signals?limit=50").json()
    assert all_rows
    symbol = all_rows[0]["symbol"]
    filtered = client.get(f"/api/signals?symbol={symbol}&limit=50").json()
    assert filtered and all(r["symbol"] == symbol for r in filtered)
    assert isinstance(filtered[0]["parts"], dict)


def test_alerts_endpoint(client):
    rows = client.get("/api/alerts?limit=10").json()
    assert isinstance(rows, list)


def test_calibration_is_reported_even_when_unsettled(client):
    body = client.get("/api/state").json()
    assert "calibration" in body
    assert "brier_score" in body
