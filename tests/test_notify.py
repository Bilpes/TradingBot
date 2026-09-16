from tradingbot.notify.telegram import (
    FakeTransport, TelegramNotifier, daily_summary, entry_alert, exit_alert,
    no_trade_alert, risk_alert,
)


def test_unconfigured_notifier_reports_itself_instead_of_throwing():
    notifier = TelegramNotifier(bot_token="", chat_id="", transport=FakeTransport())
    result = notifier.send("hello")
    assert not result.delivered
    assert "not configured" in result.error


def test_successful_send_hits_the_right_endpoint():
    transport = FakeTransport()
    notifier = TelegramNotifier(bot_token="tok", chat_id="123", transport=transport,
                                min_seconds_between=0.0)
    result = notifier.send("hello")
    assert result.delivered
    url, payload = transport.sent[0]
    assert url == "https://api.telegram.org/bot tok/sendMessage".replace(" tok", "tok")
    assert payload["chat_id"] == "123"
    assert payload["text"] == "hello"


def test_transport_failure_is_reported_not_raised():
    notifier = TelegramNotifier(bot_token="t", chat_id="1",
                                transport=FakeTransport(ok=False), min_seconds_between=0.0)
    result = notifier.send("hello")
    assert not result.delivered
    assert result.error


def test_a_transport_that_raises_does_not_kill_the_caller():
    def exploding(url, payload):
        raise ConnectionError("network down")

    notifier = TelegramNotifier(bot_token="t", chat_id="1", transport=exploding,
                                min_seconds_between=0.0)
    result = notifier.send("hello")
    assert not result.delivered
    assert "ConnectionError" in result.error


def test_rate_limit_caps_sends_per_minute():
    transport = FakeTransport()
    notifier = TelegramNotifier(bot_token="t", chat_id="1", transport=transport,
                                rate_limit_per_minute=3, min_seconds_between=0.0)
    results = [notifier.send(f"m{i}") for i in range(6)]
    assert sum(r.delivered for r in results) == 3
    assert any("rate limit" in (r.error or "") for r in results)


def test_minimum_interval_throttles_bursts():
    transport = FakeTransport()
    notifier = TelegramNotifier(bot_token="t", chat_id="1", transport=transport,
                                rate_limit_per_minute=1000, min_seconds_between=60.0)
    assert notifier.send("first").delivered
    second = notifier.send("second")
    assert not second.delivered
    assert "throttled" in second.error


def test_result_callback_always_fires(monkeypatch):
    seen = []
    notifier = TelegramNotifier(bot_token="", chat_id="", transport=FakeTransport())
    notifier.send("x", on_result=lambda r: seen.append(r))
    assert len(seen) == 1 and not seen[0].delivered


# -- message templates ------------------------------------------------------
def test_entry_alert_shows_every_vote():
    text = entry_alert("HDFCBANK", "Banking", 42, 1520.5, 0.62, 0.41,
                       {"rsi": (0.5, 0.6), "macd": (-0.2, 0.4)}, stop=1450.0)
    assert "ENTRY HDFCBANK" in text and "Banking" in text
    assert "42" in text and "rsi" in text and "macd" in text
    assert "62%" in text


def test_exit_alert_shows_the_pnl_sign():
    assert "&#8377;-" in exit_alert("X", 10, 100.0, -250.0, "stop")
    assert "&#8377;+" in exit_alert("X", 10, 100.0, 250.0, "target")


def test_no_trade_alert_carries_the_rejection_reason():
    text = no_trade_alert("TCS", "IT", 0.31, 0.22, "only 2 agreeing (need 3)")
    assert "NO TRADE TCS" in text and "only 2 agreeing" in text


def test_risk_alert_and_summary_render():
    assert "RISK" in risk_alert("stale workers", "no signals from: Metal")
    summary = daily_summary(102_000, 40_000, 0.6, -0.03, 4,
                            {"n_trades": 12, "win_rate": 0.42,
                             "realised_pnl": 2000, "total_charges": 340}, "2026-09-16")
    assert "102,000.00" in summary and "12" in summary


def test_templates_escape_user_supplied_text():
    text = entry_alert("<script>", "Banking", 1, 100.0, 0.5, 0.1, {})
    assert "<script>" not in text
    assert "&lt;script&gt;" in text
