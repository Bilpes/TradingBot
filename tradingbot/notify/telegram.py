"""Telegram alerts.

``api.telegram.org`` is not reachable from this sandbox, so the transport is
injectable and unit-tested against a fake; nothing here has been sent to a real
chat yet. Set ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` to go live, and
verify with ``tradingbot telegram-check``.

Alerts are deliberately *informational*: the bot reports what the system decided
and why. It never asks for confirmation and never places an order.
"""

from __future__ import annotations

import html
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Optional

Transport = Callable[[str, Dict[str, object]], Dict[str, object]]


def httpx_transport(url: str, payload: Dict[str, object]) -> Dict[str, object]:
    import httpx  # lazy: only live alerting needs it

    resp = httpx.post(url, json=payload, timeout=15)
    return {"status_code": resp.status_code, "json": resp.json()}


class FakeTransport:
    """Records sends instead of performing them. Used by tests."""

    def __init__(self, ok: bool = True) -> None:
        self.sent: list[tuple[str, Dict[str, object]]] = []
        self.ok = ok

    def __call__(self, url: str, payload: Dict[str, object]) -> Dict[str, object]:
        self.sent.append((url, payload))
        return {"status_code": 200 if self.ok else 429,
                "json": {"ok": self.ok, "description": "" if self.ok else "rate limited"}}


@dataclass
class AlertResult:
    delivered: bool
    error: str = ""


class TelegramNotifier:
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        transport: Optional[Transport] = None,
        rate_limit_per_minute: int = 18,
        min_seconds_between: float = 2.0,
    ) -> None:
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self.transport = transport or httpx_transport
        self.rate_limit_per_minute = rate_limit_per_minute
        self.min_seconds_between = min_seconds_between
        self._recent: Deque[float] = deque()
        self._last_send = 0.0

    @property
    def configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    # -- throttling ---------------------------------------------------------
    def _throttled(self) -> Optional[str]:
        now = time.monotonic()
        while self._recent and now - self._recent[0] > 60.0:
            self._recent.popleft()
        if len(self._recent) >= self.rate_limit_per_minute:
            return "rate limit reached (per minute)"
        if now - self._last_send < self.min_seconds_between:
            return "throttled (minimum interval)"
        return None

    def send(self, text: str, kind: str = "info", symbol: Optional[str] = None,
             on_result: Optional[Callable[[AlertResult], None]] = None) -> AlertResult:
        if not self.configured:
            result = AlertResult(False, "telegram not configured")
            if on_result:
                on_result(result)
            return result

        reason = self._throttled()
        if reason:
            result = AlertResult(False, reason)
            if on_result:
                on_result(result)
            return result

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "HTML",
                   "disable_web_page_preview": True}
        try:
            response = self.transport(url, payload)
            self._recent.append(time.monotonic())
            self._last_send = time.monotonic()
            ok = response.get("status_code") == 200 and response.get("json", {}).get("ok", False)
            error = "" if ok else str(response.get("json", {}).get("description", "send failed"))
            result = AlertResult(ok, error)
        except Exception as exc:  # network failures must never kill the strategy loop
            result = AlertResult(False, f"{type(exc).__name__}: {exc}")

        if on_result:
            on_result(result)
        return result


# -- message templates ------------------------------------------------------
def _esc(value: object) -> str:
    return html.escape(str(value))


def entry_alert(symbol: str, sector: str, quantity: int, price: float,
                confidence: float, score: float, votes: Dict[str, tuple[float, float]],
                stop: Optional[float] = None) -> str:
    lines = [
        f"&#128994; <b>ENTRY {_esc(symbol)}</b> ({_esc(sector)})",
        f"Qty <b>{quantity}</b> @ <b>&#8377;{price:,.2f}</b>",
        f"Confidence <b>{confidence:.0%}</b>  score {score:+.2f}",
        f"Stop {'&#8377;%.2f' % stop if stop else 'none'}",
        "<i>Votes</i>",
    ]
    for agent, (agent_score, agent_conf) in sorted(votes.items(),
                                                   key=lambda kv: -abs(kv[1][0] * kv[1][1])):
        arrow = "&#128200;" if agent_score > 0 else "&#128201;" if agent_score < 0 else "&#8226;"
        lines.append(f"  {arrow} {agent}: {agent_score:+.2f} @ {agent_conf:.0%}")
    return "\n".join(lines)


def exit_alert(symbol: str, quantity: int, price: float, pnl: float, reason: str) -> str:
    icon = "&#128308;" if pnl < 0 else "&#9989;"
    return "\n".join([
        f"{icon} <b>EXIT {_esc(symbol)}</b>",
        f"Qty {quantity} @ <b>&#8377;{price:,.2f}</b>",
        f"P&amp;L <b>&#8377;{pnl:+,.2f}</b>",
        f"Reason: {_esc(reason)}",
    ])


def risk_alert(kind: str, detail: str) -> str:
    return f"&#9888;&#65039; <b>RISK: {_esc(kind)}</b>\n{_esc(detail)}"


def no_trade_alert(symbol: str, sector: str, score: float, confidence: float,
                   rejection: str) -> str:
    return "\n".join([
        f"&#128309; <b>NO TRADE {_esc(symbol)}</b> ({_esc(sector)})",
        f"score {score:+.2f} conf {confidence:.0%}",
        f"Reason: {_esc(rejection)}",
    ])


def daily_summary(equity: float, cash: float, exposure: float, drawdown: float,
                  positions: int, stats: dict, session: str) -> str:
    return "\n".join([
        f"&#128202; <b>Paper summary &#8212; {_esc(session)}</b>",
        f"Equity <b>&#8377;{equity:,.2f}</b>  (cash &#8377;{cash:,.2f})",
        f"Exposure {exposure:.1%}   drawdown {drawdown:.2%}",
        f"Open positions {positions}",
        f"Trades {stats.get('n_trades', 0)}   win rate {stats.get('win_rate', 0):.1%}",
        f"Realised P&amp;L &#8377;{stats.get('realised_pnl', 0):+,.2f}",
        f"Charges paid &#8377;{stats.get('total_charges', 0):,.2f}",
    ])
