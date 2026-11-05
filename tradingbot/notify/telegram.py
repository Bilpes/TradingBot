"""Telegram alerts — trade cards, exits, daily & weekly P&L digests.

Pure `requests` (no bot framework). Messages are HTML-formatted with the ₹
amounts the user actually cares about. Silently no-ops when token/chat-id
are missing or TELEGRAM_ENABLED=false.
"""

from __future__ import annotations

import traceback
from typing import Optional

import requests

from ..config import SETTINGS
from ..logger import get_logger
from ..models import Position, Trade

log = get_logger("telegram")


class Telegram:
    def __init__(self):
        self.token = SETTINGS.telegram_bot_token
        self.chat_id = SETTINGS.telegram_chat_id
        self.enabled = SETTINGS.telegram_enabled and bool(self.token) and bool(self.chat_id)

    # ------------------------------------------------------------------
    def send(self, html: str) -> bool:
        if not self.enabled:
            log.debug("telegram disabled; message suppressed: %.80s", html.replace("\n", " "))
            return False
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": html,
                      "parse_mode": "HTML", "disable_web_page_preview": True},
                timeout=10)
            if r.status_code != 200:
                log.warning("telegram send failed %s: %.120s", r.status_code, r.text)
                return False
            return True
        except requests.RequestException as e:
            log.warning("telegram send error: %s", e)
            return False

    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_money(x: float) -> str:
        sign = "+" if x >= 0 else "−"
        return f"{sign}₹{abs(x):,.0f}"

    # ------------------------------------------------------------------
    def trade_entry(self, pos: Position) -> None:
        side_emoji = "🟢" if pos.side == "BUY" else "🔴"
        risk = abs(pos.entry_price - pos.stop) * pos.qty
        html = (
            f"{side_emoji} <b>{'BUY' if pos.side == 'BUY' else 'SELL (short)'} {pos.symbol}</b> "
            f"· <b>{pos.style}</b> ({'intraday' if pos.style == 'MIS' else 'swing'})\n"
            f"Qty: <b>{pos.qty}</b> @ ₹{pos.entry_price:,.2f}\n"
            f"Stop-loss: ₹{pos.stop:,.2f}  ({self._fmt_money(-risk)} risk)\n"
            f"Target-1: ₹{pos.target1:,.2f} · Target-2: ₹{pos.target2:,.2f}\n"
            f"Setup: <i>{pos.setup}</i> · Regime: <i>{pos.regime}</i>\n"
            f"ML P(up): <b>{pos.ml_prob:.2f}</b> · Score: {pos.score:+.0f}\n"
            + ("<b>Why:</b>\n" + "\n".join(f"• {r}" for r in pos.rationale[:3]) if pos.rationale else "")
        )
        self.send(html)

    def trade_exit(self, trade: Trade, pos: Optional[Position] = None) -> None:
        emoji = "✅" if trade.net_pnl >= 0 else "🛑"
        html = (
            f"{emoji} <b>CLOSED {trade.symbol}</b> · {trade.exit_reason}\n"
            f"{trade.qty} sh · entry ₹{trade.entry_price:,.2f} → exit ₹{trade.exit_price:,.2f}\n"
            f"Gross: {self._fmt_money(trade.gross_pnl)} · Charges: ₹{trade.charges:,.0f}\n"
            f"<b>Net P&amp;L: {self._fmt_money(trade.net_pnl)}</b> "
            f"(held {trade.held_days}d, {trade.style})"
        )
        if pos is not None and pos.qty > 0:
            html += f"\n<i>Runner open: {pos.qty} sh, stop ₹{pos.stop:,.2f}</i>"
        self.send(html)

    # ------------------------------------------------------------------
    def daily_digest(self, engine) -> None:
        pf = engine.portfolio.summary(engine.positions, engine.store.closed_positions_today())
        st = engine.store.win_stats()
        pos_lines = []
        for p in engine.positions:
            arrow = "▲" if p.total_pnl >= 0 else "▼"
            pos_lines.append(f"{arrow} {p.symbol} ({p.style}) {p.qty} sh → "
                             f"{self._fmt_money(p.total_pnl)}")
        html = (
            f"📊 <b>AlphaDesk daily digest</b> · {engine.now_ist().strftime('%d %b %Y')}\n"
            f"Day P&amp;L: <b>{self._fmt_money(pf['day_pnl'])}</b> · "
            f"Equity: ₹{pf['equity']:,.0f} ({pf['total_pnl_pct']:+.2f}% overall)\n"
            f"Realised today: {self._fmt_money(pf['realized_day'])} · "
            f"Open MTM: {self._fmt_money(pf['unrealized'])}\n"
            f"Open positions: {pf['open_positions']}/{pf['max_positions']} · "
            f"heat {pf['heat_pct']:.1f}%\n"
            + ("\n".join(pos_lines) if pos_lines else "(flat — no open positions)")
            + f"\n\nLifetime: {st['trades']} trades · win {st['win_rate']}% · "
              f"PF {st['profit_factor']}"
        )
        self.send(html)

    def weekly_digest(self, engine) -> None:
        pf = engine.portfolio.summary(engine.positions, 0)
        week = engine.store.realized_pnl_week()
        st = engine.store.win_stats()
        curve = engine.store.equity_curve(300)
        first = curve[0]["equity"] if curve else pf["equity"]
        wk_eq = pf["equity"] - first
        html = (
            f"🗓 <b>AlphaDesk weekly digest</b>\n"
            f"Week realised: <b>{self._fmt_money(week)}</b> · "
            f"Week equity move: {self._fmt_money(wk_eq)}\n"
            f"Total equity: ₹{pf['equity']:,.0f} · Lifetime P&amp;L: "
            f"{self._fmt_money(pf['total_pnl'])}\n"
            f"Trades so far: {st['trades']} · win rate {st['win_rate']}% · "
            f"PF {st['profit_factor']} · avg {self._fmt_money(st['avg_pnl'])}\n"
            f"Models: intraday [{engine.pred_intraday.mode}] · "
            f"swing [{engine.pred_swing.mode}]\n"
            f"<i>Stay flat on Sunday. Monday will be loud enough.</i>"
        )
        self.send(html)

    def risk_alert(self, text: str) -> None:
        self.send(f"⚠️ <b>RISK ALERT</b>\n{text}")

    def test(self) -> bool:
        return self.send("🤖 <b>AlphaDesk</b> connected. Alerts will appear here.")
