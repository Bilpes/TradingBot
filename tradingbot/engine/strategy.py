"""Strategy fusion — where nine agent reports become ONE trade decision.

This is the 20-years-of-desk-work layer. The order of checks is deliberate:
market context first (don't fight the index), then ML probability (the gate),
then setup construction, then per-symbol sanity (news veto, circuit band,
liquidity). Only then do we propose a Signal to the Risk layer.

Intraday (MIS) setups:
  ORB_15M      — NIFTY trend-day, stock breaks its 15-min opening range
                 with volume, price above VWAP
  VWAP_PULLBACK — uptrend day, stock dips to VWAP ± 0.25×ATR(15m) and
                 holds; enter on reclaim
Intraday short (SELL, MIS only): mirror ORB breakdown when regime is
  STRONG_TREND_DOWN and stock breaks OR low with volume.

Swing (CNC) setups:
  EMA_PULLBACK — daily uptrend (EMA50>EMA200, price>EMA50), pullback to
                 EMA21 zone with a bullish close back above it
  BREAKOUT_20D — close above 20-day high with volume ≥1.5× average
Trade construction: ATR-based stops (2×ATR daily for swing, 0.6×ATR15m or
OR-low for intraday), R-multiple targets, T1 partial for intraday.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import indicators as ta
from ..config import SETTINGS, AGENT_WEIGHTS
from ..models import AgentReport, Position, Signal
from .portfolio import Portfolio


class StrategyFusion:
    def __init__(self, pred_intraday, pred_swing):
        self.gate_intraday = SETTINGS.ml_prob_gate_intraday
        self.gate_swing = SETTINGS.ml_prob_gate_swing

    # ------------------------------------------------------------------
    @staticmethod
    def fused_score(reports: List[AgentReport]) -> float:
        """Weighted composite over configured agent weights (−100..+100)."""
        total, wsum = 0.0, 0.0
        for r in reports:
            w = AGENT_WEIGHTS.get(r.agent)
            if w is None or r.status != "OK":
                continue                     # dead agents neither add nor dilute
            conf = max(0.2, min(1.0, r.confidence))
            total += r.score * w * conf
            wsum += w
        return float(np.clip(total / wsum if wsum else 0.0, -100, 100))

    # ------------------------------------------------------------------
    @staticmethod
    def _report(reports: List[AgentReport], name: str) -> Optional[AgentReport]:
        for r in reports:
            if r.agent == name:
                return r
        return None

    @staticmethod
    def _regime(reports: List[AgentReport]) -> Tuple[str, float]:
        ms = StrategyFusion._report(reports, "MarketStructure")
        if ms and ms.status == "OK":
            return ms.data.get("regime", "NEUTRAL"), float(ms.score)
        return "NEUTRAL", 0.0

    @staticmethod
    def _rationale(reports: List[AgentReport], top: int = 3) -> List[str]:
        """Top contributing notes for the UI/Telegram card."""
        ranked = sorted(
            [r for r in reports if r.status == "OK" and r.note],
            key=lambda r: abs(r.score),
            reverse=True)[:top]
        return [f"{r.agent}: {r.note}" for r in ranked]

    # ==================================================================
    #  MAIN DECISION
    # ==================================================================
    def decide(self, ctx, reports: List[AgentReport], portfolio: Portfolio,
               mode: str) -> Optional[Signal]:
        """mode: 'intraday' | 'swing'. Returns Signal or None. Long AND short-capable."""
        symbol = ctx.symbol

        # ---- 0. hard gates ------------------------------------------------
        risk_rep = self._report(reports, "Risk")
        if risk_rep is None or not risk_rep.data.get("entries_allowed", True):
            return None

        sent = self._report(reports, "Sentiment")
        if sent and sent.data.get("veto"):
            return None

        regime, ms_score = self._regime(reports)
        macro = self._report(reports, "Macro")
        macro_flags = (macro.data.get("flags", []) if macro and macro.status == "OK" else [])

        # regime filter: no fresh longs into falling markets, no shorts into rising
        if mode == "intraday":
            if regime in ("STRONG_TREND_DOWN",) and not self._allow_short(reports):
                return None
            if regime == "HIGH_VOL" and len([p for p in ctx.positions if p.style == "MIS"]) >= 2:
                return None   # vol regime: cap intraday book
        else:  # swing
            if regime in ("STRONG_TREND_DOWN", "HIGH_VOL", "TREND_DOWN"):
                return None   # swing longs need a friendly regime

        # ---- 1. fused score & ML gate --------------------------------------
        score = self.fused_score(reports)
        ml = self._report(reports, "MLPrediction")
        if not ml or ml.status != "OK":
            return None
        p_up = ml.data.get("p_up", 0.5)
        p_swing = ml.data.get("p_swing")
        p_intra = ml.data.get("p_intraday")

        gate = self.gate_swing if mode == "swing" else self.gate_intraday
        ml_prob_for_mode = (p_swing if mode == "swing" else p_intra) or p_up

        side = "BUY"
        if mode == "intraday" and regime in ("STRONG_TREND_DOWN", "TREND_DOWN") \
                and p_intra is not None and (1 - p_intra) >= self.gate_intraday:
            side = "SELL"           # regime-aligned intraday short
            ml_prob = 1 - p_intra
        else:
            ml_prob = ml_prob_for_mode if side == "BUY" else 1 - ml_prob_for_mode

        if score < 12 and side == "BUY":
            return None             # composite too weak
        if side == "BUY" and ml_prob < gate:
            return None             # ML gate: the core discipline
        if side == "SELL" and ml_prob < self.gate_intraday:
            return None

        # ---- 2. setup construction ------------------------------------------
        sig = self._construct_intraday(ctx, side, score, ml_prob, regime, macro_flags) \
            if mode == "intraday" else self._construct_swing(ctx, side, score, ml_prob, regime, macro_flags)
        if sig is None:
            return None

        sig.rationale = self._rationale(reports)
        return sig

    # ------------------------------------------------------------------
    @staticmethod
    def _allow_short(reports: List[AgentReport]) -> bool:
        ms = StrategyFusion._report(reports, "MarketStructure")
        return bool(ms and ms.status == "OK" and ms.data.get("regime") in
                    ("STRONG_TREND_DOWN", "TREND_DOWN"))

    # ------------------------------------------------------------------
    #  INTRADAY construction
    # ------------------------------------------------------------------
    def _construct_intraday(self, ctx, side, score, ml_prob, regime, macro_flags) -> Optional[Signal]:
        intra = ctx.intraday
        idx_intra = ctx.index_intraday
        if intra is None or len(intra) < 12 or idx_intra is None or len(idx_intra) < 3:
            return None

        intra = intra.copy()
        intra["ist"] = pd.to_datetime(intra["ts"], utc=True).dt.tz_convert("Asia/Kolkata")
        if len(intra) < 12:
            return None

        ltp = float(intra["close"].iloc[-1])
        atr5 = ta.atr(intra, 14).iloc[-1]
        if not np.isfinite(atr5) or atr5 <= 0:
            return None
        vwap = ta.vwap(intra)

        # liquidity sanity: min 5 lakh avg daily turnover proxy
        df = ctx.daily
        avg_turnover = (df["close"].tail(20) * df["volume"].tail(20)).mean()
        if avg_turnover < 5e6:      # ₹50 lakh/day floor
            return None

        # circuit proximity: don't chase >7% moves today
        day_ret = ltp / float(df["close"].iloc[-1]) - 1 if len(df) else 0
        if abs(day_ret) > 0.07:
            return None

        prev_day = df.iloc[-1]
        o = intra["open"].iloc[0] if len(intra) else ltp
        # 15-min opening range
        orb = intra[intra["ist"].dt.strftime("%H:%M") < "09:30"]
        if len(orb) == 0:
            return None
        orb_h, orb_l = float(orb["high"].max()), float(orb["low"].min())

        # ORB breakout (long)
        breakout = ltp > orb_h and (ltp - orb_h) < 0.6 * atr5
        breakdown = ltp < orb_l and (orb_l - ltp) < 0.6 * atr5
        above_vwap = ltp > vwap.iloc[-1]
        below_vwap = ltp < vwap.iloc[-1]

        # session volume participation (projected full-day vs 20-day average)
        vol_today = float(intra["volume"].sum())
        vol_proj = vol_today / max(len(intra), 1) * 75
        vol_ok = vol_proj > float(df["volume"].tail(20).mean()) * 0.6

        if side == "BUY":
            ok = breakout and above_vwap and vol_ok and regime not in ("STRONG_TREND_DOWN",)
            setup = "ORB_15M"
            if not ok:
                # VWAP pullback alternative
                pulled = (ltp - vwap.iloc[-1]) < 0.25 * atr5 and ltp > vwap.iloc[-1] * 0.998
                ok = pulled and above_vwap and regime in ("TREND_UP", "STRONG_TREND_UP", "NEUTRAL")
                setup = "VWAP_PULLBACK"
            if not ok:
                return None
            entry = ltp
            stop = min(orb_l if setup == "ORB_15M" else entry - 0.6 * atr5, entry - 0.6 * atr5)
            if entry - stop < 0.15 * atr5:            # stop too tight → noise
                stop = entry - 0.6 * atr5
        else:  # SELL (intraday short)
            ok = breakdown and below_vwap and vol_ok and regime in ("STRONG_TREND_DOWN", "TREND_DOWN")
            setup = "ORB_SHORT"
            if not ok:
                return None
            entry = ltp
            stop = max(orb_h, entry + 0.6 * atr5)
            if stop - entry < 0.15 * atr5:
                stop = entry + 0.6 * atr5

        risk = abs(entry - stop)
        t1 = entry + risk * 1.5 * (1 if side == "BUY" else -1)
        t2 = entry + risk * 2.5 * (1 if side == "BUY" else -1)

        sig = Signal(symbol=ctx.symbol, side=side, style="MIS", entry=entry,
                     stop=stop, target1=t1, target2=t2, qty=0,
                     score=score, ml_prob=ml_prob, setup=setup, regime=regime)
        sig.rationale = [f"regime {regime}"] + macro_flags[:1]
        return sig

    # ------------------------------------------------------------------
    #  SWING construction
    # ------------------------------------------------------------------
    def _construct_swing(self, ctx, side, score, ml_prob, regime, macro_flags) -> Optional[Signal]:
        if side != "BUY":
            return None              # CNC long-only (India retail reality)
        df = ctx.daily
        if df is None or len(df) < 60:
            return None
        close = df["close"]
        ltp = float(close.iloc[-1])

        e21 = ta.ema(close, 21)
        e50 = ta.ema(close, 50)
        e200 = ta.ema(close, 200) if len(close) >= 200 else ta.ema(close, len(close) // 2)
        atr_d = ta.atr(df, 14).iloc[-1]
        rsi = ta.rsi(close, 14).iloc[-1]
        vol_avg = df["volume"].tail(20).mean()

        if not np.isfinite(atr_d) or atr_d <= 0:
            return None

        # liquidity
        avg_turnover = (df["close"].tail(20) * df["volume"].tail(20)).mean()
        if avg_turnover < 1e7:       # ₹1 cr/day floor for swing
            return None

        setup, entry, stop = None, None, None

        # A) EMA pullback in daily uptrend
        uptrend = e50.iloc[-1] > e200.iloc[-1] and ltp > e50.iloc[-1]
        near_ema21 = (close.tail(5) - e21.tail(5)).abs().min() / ltp < 0.02 \
            or (ltp / e21.iloc[-1] - 1) < 0.01
        reclaim = close.iloc[-1] > e21.iloc[-1] and close.iloc[-2] <= e21.iloc[-2]
        in_band = 42 <= rsi <= 70

        # B) 20-day high breakout with volume
        hi20 = close.rolling(20).max().iloc[-2]
        breakout = close.iloc[-1] > hi20 and float(df["volume"].iloc[-1]) > 1.5 * vol_avg

        if uptrend and in_band and (reclaim or near_ema21):
            setup, entry, stop = "EMA_PULLBACK", ltp, min(float(e21.iloc[-1]) - atr_d, ltp - 2 * atr_d)
        elif breakout:
            setup, entry, stop = "BREAKOUT_20D", ltp, max(ltp - 2 * atr_d, hi20 * 0.97)
        if setup is None:
            return None

        if entry - stop <= 0:
            return None
        # stop sanity: never tighter than 1.2×ATR, never wider than 3×ATR
        risk = entry - stop
        risk = max(risk, 1.2 * atr_d)
        risk = min(risk, 3.0 * atr_d)
        stop = entry - risk

        t1 = entry + risk * 2.0
        t2 = entry + risk * 3.0      # trailing after 2R instead of hard exit

        sig = Signal(symbol=ctx.symbol, side="BUY", style="CNC", entry=entry,
                     stop=stop, target1=t1, target2=t2, qty=0,
                     score=score, ml_prob=ml_prob, setup=setup, regime=regime)
        sig.rationale = [f"regime {regime}"] + macro_flags[:1]
        return sig
