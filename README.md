# AlphaDesk — Multi-Agent Trading Desk for Indian Equities (Dhan)

<p align="center">
  <b>Nine specialist agents · one disciplined book · paper-first execution</b><br>
  ML/DL prediction · News NLP · Time-series econometrics · Alternative data · Macro gates<br>
  Python 3.10+ · Dhan API v2 · FastAPI dashboard · Telegram alerts
</p>

---

## What this is

AlphaDesk is a **self-contained algorithmic trading desk** for NSE equities (Nifty-50
universe) that you run on your own machine against your own Dhan account. Every morning
it studies the market, and all through the session **nine specialist agents analyse each
stock in parallel** — market structure, Nifty levels, technicals, ML predictions,
time-series structure, news sentiment, alternative data, macro gates, and risk. Their
votes are fused into one composite decision, gated by hard risk rules, and executed on a
**paper broker** whose charge model mirrors Dhan's real brokerage/levy structure.

It decides per day — **intraday (MIS) or swing (CNC)** — based on the regime: trend
days produce opening-range and momentum intraday trades; choppy-but-bullish tape
produces EMA-pullback swing entries; bearish regimes either keep you flat or take
short, regime-aligned intraday trades. You see everything on a dark trading-desk
dashboard and get **Telegram messages** for every buy/sell with stop-loss, targets,
and daily/weekly P&L in rupees.

> **It will not make money by magic.** It is a disciplined framework with an
> explicit statistical edge-seeking process: the ML gate refuses entries without
> probability support, the risk agent caps every trade at 1% of equity, and the day
> stops at −2%. Train the models, run the backtest, paper-trade for weeks, and only
> then consider real capital. See [Honest limitations](#honest-limitations).

---

## The nine agents

| # | Agent | Reads | Decides |
|---|-------|-------|---------|
| 1 | **MarketStructure** | NIFTY daily, EMA 20/50/200 stack, ADX(14), breadth, India VIX | Regime: `STRONG_TREND_UP / TREND_UP / NEUTRAL / TREND_DOWN / STRONG_TREND_DOWN / HIGH_VOL` — the master gate |
| 2 | **NiftyLevels** | Prior-day H/L/C, weekly pivots, opening range | CPR (narrow ⇒ trend-day odds), classic pivots R1–S3, PDH/PDL, session bias |
| 3 | **Technical** | Multi-timeframe OHLCV | EMA stack, RSI sweet-spot, MACD momentum, ADX quality, volume z-score, VWAP context |
| 4 | **MLPrediction** | 24 engineered features × 2 horizons | P(up) at h=1 (intraday gate) and h=5 (swing gate) — LightGBM + optional LSTM ensemble, disagreement-penalised |
| 5 | **TimeSeries** | Log-returns, price path | Hurst persistence (trend vs revert), EWMA volatility forecast (RiskMetrics λ=0.94), Holt 5-session forecast, Bollinger stretch |
| 6 | **Sentiment** | Google News RSS per symbol + market | Lexicon NLP with negation/intensifiers/crude-context, recency weighting, regulatory-event veto |
| 7 | **AltData** | Volume profile, 52-week range, NIFTY relative strength | Accumulation/distribution anomalies, FII/DII flow attempt, delivery% attempt |
| 8 | **Macro** | India VIX shock, USD/INR, Brent, S&P overnight | VIX spike ⇒ halve intraday size; VIX>25 ⇒ no swing books; FX/crude flags |
| 9 | **Risk** | Portfolio state | **The last word**: −2% daily kill-switch, 3-position cap, heat ≤ 3%, fixed-fractional sizing (1% risk/trade), notional & sector caps |

All nine run **concurrently** per scan cycle (`ThreadPoolExecutor`); an agent that
errors is normalised to a neutral `ERROR` report and never blocks the desk.

## Decision pipeline

```
  watchlist symbol ──► MarketContext (daily + 5-min bars, macro, news, features)
                              │
                              ▼
              ┌── 9 agents in parallel ──► AgentReports (score −100..+100)
              │
              ▼
     StrategyFusion      weighted composite · regime filter · ML probability gate
              │                 (score ≥ threshold AND P(up) ≥ 0.56/0.58)
              ▼
     Setup construction   ORB_15M · VWAP_PULLBACK · ORB_SHORT · EMA_PULLBACK · BREAKOUT_20D
              │
              ▼
     Risk sizing          qty = floor(1% equity ÷ risk-per-share), notional ≤ 25%, sector ≤ 40%
              │
              ▼
     PaperBroker          fills at LTP ± 5 bps slippage, full STT/GST/stamp levy model
              │
              ▼
     Telegram + Dashboard trade cards, exits, daily & weekly digests
```

Trade construction: ATR-based stops (2×ATR daily for swing, 0.6×ATR(5m) or the
opening-range edge for intraday), 1.5R/2.5R intraday targets with a 60% T1 partial,
2R/3R swing targets with EMA20 trailing after T1, 12-session time exit for swings,
15:10 IST forced square-off for all MIS positions.

---

## Quick start

```bash
git clone https://github.com/Bilpes/TradingBot.git
cd TradingBot

python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                                    # then edit .env (below)
python scripts/setup_check.py                           # validates everything
python main.py                                          # desk on http://localhost:8000
```

No credentials yet? It still runs:

```bash
python main.py --simulate        # fully offline synthetic market — demo/CI mode
```

Feed priority (`MARKET_DATA_MODE=auto`): **Dhan → yfinance → sim**, with automatic
fallback if a feed fails mid-session.

### `.env` — what to paste

| Key | Where to get it |
|-----|-----------------|
| `DHAN_CLIENT_ID` | Your Dhan client id (e.g. `1101234567`) |
| `DHAN_ACCESS_TOKEN` | web.dhan.co → profile → **DhanHQ Trading APIs** → Generate Access Token. **Expires (usually daily)** — regenerate when the log shows 401 |
| `TELEGRAM_BOT_TOKEN` | Telegram → @BotFather → `/newbot` → copy token |
| `TELEGRAM_CHAT_ID` | Telegram → @userinfobot → your numeric id |

`scripts/setup_check.py` sends a Telegram test message and makes a live Dhan
funds call so you know everything works **before** the first session.

---

## Making it smart: train the ML models

The bot works from day one (a transparent cold-start heuristic drives the ML gate),
but you want the real ensemble:

```bash
python scripts/train_models.py                       # 5 liquid names, horizons 1 & 5
python scripts/train_models.py --all                 # whole Nifty-50 universe
python scripts/train_models.py --with-lstm           # + LSTM leg (needs torch)
pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional
```

The trainer is leakage-safe by construction: features at row *t* use only data ≤ *t*,
labels are *future* returns, and validation is **chronological walk-forward with an
embargo gap** (no shuffled K-fold leakage — the most common way retail ML trading
projects lie to themselves). Out-of-fold AUC prints per symbol; artifacts land in
`data/models/` and the MLPrediction agent picks them up on next restart
(dashboard badge switches from `coldstart` to `ensemble`).

Interpretation: daily-direction AUC ≈ 0.52–0.55 is a usable edge after costs;
0.57+ is strong. If you see 0.65+ on daily data, be suspicious — hunt leakage first.

Backtest the swing logic against real history:

```bash
python scripts/backtest.py --symbols RELIANCE,INFY,TCS,SBIN --capital 100000
```

---

## The dashboard

`http://localhost:8000` — dark trading desk, auto-refreshing every 4 s:

- **KPI strip** — equity, day P&L, total P&L (net of charges), unrealised MTM,
  exposure %, portfolio heat, win rate — all in ₹
- **Agent Brain** — 9 live cards: status dot, live score, plain-English note, latency
- **Open positions** — side, style (MIS/CNC), qty, avg, LTP, SL, T1, live ₹ P&L
- **Signals feed** — every fused decision with its rationale and ML probability;
  rejected ideas stay visible with the reason (great for learning what the desk refuses)
- **Watchlist manager** — add/remove NSE symbols at runtime
- **Equity curve**, **trade log** (entry/exit/exit-reason/net ₹), **desk controls**
  (scan now · force square-off · halt/resume)

## Telegram alerts

- **Entry card** — side/qty/entry, stop-loss (₹ risk shown), T1/T2, setup, regime,
  ML probability, top-3 agent rationale
- **Exit card** — net P&L after charges, hold duration, runner status
- **Daily digest** (15:25 IST) — day P&L, open positions, lifetime stats
- **Weekly digest** (Saturday) — week realised, equity move, win rate/PF, model modes
- **Risk alerts** — kill-switch trips, manual halts

## Going live (read this twice)

Execution is **paper by default and always on this build's main path**. The live
Dhan order path (`DhanClient.place_order`) exists and is wired, but is triple-gated:

1. `LIVE_TRADING=true` in `.env`
2. a `data/LIVE_CONFIRM` file you must create yourself
3. per-day order caps in `config.py` (CHARGES/limits block)

**Before flipping it**, please:

- Run paper mode for at least 3–4 weeks and compare fills against Dhan's real quotes
- Understand SEBI's algo-trading framework for retail: orders via broker APIs are
  subject to exchange-mapped algo registration / broker policy. **Check Dhan's
  current algo policy and complete any required registration before automating orders.**
- Start with 1 lot / 1 share and a hard daily loss limit you can emotionally afford

## Honest limitations

- **Long-only swing book.** CNC shorts don't exist for retail equity delivery in
  India; the intraday short path is regime-gated and conservative.
- **Sizing caps bind before risk sizing** at ₹1L capital — the 25% notional cap
  often reduces the 1%-risk quantity. That is intentional (concentration risk is
  the silent killer of small accounts).
- **FII/DII + delivery%** fetchers attempt NSE endpoints without cookies and fall
  back to neutral when blocked (best-effort alternative data, clearly flagged).
- **Nifty-50 membership** ships as a static list (`config/nifty50.json`) — NSE
  reshuffles semi-annually; update it each January/July.
- **NSE holidays** are not built in (weekends only); add dates to avoid harmless
  failed calls on holidays, or rely on the feed fallback.
- **yfinance** intraday is delayed ~15 min — fine for swing decisions, not for
  scalping precision; use Dhan for live intraday.
- The ML ensemble is trained on daily bars. Intraday h=1 "probability" is a
  next-session edge proxy, not a tick-level forecast.

## Project structure

```
tradingbot/
├── main.py                    # entry point (--simulate | --once | --port)
├── config/
│   ├── nifty50.json           # universe (editable)
│   └── settings.py (config)   # all constants: weights, charges, hours, limits
├── tradingbot/
│   ├── config.py  logger.py  models.py  indicators.py
│   ├── broker/    dhan_client.py · instruments.py · paper.py
│   ├── data/      market_data.py · simulator.py · store.py · fetchers.py
│   ├── agents/    9 agents + base + parallel registry
│   ├── ml/        features.py · gbdt.py · lstm.py (optional torch) · predictor.py
│   ├── nlp/       lexicon.py · news.py (RSS)
│   ├── engine/    orchestrator.py · strategy.py · portfolio.py · scheduler.py
│   ├── notify/    telegram.py
│   ├── api/       server.py (FastAPI)
│   └── web/       index.html (dark trading desk)
├── scripts/       train_models.py · backtest.py · setup_check.py
└── tests/         indicators · charges/sizing/fusion · end-to-end engine cycle
```

## Roadmap

- [ ] Options structures (hedged swing: long futures + put spreads) when F&O access matures
- [ ] FinBERT sentiment upgrade (`transformers`) behind the existing lexicon
- [ ] Portfolio correlation cap (currently sector cap only)
- [ ] NSE holiday calendar auto-fetch
- [ ] Live broker class with order-state reconciliation + Telegram /commands

## Disclaimer

Educational software, not investment advice. Trading equities involves risk of
loss including loss of principal. Past performance of any model is not
indicative of future results. You are responsible for compliance with SEBI,
exchange and broker rules that apply to you.

---
<p align="center"><sub>Built with too much respect for the Indian market's ability to humble clever models.</sub></p>
