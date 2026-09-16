# TradingBot

A multi-agent, sector-aware equity strategy engine for **NSE**, wired for Dhan
market data, with a paper-trading book, Telegram alerts, a process-per-sector
worker model, and a dashboard that records every decision and the reasoning
behind it.

```
   Dhan / CSV / synthetic
            │
   ┌────────▼─────────┐   ┌──────────────────┐   ┌──────────────────┐
   │ worker: Banking  │   │ worker: IT       │   │ worker: Metal    │  ... 12 sectors,
   │ rsi macd ma vol  │   │ rsi macd ma vol  │   │ rsi macd ma vol  │      one OS process each
   └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘
            │        ┌─────────────▼──────────────┐       │
            │        │ worker: __market__         │       │
            │        │ sector_rotation, momentum, │       │
            │        │ vol_regime  (cross-section)│       │
            │        └─────────────┬──────────────┘       │
            └──────────────────────┼──────────────────────┘
                                   │  SQLite (WAL) — the bus and the record
                     ┌─────────────▼──────────────┐
                     │ Coordinator (main app)     │  consensus + quorum
                     │ risk sizing, paper broker  │  NSE charges, ₹1L book
                     └─────────────┬──────────────┘
                          ┌────────┴────────┐
                     Telegram alerts    Dashboard
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Simulate a paper run (no credentials, no waiting two weeks)
python scripts/paper_walk.py --sessions 500 --steps 150 --fresh --db runs/paper.sqlite3

# 2. Look at every decision it made and why
python -m tradingbot.cli dashboard --db runs/paper.sqlite3      # http://0.0.0.0:8000

# 3. One live-shaped cycle
python -m tradingbot.cli paper-step --db runs/paper.sqlite3

pytest          # 200 tests
```

## Going live on Dhan

```bash
export DHAN_CLIENT_ID=...          # from the Dhan console
export DHAN_ACCESS_TOKEN=...
export TELEGRAM_BOT_TOKEN=...      # from @BotFather
export TELEGRAM_CHAT_ID=...

python scripts/make_security_map.py --scrip-master dhan_scrip_master.csv \
    --out config/security_map.json
python -m tradingbot.cli dhan-check --symbol RELIANCE
python -m tradingbot.cli telegram-check
python -m tradingbot.cli paper-step --provider dhan --security-map config/security_map.json
```

**Security IDs are never hardcoded.** They are resolved from Dhan's official
scrip master, and an unresolved symbol is a hard startup error — a wrong
security id means buying the wrong stock, which is worse than refusing to boot.

## The agents

Per-name, computed inside each sector's process:

| Agent | Question |
|---|---|
| `rsi` | Stretched — and is the regime ranging or trending? (regime-aware, so it does not fade a real trend) |
| `macd` | Is histogram momentum building or fading? |
| `moving_average` | Is the 20/50/200 ribbon aligned, and did it just cross? |
| `volume_spike` | Did volume confirm the move? (a spike on a **down** day votes negative) |
| `sentiment` | Earnings/news score, plus a conviction haircut into known earnings dates |

Cross-sectional, computed once over the whole universe by the `__market__`
worker: `sector_rotation`, `momentum`, `vol_regime` (a gate, not a voter).

**Why that split exists.** Hand a cross-sectional agent a single sector's slice
and it computes a standard deviation across one column — `NaN`. Worse,
within-sector momentum ranks four names against each other, so the "best bank"
looks bullish while the entire banking sector falls. Those agents run in one
market-wide process instead.

## Sector-specific configuration

Every sector has its own profile: which agents run, at what weight, with what
parameters. Banking leans on MACD and moving averages; FMCG leans on mean
reversion and RSI; metals weight volume confirmation; telecom weights sentiment.

```bash
python -m tradingbot.cli profiles-dump --out profiles.json   # then edit it
python -m tradingbot.cli backtest --profiles profiles.json
```

## The confidence engine

Confidence = `mean voter confidence × exp(-dispersion / tau) × gates`. The
dispersion term is the point: a unanimous weak signal and a violently split
strong signal average to the same score but deserve completely different
treatment.

A trade then needs **all** of these (`ConsensusConfig`):

* `min_voters` — enough agents spoke at all
* `min_agreeing` — enough agree on the sign, above `vote_confidence`
* `max_dissent_confidence` **and** `min_dissent_score` — a confident *and*
  substantial dissenter vetoes outright (an agent scoring −0.03 is not
  dissenting, it is seeing nothing, and must not be allowed to veto)
* `min_confidence`, `min_abs_score`

Every rejection is stored with the rule that rejected it. The dashboard has a
panel for ideas the system declined, because that is half its behaviour.

## Paper trading

₹1,00,000 starting capital, long-only, delivery (CNC) by default. Charges
modelled line by line — STT, exchange transaction, SEBI, stamp duty, GST on
brokerage, DP charges, plus a slippage assumption — because a strategy with a
thin edge is often entirely consumed by costs.

**Nothing places a real order.** Live trading means constructing
`DhanPaperOrders` with a real `dhanhq.Order` — a separate, deliberate step.

## Read this before trusting any number

**Every market-data and messaging host is blocked in the sandbox this was built
in.** `api.dhan.co`, `dhan-api.dhan.co`, `api.telegram.org` and `nseindia.com`
all return connection failures. So:

* `DhanProvider` and `AlpacaProvider` have their request construction and
  response parsing unit-tested against fixtures, but **have never reached their
  live endpoints**. Run `dhan-check` / `telegram-check` yourself.
* Every number below comes from `SyntheticProvider` — factor-model paths, not
  market data.

### What the simulated paper run actually showed

150 sessions on synthetic data (seed 13, window 2022-01-03 → 2022-07-29):

```
Final equity     ₹93,329.01        Return            -6.67%
Closed trades    30                Win rate           6.7%
Charges paid     ₹1,017.20         Brier score        0.298
```

The equal-weight universe fell **−25.16%** over that same window, with **all 45
names down**. A long-only book losing −6.67% into that is the risk layer
working, not the strategy earning anything.

**The confidence number is not calibrated.** The 20–40% bucket realised a 52.6%
win rate, and a Brier score of 0.298 is worse than a coin flip (0.25). The
machinery to *measure* this is the deliverable here; the current answer it
returns is "these confidences do not yet mean what they claim." Do not size
positions on them until that table looks diagonal on your own data.

Caveat on that measurement too: it was taken over a window where everything
fell, so the base rate was extreme. Re-measure across regimes.

### Bugs this surfaced, in the order they were found

Each one was a real defect, caught by running the system rather than by reading
it:

1. **The book bought bearish signals.** `actionable` is direction-agnostic — a
   strongly agreed *short* is as actionable as a long. With no long-only gate
   the coordinator bought `TCS` at score −0.63 and `GRASIM` at −0.77, and posted
   a **0% win rate over 40 trades**: reliably long exactly what its own agents
   expected to fall.
2. **851 trades in 150 sessions, ₹19,639 in charges (19.6% of capital).** The
   coordinator treated "this symbol emitted no signals" as an exit signal, and
   had no hysteresis. Silence is not bearish. Fix: 851 → 40 trades.
3. **Every exit was a stop; zero signal exits.** Entry needed score ≥ +0.20,
   exit needed < 0, and the static stop sat ~7% below entry — so the stop always
   won and a winner could never be locked in. Added trailing stops.
4. **`horizon_days` was written and then ignored.** Predictions settled on the
   next session, grading a five-session confidence claim on a one-session
   outcome. That reported Brier 0.101; measured correctly it is 0.298. The bug
   made the system look calibrated when it was not.
5. **The default quorum made the system trade zero times** — 0 of 45 decisions
   actionable at `min_confidence=0.35`. Tuned against *how often it acts*, never
   against returns.
6. **Cross-sectional agents produced `NaN` inside sector workers** (see above).
7. **`indicators.cross` re-fired the golden cross every bar.** `shift()` on a
   bool Series yields object dtype, where `~True` is `-2` — truthy.
8. **`hash()` broke reproducibility.** CPython salts str hashing per process, so
   the "seeded" generator produced a different universe on every invocation.
   Now `zlib.crc32`, with a regression test that spawns subprocesses because an
   in-process comparison cannot catch it.

## Guarantees enforced by tests

* **No lookahead** — agents are re-fitted on data truncated at date *t* and must
  give a bit-identical answer at *t*.
* **Hard limits** — position count, per-name weight, per-sector weight, gross
  exposure. Queued exits are *not* credited as freed slots.
* **Long-only invariant** — no buy may be recorded against a negative score.
* **Costs are real** — both sides, net P&L, no gross figures anywhere.
* **Reproducibility across processes**, not just within one.
* **Workers are isolated** — a symbol is handled by exactly one sector, and one
  sector's slice is measurably cheaper than the whole universe.

## Layout

```
tradingbot/
  agents/      10 agents + sector profile registry
  core/        confidence, consensus quorum, risk, portfolio, engine, metrics
  data/        schema, providers (synthetic / CSV / Alpaca), Dhan client
  indicators/  indicators.py — causal RSI, MACD, MAs, ATR, volume, Bollinger
  markets/     nse.py — NSE universe by sectoral index
  notify/      telegram.py
  paper/       broker.py — ₹ book with the NSE cost model
  store/       db.py — SQLite: signals, decisions, fills, predictions, alerts
  web/         app.py + dashboard.html
  workers/     sector_worker.py (per-sector + market-wide), coordinator.py
  live.py      the paper runner; cli.py the entry point
scripts/       paper_walk.py, seed_scan.py, make_security_map.py
tests/         200 tests
```

## What is not here

Survivorship-bias-free history, walk-forward parameter selection, corporate
action adjustment, an NSE holiday calendar (a holiday is currently
indistinguishable from a data outage), intraday execution, and short selling.
The sentiment agent has no live news feed — it ships with the plumbing and a
static source, because an agent reporting sentiment it never observed is worse
than no agent.

## Not investment advice

This is engineering scaffolding. It has never traded real money.
