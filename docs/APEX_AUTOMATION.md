# Apex Institutional TradingAgents Automation (DRY-RUN ONLY)

This layer sits on top of TradingAgents and automates one decision cycle for Apex-funded
futures evaluation accounts (MES, MNQ, ES, NQ). It is **dry-run only**:

- There is no broker integration. No Tradovate, NinjaTrader or Rithmic calls are made and no webhook fires.
- The execution "dispatcher" writes the bracket payload as JSON to disk and to the log. Nothing else.
- `run_apex_automation.py` rejects any `--mode` other than `dry-run`.
- **Deterministic Python enforces every risk rule.** LLMs write narrative and propose a rating. Code
  parses that rating strictly and checks it against the detected setup, and the
  Apex Risk Governor alone decides whether an order exists and how big it is.

> **Tier numbers follow Apex's help-center evaluation pages (read 2026-10-07).** The micro allowance and the evaluation freeze are still unverified. See [Doc discrepancies](#doc-discrepancies).

## Six-stage cycle

| # | Stage | When (ET) | Where | Status |
|---|-------|-----------|-------|--------|
| 1 | Pre-market ingest | 08:00 | `apex/market.py`, `apex/calendar.py`, `vault/raw/` | Interfaces plus fixtures. No live feed or scheduler yet |
| 2 | Pre-flight gatekeeping | per cycle | `vault_preflight_reader`, parallel `wsgta_technicals` / `macro_news_gate` / `social_sentiment`, `gatekeeper_router` | Built |
| 3 | Adversarial debate | per cycle | `bull_researcher` -> `bear_researcher` -> `research_manager` (up to 2 rounds) | Built |
| 4 | Risk governance | per cycle | `apex_risk_governor` <-> `contract_resizer` (loop capped at 2) | Built, deterministic |
| 5 | Dry-run execution | per cycle | `portfolio_manager` -> `execution_dispatcher` (JSON sink) | Built, dry-run only |
| 6 | Reflection into vault | 17:00 | `vault_post_trade_reflector` | Runs at the end of every cycle; no 17:00 scheduler yet |

### Graph topology (`apex/graph.py`)

```
START -> vault_preflight_reader
      -> [wsgta_technicals | macro_news_gate | social_sentiment]   (parallel fan-out)
      -> gatekeeper_router                                          (fan-in)
           news blackout or daily halt -> circuit_breaker_halt
           no setup or ADX < 20        -> terminal_no_trade
           else                        -> bull_researcher
bull_researcher -> bear_researcher -> research_manager
           actionable rating -> apex_risk_governor
           debate_round < 2  -> bull_researcher
           else              -> terminal_no_trade
apex_risk_governor
           APPROVED                 -> portfolio_manager
           RESIZE_MICRO (iter < 2)  -> contract_resizer -> apex_risk_governor
           else                     -> terminal_no_trade
portfolio_manager -> execution_dispatcher (DRY-RUN) -> vault_post_trade_reflector -> END
terminal_no_trade / circuit_breaker_halt            -> vault_post_trade_reflector -> END
```

State: `apex/state.py::WSGTAState` holds every field from the design, plus bookkeeping
(`order_symbol`, `order_requested_contracts`, `resize_iterations`, `risk_reasons`,
`terminal_reason`, `dispatch_path`, `vault_result`).

### Risk Governor (`apex/risk.py`)

The rules are listed below in evaluation order. Any one of them can reject; sizing comes last.

1. Account breached (equity <= trailing threshold): reject, flatten, lock.
2. Daily loss <= -`daily_loss_limit` (exactly -$1,000 on 50K halts): reject, flatten, lock until `new_session()`.
3. Session locked.
4. 3 consecutive stop-outs: session shutdown. 2 stop-outs: 60-minute cooldown from the last stop-out.
5. News lockout from T-5 to T+5 minutes, both ends inclusive, around FOMC/CPI/PPI/NFP/GDP (or any event marked high impact): reject and flatten.
6. Session clock: before 09:30 closed; 09:30-09:45 no entries; 09:45-11:30 prime; 11:30-13:30 half size;
   13:30-15:45 A+ only; 15:45-15:55 no new entries; at or after 15:55 mandatory flatten.
7. Consistency cap: day P&L >= `profit_target x consistency_cap_ratio` blocks new positions. Off for every Apex tier (evaluations have no consistency rule); set the ratio on a custom tier to enable it.
8. Within 15% of the daily circuit breaker (remaining budget <= 15% of the limit): reject.
9. Spread wider than 2 ticks: reject.
10. Sizing: budget = effective buffer x risk fraction, clamped to [1%, 2%], times the multipliers
    (midday x0.5, soft warning at 15% of the daily budget used x0.5). The budget is also capped
    so that a stop-out cannot cross the circuit-breaker margin.
    Contracts = min(requested, floor(budget / risk per contract), tier cap).
11. A mini that does not fit but whose micro does returns `RESIZE_MICRO`. The resizer converts
    it 10:1 and the governor checks it again. If nothing fits, the order is rejected.

Trailing drawdown (`AccountState`):

- **LEGACY**: the high-water mark follows intraday equity, unrealized included.
- **EOD**: the high-water mark moves only on `end_of_day()` with the realized balance.
- The threshold is `HWM - total_drawdown`, capped at `nominal + 100`. Once it reaches that
  level it is frozen permanently.
- Worked example (the design doc's 50K: $2,500 DD; the test fixture `DOC50K`), a trade that runs to +$800 and closes at +$200: LEGACY threshold
  48,300 (buffer 1,900, max risk $38); EOD threshold 47,500 intraday, then 47,700 at the close
  (buffer 2,500, max risk $50).

Contract math: MES $5/pt, MNQ $2/pt, ES $50/pt, NQ $20/pt, all with a 0.25 tick.

### Brackets (`apex/execution.py`)

- Limit entry at the 21 EMA retest. The stop sits `stop_points` away: defaults are 4.0 pts for MES/ES and 16.0 pts for MNQ/NQ.
  Set `BracketParams.atr_stop_multiple` to use `max(default, k x ATR)` instead.
- **EOD** accounts: C1 takes `ceil(n/2)` contracts at the fixed target (2.5 pts MES, 12.0 pts MNQ).
  After C1 fills, C2's stop moves to breakeven + 1 tick and C2 trails the 21 EMA.
- **LEGACY** accounts: 70% scales out at T1 = 1.5R, the stop moves to breakeven at 1R, and the runner trails the 21 EMA.

### Setups (`apex/setups.py`, deterministic)

| Setup | Rule |
|-------|------|
| RMA | Trend (ema21 vs ema65, price vs VWAP). The bar tags the 21 EMA and closes beyond the 30 EMA |
| FFMA | RSI > 80 means short, RSI < 20 means long (fade) |
| TREND | Compressed 9/15 ribbon (gap <= 0.15 ATR) that crossed no more than 3 bars ago |
| MOMO | RVOL >= 2, range >= 1.5 ATR, ADX >= 25, and a 9/15 cross in the same direction no more than 3 bars ago |
| DB_DT | Two swing extremes within 10% of the minimum height, at least 5 bars apart, with height >= 10 pts (ES) or 30 pts (NQ) |

Grade: A+ needs 4 of these 5 aligned: price vs VWAP, ema21 vs ema65, ema65 vs ema200, ADX >= 25, RVOL >= 1.5. A needs 3. Anything less is B.

### Vault (`apex/vault.py`)

```
vault/
  raw/            immutable: market_data/, news/, executions/YYYY-MM-DD.json (+ -2, -3 ... for later runs that day)
  wiki/           strategies/ (notes + stats.json + stats tables), entities/, accounts/, regimes/, post-mortems/YYYY-MM-DD.md
  index.md        regenerated catalogue + setup stats
  log.md          append-only "## [YYYY-MM-DD] <op> | <summary>"
```

Raw files are opened in exclusive-create mode, so an existing raw file can never be rewritten.
The preflight reader loads `wiki/entities/<SYMBOL>.md`, `wiki/accounts/<TIER>.md`, every strategy
note and the stats. The reflector writes the raw execution record, appends to the day's
post-mortem, updates decision counts, appends to `log.md` and regenerates `index.md`. Win rate and EV
update through `Vault.record_outcome(setup, pnl, r_multiple)` once a real fill outcome exists.
A dry run has none, so those figures stay `n/a`.

## How to run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# Fully offline: scripted LLMs, fixture bars, calendar, news and sentiment. No keys, no network.
python run_apex_automation.py --mode dry-run --tier 50K --eval-type EOD --offline --verbose

# Other knobs
python run_apex_automation.py --mode dry-run --tier 100K --eval-type LEGACY --symbol MES --offline --time 11:45
python run_apex_automation.py --mode dry-run --offline --account-json acct.json   # e.g. {"balance": 49350}

# With real LLMs (local Ollama by default) and recorded bars
python run_apex_automation.py --mode dry-run --tier 50K --eval-type EOD --symbol MNQ `
    --bars-csv bars.csv --calendar-json events.json --profile local

# Tests (offline)
python -m pytest tests/test_apex_*.py -q
```

Outputs: dry-run order JSON goes to `results/apex_dryrun/` (`--out`) and memory to `vault/` (`--vault`).
Both directories are gitignored.

## Config and env reference

| Setting | Where | Default |
|---------|-------|---------|
| Tier table | `apex/config.py::_TIERS`, `register_tier()` | Apex help-center values, 2026-10-07 |
| Risk knobs | `apex/config.py::RiskParams` | 2% risk fraction, [1%, 2%] clamp, 15% soft/circuit, 2-tick spread, 2/3-loss rules, +/-5 min news window, ADX 20, 2 debate rounds, 2 resize iterations |
| Bracket knobs | `BracketParams` | stops 4/16 pts, C1 2.5/12 pts, LEGACY 70% @ 1.5R, BE @ 1R |
| Setup knobs | `SetupParams` | DB/DT 10/30 pts, max 3 bars past cross, RSI 80/20 |
| `APEX_LLM_PROFILE` | env | `local` (all roles on Ollama: `qwen2.5-coder:14b` / `:7b`) |
| `APEX_ROLE_<ROLE>_PROVIDER/_MODEL/_BASE_URL` | env | per profile |
| `OLLAMA_BASE_URL` | env (upstream) | `http://localhost:11434/v1` |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` | env (upstream) | unset |
| `OPENAI_COMPATIBLE_API_KEY` | env (upstream) | Perplexity key when macro_news uses `openai_compatible` |

Roles: `technical`, `macro_news`, `sentiment`, `bull`, `bear`, `research_manager`, `risk_narrative`,
`portfolio_manager`. Real clients are built with upstream `tradingagents.llm_clients.create_llm_client`,
so every upstream provider works. The provider `stub` means no model: sentiment falls back to the
`JevSentimentStub` interface, which scores 0.

`cloud` profile: technical `anthropic/claude-sonnet-5`, macro_news `openai_compatible/sonar-pro`
at `https://api.perplexity.ai`, sentiment `stub`, bull `openrouter/nousresearch/hermes-4-405b`,
bear `openrouter/qwen/qwen3-235b-a22b`, research_manager `anthropic/claude-opus-5-5`, risk_narrative
`anthropic/claude-sonnet-5`, portfolio_manager `openai/gpt-6-sol`. The Anthropic and OpenAI IDs come from upstream's
model catalog. Verify the OpenRouter and Perplexity IDs before use.

Template: `.env.apex.example`. Copy it to `.env.apex` (gitignored), which `run_apex_automation.py` loads.

## Doc discrepancies

These are the places where this build deviates from the design documents, and why.

1. **`config["api_key"]` does not exist upstream.** TradingAgents reads keys from provider-specific env vars
   (`tradingagents/llm_clients/api_key_env.py`). Apex does the same.
2. **`config["custom_risk_context"]` does not exist upstream.** Risk is not a prompt context here.
   It lives in the deterministic governor, and LLMs only see its verdict.
3. **`config["debate_bull_llm"]` (and per-agent LLM keys) do not exist upstream.** Upstream has only
   `quick_think_llm` / `deep_think_llm` with optional per-tier provider and URL. Apex adds its own role map
   (`apex/llm.py`) and builds each role with upstream's `create_llm_client`.
4. **`tradingagents backtest` exists upstream, but it does something else.** It scores the equity graph's past
   decisions over a ticker/date grid. It is not a futures or Apex-rules backtest. No Apex backtest harness was built.
5. **`final_state["risk_assessment"]` does not exist upstream.** Upstream has `risk_debate_state`,
   `final_trade_decision` and `final_rating`. The Apex graph uses `risk_verdict` and `risk_reasons`.
6. **The upstream equity graph is not reused for futures.** Its analysts, prompts and tools are
   built around stocks and crypto (fundamentals, SEC filings). WSGTA is a separate LangGraph that reuses upstream's
   LLM client layer and LangGraph conventions.
7. **Models.** `claude-3-5-sonnet-20241022` is stale and was replaced with `claude-sonnet-5` / `claude-opus-5-5`
   from upstream's catalog. "GPT-6 Sol" does exist (`gpt-6-sol` in upstream's catalog) and is kept for the PM in the
   cloud profile. The doc's "Orca" is WizardLM; the bear role uses Qwen (`qwen/qwen3-235b-a22b` on OpenRouter) instead.
   "TypeSafe Jev" sentiment: no such API could be found, so it is a stub interface (`JevSentimentStub`) with
   `LLMSentiment` and `FixtureSentiment` as working implementations. Perplexity has no upstream provider, so it is
   reached through `openai_compatible` + `base_url`. That provider uses one global `OPENAI_COMPATIBLE_API_KEY`,
   which means two different keyed `openai_compatible` roles cannot coexist. The local Ollama profile is the default.
8. **Tier numbers corrected (2026-10-07).** Apex's own EOD and intraday evaluation pages, read in a browser
   (automated fetches get HTTP 403), list 25K: $1,500 target / $1,000 DD / 4 contracts / $500 DLL;
   50K: $3,000 / $2,000 / 6 / $1,000; 100K: $6,000 / $3,000 / 8 / $1,500; 150K: $9,000 / $4,000 / 12 / $2,000.
   The DLL applies to EOD evaluations only. There are no 250K/300K evaluations and no consistency rule or scaling
   during evaluation. The design doc's numbers ($2,500 DD, $650 DLL, 10 contracts, 30% cap) were replaced. Still
   unverified: the micro allowance (capped 1:1 with minis until Apex confirms) and whether the +$100 freeze applies
   during evaluations (third-party summaries tie it to PA accounts). Use `register_tier()` to override.
9. **Freeze interpretation.** "Trailing stops once equity >= initial + $100" is implemented as Apex
   describes it: the threshold stops trailing once it reaches `nominal + $100`, which happens when the peak
   balance reaches `nominal + DD + $100`.
10. **The consistency cap is applied as a self-imposed daily stop.** Apex's consistency rule is a payout rule, per
    third-party sources.
11. **The daily loss limit applies to both EOD and LEGACY accounts.** Third-party sources say Apex's DLL applies to EOD only.
12. **15:45-15:55 ET is unspecified** in the docs, so it is treated as "no new entries". The 15:55 flatten
    also blocks entries until the close. The clock covers RTH only, with no Globex session.
13. **The spread gate applies to every symbol,** not just NQ/ES. Micros trade the same tick grid.
14. **The "don't chase >3 bars past crossover" rule applies to the crossover-triggered setups** (TREND, MOMO).
    RMA, FFMA and DB/DT are not crossover entries.
15. **The setup definitions and the A/A+/B grading are concrete interpretations.** The docs name the setups but do not
    fully specify them.
16. **The ATR stop defaults to the fixed point values.** The ATR multiple is opt-in (`atr_stop_multiple`), because the docs give
    fixed defaults.
17. **C1/C2 split is `ceil(n/2)` / rest.** A single contract has C1 only and no runner.
18. **Added guard:** a research-manager rating that contradicts the setup direction, or cannot be parsed, becomes
    NEUTRAL.
19. **Sentiment is narrative and context only.** The docs gave no deterministic sentiment gate.
20. **"Flatten" in dry-run** is recorded as `order_action = HALT_FLATTEN` (or `flatten_required` on a governor decision).
    No positions exist to close.
21. **Same-day re-runs** write `raw/executions/YYYY-MM-DD-2.json` rather than mutating the existing raw file.

## Not yet built

- Broker webhook / FastAPI order receiver, and any live execution path. This is intentionally absent.
- Live market data (futures bars and quotes), live news, and a live economic calendar feed.
- Scheduler for the 08:00 ET ingest and 17:00 ET reflection (cron, APScheduler, or similar).
- Backtest harness for Apex rules over historical futures bars.
- VIX > 28 de-risk trigger.
- Apex promotion / rule-change monitoring.
- A real "TypeSafe Jev" sentiment integration.
- Position tracking across cycles (fills, open-position management, trailing the C2 stop live).
