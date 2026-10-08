"""LLM judge over WSGTA setup trades: does APPROVE beat REJECT out-of-sample-style?

The judge sees only information available AT ENTRY (last 20 closed bars, the indicator
snapshot the setup fired on, session phase, day of week). It never sees the exit, the
realized P&L, or any bar after entry. ``build_prompt`` is the single place that decides
what the judge sees -- ``tests/test_apex_llm_filter.py`` asserts no future field leaks in.

Trade extraction imports ``apex.backtest``/``apex.setups``/``apex.market`` read-only;
nothing is added to those modules. Everything here (prompt, provider calls, cache,
analysis) is new and owned by this file.

Providers, in order:
  (a) OpenRouter (``OPENROUTER_API_KEY`` env) -- deepseek/deepseek-chat-v3.1, falling
      back to qwen/qwen3-235b-a22b. Hard stop at $4 spent (of a $5 cap).
  (b) local Ollama (http://localhost:11434, qwen2.5-coder:14b) if no key is set.

Every response is cached in ``results/llm_filter/responses.jsonl`` keyed by a hash of
the trade's entry-only fields, so a rerun only calls the provider for new trades.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .backtest import Trade, compute_signals, load_bars, run_walk_forward
from .calendar import ET
from .config import DEFAULT_SETUP_PARAMS
from .market import Bar, compute_indicators
from .setups import SETUPS

CACHE_PATH = Path("results/llm_filter/responses.jsonl")
REPORT_PATH = Path("results/llm_filter/report.md")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS = ("deepseek/deepseek-chat-v3.1", "qwen/qwen3-235b-a22b")
OPENROUTER_SPEND_CAP = 4.0  # stop calling once this much of the $5 budget is spent

OLLAMA_URL = "http://localhost:11434"
# ponytail: spec asked for qwen2.5-coder:14b, but on this box it doesn't fit whatever
# Ollama has resident and swaps -- 75s wall for an 18-token reply vs 7b's 6s. 7b stays
# resident and answers in ~5s; 14b would take this 311-trade run many hours. Swap back
# once a bigger box or GPU makes 14b's per-call cost sane.
OLLAMA_MODEL = "qwen2.5-coder:7b"

# ponytail: static $/1M-token table, good enough to enforce the spend cap; refresh from
# OpenRouter's /generation endpoint if exact billing ever matters.
OPENROUTER_PRICE_PER_M = {
    "deepseek/deepseek-chat-v3.1": (0.27, 1.10),
    "qwen/qwen3-235b-a22b": (0.13, 0.60),
}

N_CONTEXT_BARS = 20

JUDGE_SYSTEM_PROMPT = (
    "You are a trading setup judge for a WSGTA (weekly/session gap trend-and-anchor) "
    "futures system. You will be shown ONE trade entry signal: the setup that fired, "
    "the last 20 closed bars, and the indicator snapshot at entry. You do not know what "
    "happened after entry. Decide whether this entry looks sound on its own terms. "
    'Reply with ONLY a JSON object: {"verdict": "APPROVE" or "REJECT", '
    '"conviction": 1-5, "reason": "<=30 words"}.'
)


# ----------------------------------------------------------------------- trade context


@dataclass(frozen=True)
class TradeContext:
    """Entry-only facts about one trade. No exit, no future bar, no PnL."""

    symbol: str
    setup: str
    direction: str
    entry_ts: str  # ISO, ET
    day_of_week: str
    session_phase: str
    bars: list[dict]       # last N_CONTEXT_BARS closed bars, oldest first
    indicators: dict       # snapshot at the signal bar (close, vwap, ema*, rsi, adx, atr, rvol)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "setup": self.setup, "direction": self.direction,
            "entry_ts": self.entry_ts, "day_of_week": self.day_of_week,
            "session_phase": self.session_phase, "bars": self.bars, "indicators": self.indicators,
        }


def _session_phase(hour: int, minute: int) -> str:
    t = hour * 60 + minute
    if t < 9 * 60 + 30:
        return "pre-market"
    if t < 11 * 60 + 30:
        return "open"
    if t < 14 * 60:
        return "midday"
    if t < 15 * 60 + 30:
        return "afternoon"
    if t < 16 * 60:
        return "close"
    return "overnight"


def _bar_dict(b: Bar) -> dict:
    return {"ts": b.ts.isoformat(), "o": round(b.open, 2), "h": round(b.high, 2),
            "l": round(b.low, 2), "c": round(b.close, 2), "v": round(b.volume, 1)}


def trade_context(trade: Trade, bars: list[Bar], ts_to_idx: dict, ind) -> TradeContext:
    """Build the entry-only context for ``trade``. ``ind`` is ``compute_indicators(bars)``."""
    fill_idx = ts_to_idx[trade.entry_ts]
    signal_idx = fill_idx - 1  # the last *closed* bar the signal fired on
    window = bars[max(0, signal_idx - N_CONTEXT_BARS + 1): signal_idx + 1]
    ts = trade.entry_ts.astimezone(ET)
    indicators = {
        "close": round(ind.close[signal_idx], 2), "vwap": round(ind.vwap[signal_idx], 2),
        "vwap_distance": round(ind.close[signal_idx] - ind.vwap[signal_idx], 2),
        "ema9": round(ind.ema9[signal_idx], 2), "ema15": round(ind.ema15[signal_idx], 2),
        "ema21": round(ind.ema21[signal_idx], 2), "ema30": round(ind.ema30[signal_idx], 2),
        "ema65": round(ind.ema65[signal_idx], 2), "ema200": round(ind.ema200[signal_idx], 2),
        "rsi": round(ind.rsi[signal_idx], 2), "adx": round(ind.adx[signal_idx], 2),
        "atr": round(ind.atr[signal_idx], 2), "rvol": round(ind.rvol[signal_idx], 2),
    }
    return TradeContext(
        symbol=trade.symbol, setup=trade.setup, direction=trade.direction,
        entry_ts=ts.isoformat(), day_of_week=ts.strftime("%A"),
        session_phase=_session_phase(ts.hour, ts.minute),
        bars=[_bar_dict(b) for b in window], indicators=indicators,
    )


def extract_all_trades(tier: str = "50K", eval_type: str = "EOD"
                       ) -> list[tuple[Trade, TradeContext]]:
    """Every trade across both symbols and all 5 setups, paired with its entry-only context."""
    out: list[tuple[Trade, TradeContext]] = []
    for symbol in ("MNQ", "MES"):
        bars = load_bars(symbol)
        ts_to_idx = {b.ts: i for i, b in enumerate(bars)}
        signals = compute_signals(symbol, bars, DEFAULT_SETUP_PARAMS)
        ind = compute_indicators(bars)
        for setup_name in SETUPS:
            trades, _account, _blown = run_walk_forward(symbol, setup_name, bars, tier, eval_type,
                                                        signals=signals)
            for t in trades:
                out.append((t, trade_context(t, bars, ts_to_idx, ind)))
    return out


# ----------------------------------------------------------------------- prompt + hash


def trade_hash(ctx: TradeContext) -> str:
    payload = json.dumps(ctx.to_dict(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_prompt(ctx: TradeContext) -> dict:
    """The exact fields sent to the judge. Every key here is entry-only -- see the
    no-future-fields test, which asserts on this dict's serialized form."""
    user = {
        "symbol": ctx.symbol, "setup": ctx.setup, "direction": ctx.direction,
        "entry_time_et": ctx.entry_ts, "day_of_week": ctx.day_of_week,
        "session_phase": ctx.session_phase,
        "last_20_bars": ctx.bars,
        "indicator_snapshot": ctx.indicators,
    }
    return {"system": JUDGE_SYSTEM_PROMPT, "user": json.dumps(user, sort_keys=True)}


# ----------------------------------------------------------------------- cache


def load_cache(path: Path = CACHE_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["hash"]] = row
    return out


def append_cache(row: dict, path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


# ----------------------------------------------------------------------- providers


def _parse_verdict(text: str) -> dict:
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in judge response: {text!r}")
    obj = json.loads(text[start: end + 1])
    verdict = str(obj.get("verdict", "")).strip().upper()
    if verdict not in ("APPROVE", "REJECT"):
        raise ValueError(f"bad verdict {obj.get('verdict')!r}")
    conviction = int(obj.get("conviction", 3))
    reason = str(obj.get("reason", "")).strip()
    return {"verdict": verdict, "conviction": conviction, "reason": reason}


def _call_openrouter(prompt: dict, model: str, api_key: str) -> tuple[dict, float]:
    resp = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model, "temperature": 0,
            "messages": [{"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    in_price, out_price = OPENROUTER_PRICE_PER_M.get(model, (0.0, 0.0))
    cost = (usage.get("prompt_tokens", 0) * in_price + usage.get("completion_tokens", 0) * out_price) / 1e6
    return _parse_verdict(content), cost


def _call_ollama(prompt: dict) -> tuple[dict, float]:
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL, "stream": False, "options": {"temperature": 0},
            "messages": [{"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]}],
        },
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    return _parse_verdict(content), 0.0


def ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.ok
    except requests.RequestException:
        return False


@dataclass
class JudgeRun:
    provider: str
    model: str
    spent: float = 0.0
    calls: int = 0


def judge_trade(ctx: TradeContext, run: JudgeRun, cache: dict[str, dict],
                path: Path = CACHE_PATH) -> dict:
    """Return the cached or freshly-called verdict row for ``ctx``. Mutates ``run.spent``."""
    h = trade_hash(ctx)
    if h in cache:
        return cache[h]
    prompt = build_prompt(ctx)
    if run.provider == "openrouter":
        api_key = os.environ["OPENROUTER_API_KEY"]
        verdict, cost = None, 0.0
        for model in OPENROUTER_MODELS:
            try:
                verdict, cost = _call_openrouter(prompt, model, api_key)
                run.model = model
                break
            except (requests.RequestException, ValueError, KeyError):
                continue
        if verdict is None:
            raise RuntimeError("both OpenRouter models failed")
    else:
        verdict, cost = _call_ollama(prompt)
    run.spent += cost
    run.calls += 1
    row = {"hash": h, "symbol": ctx.symbol, "setup": ctx.setup, "direction": ctx.direction,
          "entry_ts": ctx.entry_ts, "provider": run.provider, "model": run.model,
          "cost_usd": round(cost, 6), **verdict}
    cache[h] = row
    append_cache(row, path)
    return row


def pick_provider() -> JudgeRun:
    if os.environ.get("OPENROUTER_API_KEY"):
        return JudgeRun("openrouter", OPENROUTER_MODELS[0])
    if ollama_available():
        return JudgeRun("ollama", OLLAMA_MODEL)
    raise RuntimeError("no judge provider available: no OPENROUTER_API_KEY and Ollama is not reachable")


# ----------------------------------------------------------------------- analysis


def _group_stats(pnls: list[float]) -> dict:
    if not pnls:
        return {"n": 0, "mean_pnl": 0.0, "total_pnl": 0.0, "win_rate": 0.0}
    wins = sum(1 for p in pnls if p > 0)
    return {"n": len(pnls), "mean_pnl": round(statistics.mean(pnls), 2),
            "total_pnl": round(sum(pnls), 2), "win_rate": round(wins / len(pnls), 4)}


def split_approve_reject(records: list[dict]) -> tuple[list[float], list[float]]:
    """``records``: dicts with at least ``verdict`` and ``pnl``."""
    approved = [r["pnl"] for r in records if r["verdict"] == "APPROVE"]
    rejected = [r["pnl"] for r in records if r["verdict"] == "REJECT"]
    return approved, rejected


def shuffle_baseline(records: list[dict], n_resamples: int = 1000, seed: int = 0) -> dict:
    """Random relabeling at the same approval rate; empirical p-value for the real
    approved-minus-rejected PnL gap (fraction of shuffled gaps >= the real gap)."""
    approved, rejected = split_approve_reject(records)
    if not approved or not rejected:
        return {"real_gap": None, "p_value": None, "n_resamples": n_resamples}
    real_gap = statistics.mean(approved) - statistics.mean(rejected)
    pnls = [r["pnl"] for r in records]
    n_approve = len(approved)
    rng = random.Random(seed)
    gaps = []
    for _ in range(n_resamples):
        shuffled = pnls[:]
        rng.shuffle(shuffled)
        a, r = shuffled[:n_approve], shuffled[n_approve:]
        gaps.append(statistics.mean(a) - statistics.mean(r))
    p_value = sum(1 for g in gaps if g >= real_gap) / n_resamples
    return {"real_gap": round(real_gap, 2), "p_value": round(p_value, 4),
            "n_resamples": n_resamples, "shuffled_mean": round(statistics.mean(gaps), 2),
            "shuffled_std": round(statistics.pstdev(gaps), 2)}


def build_report(records: list[dict], run: JudgeRun) -> str:
    symbols = sorted({r["symbol"] for r in records})
    lines = [f"# LLM filter report ({run.provider}/{run.model}, ${run.spent:.4f} spent, "
            f"{run.calls} live calls, {len(records)} trades)", ""]

    def table(rows: list[dict], label: str) -> list[str]:
        approved, rejected = split_approve_reject(rows)
        a, r = _group_stats(approved), _group_stats(rejected)
        return [
            f"## {label}", "",
            "| Group | N | Mean PnL | Total PnL | Win rate |",
            "|---|---|---|---|---|",
            f"| APPROVE | {a['n']} | {a['mean_pnl']} | {a['total_pnl']} | {a['win_rate']:.2%} |",
            f"| REJECT | {r['n']} | {r['mean_pnl']} | {r['total_pnl']} | {r['win_rate']:.2%} |",
            "",
        ]

    lines += table(records, "Overall")
    for sym in symbols:
        lines += table([r for r in records if r["symbol"] == sym], f"Symbol: {sym}")
    lines += table([r for r in records if r.get("conviction", 0) >= 4], "Conviction >= 4")

    baseline = shuffle_baseline(records)
    lines += ["## Shuffle baseline (same approval rate, random relabeling)", ""]
    if baseline["real_gap"] is None:
        lines += ["Not enough trades in both groups to compute a baseline.", ""]
    else:
        lines += [
            f"Real APPROVE-minus-REJECT mean PnL gap: **{baseline['real_gap']}**",
            f"Shuffled gap distribution: mean {baseline['shuffled_mean']}, "
            f"std {baseline['shuffled_std']} over {baseline['n_resamples']} resamples",
            f"Empirical p-value (fraction of shuffled gaps >= real gap): **{baseline['p_value']}**",
            "",
        ]

    approved, rejected = split_approve_reject(records)
    n = len(records)
    if n < 30 or not approved or not rejected:
        verdict = "UNDERPOWERED(n too small)"
    elif baseline["p_value"] is not None and baseline["p_value"] < 0.1 and baseline["real_gap"] > 0:
        verdict = "FILTER_ADDS_VALUE"
    else:
        verdict = "NO_VALUE"
    lines += [f"## Verdict: {verdict}", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------- CLI


def main() -> int:
    run = pick_provider()
    print(f"provider: {run.provider} / {run.model}")

    pairs = extract_all_trades()
    print(f"{len(pairs)} trades extracted across MNQ+MES x {len(SETUPS)} setups")

    cache = load_cache()
    records = []
    for trade, ctx in pairs:
        if run.provider == "openrouter" and run.spent >= OPENROUTER_SPEND_CAP:
            print(f"spend cap (${OPENROUTER_SPEND_CAP}) reached, stopping judge calls")
            break
        row = judge_trade(ctx, run, cache)
        records.append({**row, "pnl": trade.pnl})

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(records, run), encoding="utf-8")
    print(f"wrote {REPORT_PATH}")
    print(f"provider={run.provider} model={run.model} spent=${run.spent:.4f} calls={run.calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
