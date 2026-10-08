"""No network needed. Covers: no future fields in the prompt, cache round-trip, and
the analysis math (group stats, shuffle baseline) on synthetic verdicts."""

from __future__ import annotations

import json

from apex.llm_filter import (
    TradeContext,
    build_prompt,
    build_report,
    JudgeRun,
    judge_trade,
    load_cache,
    shuffle_baseline,
    split_approve_reject,
    trade_hash,
)

FORBIDDEN_SUBSTRINGS = (
    "exit", "pnl", "profit", "p_n_l", "realized", "stop_price", "target_price", "future",
)


def _sample_ctx(**overrides) -> TradeContext:
    base = dict(
        symbol="MNQ", setup="RMA", direction="LONG", entry_ts="2026-01-05T10:00:00-05:00",
        day_of_week="Monday", session_phase="open",
        bars=[{"ts": "2026-01-05T09:00:00-05:00", "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 100.0}],
        indicators={"close": 1.5, "vwap": 1.4, "vwap_distance": 0.1, "ema9": 1.3, "ema15": 1.2,
                   "ema21": 1.1, "ema30": 1.0, "ema65": 0.9, "ema200": 0.8, "rsi": 55.0,
                   "adx": 30.0, "atr": 0.2, "rvol": 1.8},
    )
    base.update(overrides)
    return TradeContext(**base)


def test_prompt_has_no_future_fields():
    ctx = _sample_ctx()
    prompt = build_prompt(ctx)
    text = prompt["user"].lower()  # the data payload, not the fixed instruction text
    for bad in FORBIDDEN_SUBSTRINGS:
        assert bad not in text, f"prompt leaked a future/outcome field: {bad!r}"
    # sanity: the entry-only fields we DO expect are present
    assert "entry_time_et" in prompt["user"]
    assert "indicator_snapshot" in prompt["user"]


def test_prompt_dict_keys_are_entry_only():
    ctx = _sample_ctx()
    prompt = build_prompt(ctx)
    user = json.loads(prompt["user"])
    allowed = {"symbol", "setup", "direction", "entry_time_et", "day_of_week",
              "session_phase", "last_20_bars", "indicator_snapshot"}
    assert set(user.keys()) == allowed


def test_cache_round_trip(tmp_path):
    cache_path = tmp_path / "responses.jsonl"
    ctx = _sample_ctx()
    run = JudgeRun(provider="ollama", model="stub")
    calls = {"n": 0}

    def fake_call(prompt):
        calls["n"] += 1
        return {"verdict": "APPROVE", "conviction": 5, "reason": "test"}, 0.0

    import apex.llm_filter as m
    orig = m._call_ollama
    m._call_ollama = lambda prompt: fake_call(prompt)
    try:
        cache = load_cache(cache_path)
        row1 = judge_trade(ctx, run, cache, path=cache_path)
        assert calls["n"] == 1
        assert row1["verdict"] == "APPROVE"

        # rerun with a fresh cache load from disk: must hit cache, not call the provider
        cache2 = load_cache(cache_path)
        row2 = judge_trade(ctx, run, cache2, path=cache_path)
        assert calls["n"] == 1  # no second call
        assert row2 == row1
    finally:
        m._call_ollama = orig


def test_trade_hash_stable_and_sensitive():
    ctx = _sample_ctx()
    h1 = trade_hash(ctx)
    h2 = trade_hash(_sample_ctx())
    assert h1 == h2
    h3 = trade_hash(_sample_ctx(direction="SHORT"))
    assert h3 != h1


def test_split_and_group_stats_math():
    records = [
        {"symbol": "MNQ", "verdict": "APPROVE", "conviction": 5, "pnl": 10.0},
        {"symbol": "MNQ", "verdict": "APPROVE", "conviction": 4, "pnl": -5.0},
        {"symbol": "MNQ", "verdict": "REJECT", "conviction": 2, "pnl": -20.0},
        {"symbol": "MES", "verdict": "REJECT", "conviction": 1, "pnl": 3.0},
    ]
    approved, rejected = split_approve_reject(records)
    assert approved == [10.0, -5.0]
    assert rejected == [-20.0, 3.0]


def test_shuffle_baseline_p_value_in_range():
    records = [{"symbol": "MNQ", "verdict": "APPROVE" if i % 2 == 0 else "REJECT",
               "conviction": 5, "pnl": float(i)} for i in range(20)]
    result = shuffle_baseline(records, n_resamples=200, seed=1)
    assert result["real_gap"] is not None
    assert 0.0 <= result["p_value"] <= 1.0


def test_shuffle_baseline_handles_single_group():
    records = [{"symbol": "MNQ", "verdict": "APPROVE", "conviction": 5, "pnl": 1.0}]
    result = shuffle_baseline(records)
    assert result["real_gap"] is None
    assert result["p_value"] is None


def test_build_report_underpowered_on_small_n():
    records = [{"symbol": "MNQ", "verdict": "APPROVE", "conviction": 5, "pnl": 1.0},
              {"symbol": "MNQ", "verdict": "REJECT", "conviction": 2, "pnl": -1.0}]
    run = JudgeRun(provider="ollama", model="qwen2.5-coder:14b", spent=0.0, calls=2)
    report = build_report(records, run)
    assert "UNDERPOWERED" in report
