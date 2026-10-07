"""Deterministic WSGTA setup detection and indicator sanity. Offline."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from apex.calendar import at_et
from apex.config import EvaluationType, SetupParams, get_tier
from apex.execution import build_bracket
from apex.market import Bar, Indicators, adx, ema, fixture_bars, rsi
from apex.setups import _bars_since_cross, _dbdt, _trend, detect_setup

pytestmark = pytest.mark.unit
DAY = date(2026, 10, 7)


def _bars(closes, start=None, rng=0.5, vol=1000.0):
    start = start or at_et(DAY, 6, 0)
    out = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i else c
        out.append(Bar(start + timedelta(minutes=5 * i), o, max(o, c) + rng, min(o, c) - rng, c, vol))
    return out


def test_fixture_is_an_rma_long_with_adx_above_20():
    for sym in ("MNQ", "MES"):
        sig = detect_setup(fixture_bars(sym, DAY), sym)
        assert sig.setup == "RMA" and sig.direction == "LONG"
        assert sig.confluence["adx"] >= 20


def test_ffma_fades_rsi_above_80():
    bars = _bars([6000 + i for i in range(120)])
    sig = detect_setup(bars, "MES")
    assert sig.setup == "FFMA" and sig.direction == "SHORT"


def test_ffma_fades_rsi_below_20():
    bars = _bars([6000 - i for i in range(120)])
    sig = detect_setup(bars, "MES")
    assert sig.setup == "FFMA" and sig.direction == "LONG"


def test_too_few_bars_is_no_setup():
    assert detect_setup(_bars([6000.0] * 10), "MES").setup is None


def test_flat_market_has_no_setup():
    sig = detect_setup(_bars([6000.0] * 120, rng=0.25), "MES")
    assert sig.setup is None


def _ind(ema9, ema15, atr=10.0):
    n = len(ema9)
    z = [0.0] * n
    return Indicators(close=z, ema9=ema9, ema15=ema15, ema21=z, ema30=z, ema65=z, ema200=z,
                      rsi=[50.0] * n, adx=[30.0] * n, atr=[atr] * n, vwap=z, rvol=[1.0] * n)


@pytest.mark.parametrize("bars_ago,hit", [(0, True), (3, True), (4, False)])
def test_trend_does_not_chase_more_than_3_bars_past_crossover(bars_ago, hit):
    # ema9 crosses above ema15 `bars_ago` bars before the last bar, ribbon compressed.
    n = 20
    cross_at = n - 1 - bars_ago
    e15 = [100.0] * n
    e9 = [99.5 if i < cross_at else 100.5 for i in range(n)]
    since, direction = _bars_since_cross(e9, e15)
    assert since == bars_ago and direction == "LONG"
    assert (_trend(None, _ind(e9, e15), SetupParams()) is not None) == hit


def test_trend_requires_compressed_ribbon():
    n = 20
    e15 = [100.0] * n
    e9 = [99.0] * (n - 1) + [105.0]  # 5 pts apart, ATR 10 -> not compressed (k=0.15)
    assert _trend(None, _ind(e9, e15), SetupParams()) is None


def _w_pattern(height, base=6000.0):
    path = []
    for i in range(40):
        if i <= 10:
            v = base + (10 - i) * height / 10
        elif i <= 20:
            v = base + (i - 10) * height / 10
        elif i <= 30:
            v = base + (30 - i) * height / 10
        else:
            v = base + (i - 30) * height / 20
        path.append(v)
    start = at_et(DAY, 9, 0)
    return [Bar(start + timedelta(minutes=5 * i), v, v + 0.25, v, v + 0.1, 1000.0) for i, v in enumerate(path)]


@pytest.mark.parametrize("height,hit", [(12.0, True), (10.0, True), (8.0, False)])
def test_double_bottom_needs_10_points_on_es(height, hit):
    res = _dbdt(_w_pattern(height), None, SetupParams(), "MES")
    assert (res is not None and res[0] == "LONG") == hit


def test_double_bottom_needs_30_points_on_nq():
    assert _dbdt(_w_pattern(20.0, 20000.0), None, SetupParams(), "MNQ") is None
    assert _dbdt(_w_pattern(32.0, 20000.0), None, SetupParams(), "MNQ") is not None


def test_indicators_basic_properties():
    assert ema([1.0, 1.0, 1.0], 3) == [1.0, 1.0, 1.0]
    assert rsi([float(i) for i in range(30)])[-1] == 100.0
    assert adx(_bars([6000.0 + i for i in range(60)]))[-1] > 20


# --------------------------------------------------------------------- bracket


def test_eod_bracket_mnq_defaults():
    b = build_bracket("MNQ", "LONG", 2, 20000.0, 16.0, EvaluationType.EOD)
    assert b["stop"]["price"] == 19984.0
    c1, c2 = b["targets"]
    assert (c1["contracts"], c1["price"]) == (1, 20012.0)
    assert c2["contracts"] == 1
    assert b["stop_management"]["after_c1_fill_stop_to"] == 20000.25  # breakeven + 1 tick
    assert b["risk_dollars"] == 64.0


def test_eod_bracket_mes_short():
    b = build_bracket("MES", "SHORT", 1, 6000.0, 4.0, EvaluationType.EOD)
    assert b["side"] == "SELL" and b["stop"]["price"] == 6004.0
    assert b["targets"][0]["price"] == 5997.5


def test_legacy_bracket_scales_70_percent_at_1_5r_and_breakeven_at_1r():
    b = build_bracket("MES", "LONG", 10, 6000.0, 4.0, EvaluationType.LEGACY)
    t1, runner = b["targets"]
    assert (t1["contracts"], t1["price"]) == (7, 6006.0)
    assert runner["contracts"] == 3
    assert b["stop_management"]["move_to_breakeven_at"] == 6004.0


def test_tier_table_values_match_design_doc():
    t = get_tier("50K")
    assert (t.nominal_size, t.profit_target, t.total_drawdown, t.max_contracts_mini,
            t.max_contracts_micro, t.daily_loss_limit) == (50_000, 3_000, 2_500, 10, 100, 650)
    assert get_tier("300k").max_contracts_micro == 350
    with pytest.raises(ValueError):
        get_tier("75K")
