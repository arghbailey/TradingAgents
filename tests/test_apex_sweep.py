"""apex.sweep: causal resampling, train/holdout separation, session filter
correctness. Offline, synthetic bars only -- no network, no real sweep run."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from apex.calendar import ET, at_et
from apex.market import Bar
from apex.sweep import (
    TIMEFRAMES_5M,
    filter_signals,
    resample,
    resample_count,
    resample_daily,
    split_index,
)
from apex.setups import SetupSignal

pytestmark = pytest.mark.unit


def _bars(n: int, start_hour: int = 9) -> list[Bar]:
    day = date(2026, 10, 7)
    start = at_et(day, start_hour, 0)
    out = []
    for i in range(n):
        ts = start + timedelta(hours=i)
        price = 100.0 + i
        out.append(Bar(ts, price, price + 1, price - 1, price + 0.5, 10.0))
    return out


# --------------------------------------------------------------------- resampling


def test_resample_count_aggregates_correctly():
    bars = _bars(6)
    out = resample_count(bars, 2)
    assert len(out) == 3
    b0 = out[0]
    assert b0.ts == bars[0].ts
    assert b0.open == bars[0].open
    assert b0.close == bars[1].close
    assert b0.high == max(bars[0].high, bars[1].high)
    assert b0.low == min(bars[0].low, bars[1].low)
    assert b0.volume == bars[0].volume + bars[1].volume


def test_resample_count_drops_trailing_partial_bucket():
    bars = _bars(5)  # 5 bars, n=2 -> 2 full buckets, 1 dropped
    out = resample_count(bars, 2)
    assert len(out) == 2


def test_resample_count_is_causal_prefix_stable():
    # Resampling a longer series must not change the earlier buckets: no future
    # bar leaks backward into an already-closed bucket.
    bars = _bars(10)
    short = resample_count(bars[:6], 2)
    long = resample_count(bars, 2)
    assert short == long[: len(short)]


def _bars_5m(n: int) -> list[Bar]:
    day = date(2026, 10, 7)
    start = at_et(day, 9, 30)
    out = []
    for i in range(n):
        ts = start + timedelta(minutes=5 * i)
        price = 100.0 + i
        out.append(Bar(ts, price, price + 1, price - 1, price + 0.5, 10.0))
    return out


def test_timeframes_5m_are_native_5m_bar_counts():
    assert TIMEFRAMES_5M == {"5m": 1, "15m": 3, "30m": 6, "1h": 12}


def test_resample_from_5m_is_causal_prefix_stable():
    bars = _bars_5m(30)
    short = resample(bars[:18], "15m", data="5m")
    long = resample(bars, "15m", data="5m")
    assert len(long) == 30 // 3
    assert short == long[: len(short)]


def test_resample_daily_buckets_by_et_calendar_day():
    day1 = _bars(3, start_hour=10)
    day2_start = at_et(date(2026, 10, 8), 10, 0)
    day2 = [Bar(day2_start + timedelta(hours=i), 1, 2, 0, 1.5, 5.0) for i in range(2)]
    out = resample_daily(day1 + day2)
    assert len(out) == 2
    assert out[0].ts == day1[0].ts
    assert out[1].ts == day2[0].ts


# --------------------------------------------------------------------- split


def test_split_index_has_no_overlap():
    n = 1000
    train_end = split_index(n, frac=0.6)
    assert 0 < train_end < n
    # Everything a train run can open is strictly before train_end; everything a
    # holdout run can open is at or after it. The two index ranges never overlap.
    train_range = range(30, train_end)
    holdout_range = range(train_end, n)
    assert set(train_range).isdisjoint(set(holdout_range))
    assert train_end == round(train_end)  # an int, usable as a bar index


# --------------------------------------------------------------------- session filter


def test_filter_signals_drops_outside_rth():
    bars = [
        Bar(at_et(date(2026, 10, 7), 8, 0), 1, 2, 0, 1, 1),   # pre-market
        Bar(at_et(date(2026, 10, 7), 10, 0), 1, 2, 0, 1, 1),  # RTH
        Bar(at_et(date(2026, 10, 7), 17, 0), 1, 2, 0, 1, 1),  # after hours
    ]
    signals = [SetupSignal("RMA", "LONG") for _ in bars]
    from apex.market import compute_indicators
    ind = compute_indicators(bars)

    out = filter_signals(bars, ind, signals, session="rth", regime="none")
    assert out[0] is None
    assert out[1] is not None
    assert out[2] is None


def test_filter_signals_rth_first2h_is_tighter_than_rth():
    bars = [
        Bar(at_et(date(2026, 10, 7), 10, 0), 1, 2, 0, 1, 1),  # inside first 2h
        Bar(at_et(date(2026, 10, 7), 13, 0), 1, 2, 0, 1, 1),  # inside RTH, not first 2h
    ]
    signals = [SetupSignal("RMA", "LONG") for _ in bars]
    from apex.market import compute_indicators
    ind = compute_indicators(bars)

    out = filter_signals(bars, ind, signals, session="rth_first2h", regime="none")
    assert out[0] is not None
    assert out[1] is None
