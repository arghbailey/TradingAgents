"""apex.strategies2: no-lookahead invariance, ORB range RTH-only, gap math across the
18:00 Globex reopen, eval-breach detection, and cost application. Offline, synthetic
bars only -- no network, no real sweep run."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest

from apex.strategies2 import (
    ET,
    EvalAccount,
    _apply_slippage,
    _favorable_target,
    _make_trade,
    build_gap_table,
    orb_range,
    with_indicators,
)

pytestmark = pytest.mark.unit


def _mk_df(rows: list[tuple[datetime, float, float, float, float, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(ET)
    df = df.sort_values("ts").reset_index(drop=True)
    df["et_date"] = df["ts"].dt.date
    df["et_time"] = df["ts"].dt.time
    return df


def _synthetic_day(day: date, n_bars: int = 90, start_hour: int = 9, start_min: int = 30,
                   base: float = 100.0) -> list[tuple[datetime, float, float, float, float, float]]:
    start = datetime(day.year, day.month, day.day, start_hour, start_min, tzinfo=ET)
    rows = []
    price = base
    for i in range(n_bars):
        ts = start + timedelta(minutes=5 * i)
        o = price
        c = price + (0.5 if i % 3 else -0.3)
        h = max(o, c) + 0.4
        lo = min(o, c) - 0.4
        rows.append((ts, o, h, lo, c, 1000.0 + 10 * i))
        price = c
    return rows


# --------------------------------------------------------------------- no-lookahead


def test_indicators_are_future_invariant():
    day = date(2026, 10, 7)
    rows = _synthetic_day(day, n_bars=120)
    full = with_indicators(_mk_df(rows))
    prefix = with_indicators(_mk_df(rows[:60]))
    for col in ("atr14", "adx14", "vwap", "vwap_dev", "vwap_dev_std"):
        got = full[col].iloc[:60].to_numpy()
        want = prefix[col].to_numpy()
        np.testing.assert_allclose(got, want, equal_nan=True, err_msg=col)


# --------------------------------------------------------------------- ORB range


def test_orb_range_ignores_premarket_bars():
    day = date(2026, 10, 7)
    premarket = [
        (datetime(2026, 10, 7, 8, 0, tzinfo=ET), 100, 99999.0, -99999.0, 100, 10.0),
        (datetime(2026, 10, 7, 9, 0, tzinfo=ET), 100, 50000.0, -50000.0, 100, 10.0),
    ]
    rth = _synthetic_day(day, n_bars=40, base=100.0)
    df = _mk_df(premarket + rth)
    df = with_indicators(df)
    is_rth = df[df["is_rth"]]
    idxs = is_rth.index.to_numpy()
    times = df["et_time"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    rng = orb_range(idxs, times, highs, lows, range_min=15)
    assert rng is not None
    range_high, range_low, start_pos = rng
    assert range_high < 1000  # premarket's absurd 99999/50000 extremes excluded
    assert range_low > -1000
    assert start_pos == 3  # first 15 minutes = 3 bars of 5m


# --------------------------------------------------------------------- gap math


def test_gap_table_uses_prior_rth_close_not_overnight_bar():
    day1 = date(2026, 10, 6)
    day2 = date(2026, 10, 7)
    day1_rth = _synthetic_day(day1, n_bars=10, base=100.0)
    day1_close_price = day1_rth[-1][4]
    # Overnight Globex bars straddling the 18:00 reopen with a deliberately extreme
    # price that must NOT be mistaken for "yesterday's close".
    overnight = [
        (datetime(2026, 10, 6, 18, 0, tzinfo=ET), 9999.0, 9999.0, 9999.0, 9999.0, 5.0),
        (datetime(2026, 10, 6, 22, 0, tzinfo=ET), 9999.0, 9999.0, 9999.0, 9999.0, 5.0),
        (datetime(2026, 10, 7, 2, 0, tzinfo=ET), 9999.0, 9999.0, 9999.0, 9999.0, 5.0),
    ]
    day2_rth = _synthetic_day(day2, n_bars=10, base=day1_close_price + 1.0)
    df = with_indicators(_mk_df(day1_rth + overnight + day2_rth))

    table = build_gap_table(df)
    prior_close, gap_pct = table[day2]
    assert prior_close == pytest.approx(day1_close_price)
    expected_gap = (day2_rth[0][1] - day1_close_price) / day1_close_price
    assert gap_pct == pytest.approx(expected_gap)
    # The first day in the table has no prior close.
    assert table[day1] == (None, None)


# --------------------------------------------------------------------- eval breach


def test_eval_account_flags_trailing_dd_breach():
    acct = EvalAccount()
    acct.realize(-500.0, date(2026, 10, 1))
    acct.realize(-600.0, date(2026, 10, 2))  # EOD checkpoint on day 2 entry: balance -500, no breach yet
    acct.realize(-1000.0, date(2026, 10, 3))  # EOD checkpoint on day 3: balance -1100 < HWM(50000)-2000
    acct.finalize()
    assert acct.breached
    assert acct.breach_type == "TRAILING_DD"


def test_eval_account_flags_daily_loss_breach():
    acct = EvalAccount()
    acct.realize(-1200.0, date(2026, 10, 1))  # single day, exceeds $1,000 daily loss stop
    assert acct.breached
    assert acct.breach_type == "DAILY_LOSS"


def test_eval_account_no_breach_on_safe_sequence():
    acct = EvalAccount()
    acct.realize(100.0, date(2026, 10, 1))
    acct.realize(-200.0, date(2026, 10, 2))
    acct.realize(150.0, date(2026, 10, 3))
    acct.finalize()
    assert not acct.breached
    assert acct.breach_type is None


# --------------------------------------------------------------------- cost application


def test_apply_slippage_worsens_price_against_position():
    tick = 0.25
    assert _apply_slippage(100.0, tick, "LONG", "entry") == pytest.approx(100.25)
    assert _apply_slippage(100.0, tick, "LONG", "exit") == pytest.approx(99.75)
    assert _apply_slippage(100.0, tick, "SHORT", "entry") == pytest.approx(99.75)
    assert _apply_slippage(100.0, tick, "SHORT", "exit") == pytest.approx(100.25)


def test_make_trade_applies_commission_and_slippage():
    # LONG: entry 100 -> +0.25 tick = 100.25; exit 110 -> -0.25 tick = 109.75.
    # Gross = (109.75 - 100.25) * point_value(5.0) = 47.5; minus $1.04 commission.
    trade = _make_trade(date(2026, 10, 7), "LONG", 0, 100.0, 1, 110.0, "TARGET", 0.25, 5.0)
    assert trade.entry_price == pytest.approx(100.25)
    assert trade.exit_price == pytest.approx(109.75)
    assert trade.pnl == pytest.approx(47.5 - 1.04)


def test_favorable_target_drops_target_behind_entry():
    # LONG with a target below entry (e.g. a degenerate "gap fill" target for a
    # follow-mode trade) must not fire as a bogus instant exit.
    assert _favorable_target(100.0, "LONG", 95.0) is None
    assert _favorable_target(100.0, "LONG", 105.0) == 105.0
    assert _favorable_target(100.0, "SHORT", 105.0) is None
    assert _favorable_target(100.0, "SHORT", 95.0) == 95.0
