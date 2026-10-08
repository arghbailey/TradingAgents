"""Apex backtest harness: no-lookahead, costs, trailing-DD breach, a deterministic
synthetic run. Offline, no network (no yfinance calls; bars are built by hand or via
``apex.market.fixture_bars``)."""

from __future__ import annotations

from datetime import timedelta

import pytest

from apex.backtest import (
    TICK_VALUE,
    BacktestResult,
    backtest_symbol_setup,
    compute_signals,
    run_walk_forward,
)
from apex.market import Bar, fixture_bars
from apex.risk import ApexRiskGovernor, RiskDecision, Verdict
from apex.setups import detect_setup

pytestmark = pytest.mark.unit


def _flat_bars(symbol: str, n: int = 60, price: float = 6000.0) -> list[Bar]:
    base = fixture_bars(symbol, __import__("datetime").date(2026, 10, 7), n_bars=n)
    # Flatten the generated series so detect_setup never fires: used only to drive
    # the walk loop forward and exercise the no-lookahead check.
    return [Bar(b.ts, price, price + 0.1, price - 0.1, price, 1000.0) for b in base]


def test_tick_values_match_config():
    assert TICK_VALUE["MNQ"] == pytest.approx(0.50)
    assert TICK_VALUE["MES"] == pytest.approx(1.25)


# --------------------------------------------------------------------- no-lookahead


def test_fast_signal_matches_detect_setup_on_closed_slices():
    # compute_signals() reads indicators off the whole series at once (they're all
    # causal recurrences), instead of recomputing detect_setup(bars[: i + 1]) from
    # scratch every bar. Check the two agree at every index.
    bars = fixture_bars("MES", __import__("datetime").date(2026, 10, 7), n_bars=260)
    fast = compute_signals("MES", bars)
    for i in range(30, len(bars)):
        slow = detect_setup(bars[: i + 1], "MES")
        assert (fast[i].setup, fast[i].direction) == (slow.setup, slow.direction), i


def test_future_bars_do_not_change_past_signals():
    bars_short = _flat_bars("MES", n=60)
    extra = fixture_bars("MES", __import__("datetime").date(2026, 10, 8), n_bars=40)
    bars_long = bars_short + extra  # a different, signal-triggering future

    sig_short = compute_signals("MES", bars_short)
    sig_long = compute_signals("MES", bars_long)
    for i in range(30, len(bars_short)):
        assert (sig_short[i].setup, sig_short[i].direction) == (sig_long[i].setup, sig_long[i].direction)


# --------------------------------------------------------------------- costs


def test_deterministic_rma_long_applies_slippage_and_commission():
    day = __import__("datetime").date(2026, 10, 7)
    base = fixture_bars("MES", day, n_bars=260)  # ends in an RMA long on the last bar
    last_close = base[-1].close
    # Fill bar: opens at the last close, and its own range hits the C1 target (2.5 pts).
    fill_ts = base[-1].ts + timedelta(minutes=5)
    fill_bar = Bar(fill_ts, last_close, last_close + 3.0, last_close - 0.5, last_close + 1.0, 1000.0)
    bars = base + [fill_bar]

    trades, account, blown = run_walk_forward("MES", "RMA", bars, "50K", "EOD")

    assert blown is None
    assert len(trades) == 1
    t = trades[0]
    assert t.direction == "LONG" and t.exit_reason == "TARGET"
    # Entry/exit are each pushed one tick (0.25) against the position.
    assert t.entry_price == pytest.approx(last_close + 0.25)
    assert t.exit_price == pytest.approx(last_close + 2.5 - 0.25)
    gross_per_contract = (t.exit_price - t.entry_price) * 5.0  # MES point value
    expected_pnl = round(gross_per_contract * t.contracts - 1.04 * t.contracts, 2)
    assert t.pnl == pytest.approx(expected_pnl)
    assert t.pnl > 0


# --------------------------------------------------------------------- trailing DD


def test_trailing_dd_breach_is_recorded(monkeypatch):
    # Force one oversized APPROVED fill (bypassing normal sizing) to drive a loss
    # bigger than the 50K tier's $2,000 trailing drawdown in a single trade, and
    # check the harness records the breach and the date it happened.
    day = __import__("datetime").date(2026, 10, 7)
    base = fixture_bars("MES", day, n_bars=260)
    last_close = base[-1].close
    fill_ts = base[-1].ts + timedelta(minutes=5)
    # This bar's low hits the stop (4 pts away); no target hit first.
    fill_bar = Bar(fill_ts, last_close, last_close + 0.1, last_close - 5.0, last_close - 4.0, 1000.0)
    bars = base + [fill_bar]

    big_loss_decision = RiskDecision(Verdict.APPROVED, contracts=150, symbol="MES")
    monkeypatch.setattr(ApexRiskGovernor, "evaluate", lambda self, account, market, order: big_loss_decision)

    trades, account, blown = run_walk_forward("MES", "RMA", bars, "50K", "EOD")

    assert len(trades) == 1
    assert trades[0].exit_reason == "STOP"
    assert trades[0].pnl < -2_000  # breached the $2,000 trailing DD in one trade
    assert account.failed is True
    assert blown == fill_bar.ts.date().isoformat()


# --------------------------------------------------------------------- end to end


def test_backtest_symbol_setup_returns_a_result_with_consistent_fields():
    bars = _flat_bars("MNQ", n=60)
    result = backtest_symbol_setup("MNQ", "RMA", bars, "50K", "EOD")
    assert isinstance(result, BacktestResult)
    assert result.trades == 0  # flat bars never trigger a setup
    assert result.net_pnl == 0.0
    assert result.beats_bh == (result.net_pnl > result.buy_hold_pnl)
