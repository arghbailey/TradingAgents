"""Apex Risk Governor: every rule, with its boundary cases. Offline, no LLM."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from apex.calendar import EconomicEvent, at_et
from apex.config import CONTRACTS, EvaluationType, RiskParams, get_tier
from apex.risk import (
    AccountState,
    ApexRiskGovernor,
    MarketContext,
    OrderProposal,
    SessionPhase,
    Verdict,
    active_news_lockout,
    max_risk_per_trade,
    resize_to_micro,
    session_phase,
)

pytestmark = pytest.mark.unit

DAY = date(2026, 10, 7)
T50 = get_tier("50K")
GOV = ApexRiskGovernor()


def acct(eval_type=EvaluationType.EOD, tier=T50) -> AccountState:
    return AccountState.fresh(tier, eval_type)


def order(symbol="MNQ", contracts=1, stop=16.0, grade="A") -> OrderProposal:
    return OrderProposal(symbol, "LONG", contracts, 20_000.0, stop, grade)


def mkt(hh=10, mm=0, ss=0, spread=1, events=()) -> MarketContext:
    return MarketContext(at_et(DAY, hh, mm, ss), spread, tuple(events))


def reasons(d) -> str:
    return " | ".join(d.reasons)


# ---------------------------------------------------------------- contract math


def test_contract_point_values():
    assert CONTRACTS["MES"].point_value == 5.0
    assert CONTRACTS["MNQ"].point_value == 2.0
    assert CONTRACTS["ES"].point_value == 50.0
    assert CONTRACTS["NQ"].point_value == 20.0
    assert CONTRACTS["ES"].micro_symbol == "MES" and CONTRACTS["NQ"].micro_symbol == "MNQ"


# ----------------------------------------------------------- effective sizing


def test_max_risk_per_trade_is_buffer_times_clamped_fraction():
    assert max_risk_per_trade(2500) == pytest.approx(50.0)
    assert max_risk_per_trade(2500, RiskParams(risk_fraction=0.01)) == pytest.approx(25.0)
    assert max_risk_per_trade(2500, RiskParams(risk_fraction=0.001)) == pytest.approx(25.0)  # clamped up
    assert max_risk_per_trade(2500, RiskParams(risk_fraction=0.10)) == pytest.approx(50.0)   # clamped down


def test_baseline_approval_sizes_to_budget():
    d = GOV.evaluate(acct(), mkt(), order(contracts=2))  # $32/contract vs $50 budget
    assert d.verdict is Verdict.APPROVED
    assert d.contracts == 1
    assert d.max_risk_dollars == pytest.approx(50.0)


def test_tier_max_contracts_cap():
    d = GOV.evaluate(acct(), mkt(), order(contracts=500, stop=0.25))  # $0.50/contract
    assert d.verdict is Verdict.APPROVED
    assert d.contracts == T50.max_contracts_micro == 100


# --------------------------------------------------------------- daily loss


def test_daily_loss_exactly_limit_halts_flattens_and_locks():
    a = acct()
    a.mark_to_market(-650.0)
    assert a.session_locked
    d = GOV.evaluate(a, mkt(), order())
    assert d.verdict is Verdict.REJECT
    assert d.flatten_required and d.lock_session
    assert "daily loss" in reasons(d)


def test_daily_loss_one_cent_short_of_limit_does_not_lock():
    a = acct()
    a.mark_to_market(-649.99)
    assert not a.session_locked
    d = GOV.evaluate(a, mkt(), order())
    assert d.verdict is Verdict.REJECT  # but via the 15% circuit-breaker proximity rule
    assert "circuit breaker" in reasons(d) and not d.lock_session


def test_lock_persists_until_next_session():
    a = acct()
    a.realize(-650.0, stopped_out=True, at=at_et(DAY, 10, 0))
    a.realize(0.0, stopped_out=False)
    assert a.session_locked
    a.new_session()
    assert not a.session_locked
    assert GOV.evaluate(a, mkt(), order()).verdict is Verdict.APPROVED


def test_soft_warning_at_15_percent_halves_size():
    a = acct()
    a.mark_to_market(-97.5)  # exactly 15% of 650
    d = GOV.evaluate(a, mkt(), order())
    assert d.size_multiplier == 0.5
    assert d.max_risk_dollars == pytest.approx(0.02 * (2500 - 97.5) * 0.5, abs=0.01)
    b = acct()
    b.mark_to_market(-97.0)
    assert GOV.evaluate(b, mkt(), order()).size_multiplier == 1.0


def test_reject_within_15_percent_of_circuit_breaker():
    a = acct()
    a.mark_to_market(-552.5)  # remaining 97.5 == 15% of 650
    d = GOV.evaluate(a, mkt(), order())
    assert d.verdict is Verdict.REJECT and "circuit breaker" in reasons(d)
    b = acct()
    b.mark_to_market(-552.0)
    assert "circuit breaker" not in reasons(GOV.evaluate(b, mkt(), order()))


# ------------------------------------------------------------- consistency


def test_consistency_cap_exactly_900_blocks_new_positions():
    assert T50.max_single_day_profit == pytest.approx(900.0)
    a = acct()
    a.realize(900.0, stopped_out=False)
    d = GOV.evaluate(a, mkt(), order())
    assert d.verdict is Verdict.REJECT and "consistency cap" in reasons(d)
    b = acct()
    b.realize(899.99, stopped_out=False)
    assert GOV.evaluate(b, mkt(), order()).verdict is Verdict.APPROVED


# --------------------------------------------------------- trailing drawdown


def test_threshold_freezes_at_initial_plus_100():
    assert T50.threshold_freeze_level == 50_100
    a = acct(EvaluationType.EOD)
    a.realize(2_599.0, stopped_out=False)
    a.end_of_day()
    assert a.threshold == pytest.approx(50_099.0) and not a.frozen
    a.realize(1.0, stopped_out=False)
    a.end_of_day()
    assert a.threshold == pytest.approx(50_100.0) and a.frozen
    a.realize(5_000.0, stopped_out=False)
    a.end_of_day()
    assert a.threshold == pytest.approx(50_100.0)  # never trails again


def test_freeze_overshoot_caps_at_freeze_level():
    a = acct(EvaluationType.LEGACY)
    a.mark_to_market(4_000.0)
    assert a.threshold == pytest.approx(50_100.0) and a.frozen


def test_legacy_vs_eod_ratchet_plus_800_then_plus_200():
    legacy, eod = acct(EvaluationType.LEGACY), acct(EvaluationType.EOD)
    for a in (legacy, eod):
        a.mark_to_market(800.0)       # open trade runs to +$800
        a.realize(200.0, stopped_out=False)  # closes for +$200
    # LEGACY ratcheted on the unrealized peak; EOD has not moved intraday.
    assert legacy.threshold == pytest.approx(48_300.0)
    assert eod.threshold == pytest.approx(47_500.0)
    assert legacy.effective_buffer == pytest.approx(1_900.0)
    assert eod.effective_buffer == pytest.approx(2_700.0)
    for a in (legacy, eod):
        a.end_of_day()
    assert legacy.threshold == pytest.approx(48_300.0)
    assert eod.threshold == pytest.approx(47_700.0)  # ratchets on the realized close only
    assert eod.effective_buffer == pytest.approx(2_500.0)
    # Sizing follows the buffer: $38 vs $50.
    assert GOV.evaluate(legacy, mkt(), order()).max_risk_dollars == pytest.approx(38.0)


def test_breached_account_rejects_and_flattens():
    a = acct()
    a.threshold = 50_000.0
    d = GOV.evaluate(a, mkt(), order())
    assert d.verdict is Verdict.REJECT and d.flatten_required and "breached" in reasons(d)


# -------------------------------------------------------- consecutive losses


def test_two_stop_outs_trigger_60_minute_cooldown():
    a = acct()
    t0 = at_et(DAY, 10, 0)
    a.realize(-20.0, stopped_out=True, at=t0 - timedelta(minutes=10))
    a.realize(-20.0, stopped_out=True, at=t0)
    d = GOV.evaluate(a, MarketContext(t0 + timedelta(minutes=59, seconds=59), 1), order())
    assert d.verdict is Verdict.REJECT and "cooldown" in reasons(d)
    d = GOV.evaluate(a, MarketContext(t0 + timedelta(minutes=60), 1), order())
    assert d.verdict is Verdict.APPROVED


def test_three_stop_outs_shut_the_session_down():
    a = acct()
    for i in range(3):
        a.realize(-10.0, stopped_out=True, at=at_et(DAY, 9, 50 + i))
    d = GOV.evaluate(a, mkt(12, 0), order())
    assert d.verdict is Verdict.REJECT and "shutdown" in reasons(d) and d.lock_session


def test_a_win_resets_the_loss_streak():
    a = acct()
    a.realize(-10.0, stopped_out=True, at=at_et(DAY, 9, 50))
    a.realize(15.0, stopped_out=False)
    assert a.consecutive_losses == 0


# --------------------------------------------------------------- news lockout


@pytest.mark.parametrize("hh,mm,ss,locked", [
    (13, 54, 59, False), (13, 55, 0, True), (14, 0, 0, True), (14, 5, 0, True), (14, 5, 1, False),
])
def test_news_window_edges(hh, mm, ss, locked):
    fomc = EconomicEvent("FOMC", at_et(DAY, 14, 0))
    assert (active_news_lockout(at_et(DAY, hh, mm, ss), [fomc]) is not None) == locked
    d = GOV.evaluate(acct(), mkt(hh, mm, ss, events=[fomc]), order(grade="A+"))
    assert (d.verdict is Verdict.REJECT and d.flatten_required) == locked


def test_low_impact_unlisted_event_is_ignored():
    ev = EconomicEvent("Beige Book", at_et(DAY, 14, 0), impact="medium")
    assert active_news_lockout(at_et(DAY, 14, 0), [ev]) is None


@pytest.mark.parametrize("name", ["FOMC", "CPI", "PPI", "NFP", "GDP"])
def test_all_listed_events_lock(name):
    ev = EconomicEvent(name, at_et(DAY, 10, 0), impact="low")
    assert active_news_lockout(at_et(DAY, 10, 0), [ev]) is ev


# ------------------------------------------------------------- session clock


@pytest.mark.parametrize("hh,mm,phase", [
    (9, 29, SessionPhase.CLOSED), (9, 30, SessionPhase.OPENING_RANGE), (9, 44, SessionPhase.OPENING_RANGE),
    (9, 45, SessionPhase.PRIME), (11, 29, SessionPhase.PRIME), (11, 30, SessionPhase.MIDDAY),
    (13, 30, SessionPhase.AFTERNOON), (15, 44, SessionPhase.AFTERNOON), (15, 45, SessionPhase.LATE),
    (15, 54, SessionPhase.LATE), (15, 55, SessionPhase.FLATTEN), (16, 0, SessionPhase.CLOSED),
])
def test_session_phase_boundaries(hh, mm, phase):
    assert session_phase(at_et(DAY, hh, mm)) is phase


def test_0944_rejected_0945_approved():
    assert GOV.evaluate(acct(), mkt(9, 44), order()).verdict is Verdict.REJECT
    assert GOV.evaluate(acct(), mkt(9, 45), order()).verdict is Verdict.APPROVED


def test_midday_half_size():
    d = GOV.evaluate(acct(), mkt(11, 30), order())
    assert d.size_multiplier == 0.5 and d.max_risk_dollars == pytest.approx(25.0)
    assert d.verdict is Verdict.REJECT  # $32 MNQ stop does not fit $25
    assert GOV.evaluate(acct(), mkt(11, 30), order(stop=10.0)).verdict is Verdict.APPROVED


def test_afternoon_a_plus_only():
    assert GOV.evaluate(acct(), mkt(13, 30), order(grade="A")).verdict is Verdict.REJECT
    assert GOV.evaluate(acct(), mkt(13, 30), order(grade="A+")).verdict is Verdict.APPROVED


def test_1555_mandatory_flatten_and_no_entries():
    d = GOV.evaluate(acct(), mkt(15, 55), order(grade="A+"))
    assert d.verdict is Verdict.REJECT and d.flatten_required
    d = GOV.evaluate(acct(), mkt(15, 50), order(grade="A+"))
    assert d.verdict is Verdict.REJECT and not d.flatten_required


# ------------------------------------------------------------------- spread


def test_spread_gate_two_ticks_ok_three_rejected():
    assert GOV.evaluate(acct(), mkt(spread=2), order()).verdict is Verdict.APPROVED
    d = GOV.evaluate(acct(), mkt(spread=3), order())
    assert d.verdict is Verdict.REJECT and "spread" in reasons(d)


# ------------------------------------------------------------------- resizer


def test_mini_that_does_not_fit_resizes_to_micro_and_rechecks():
    es = OrderProposal("ES", "LONG", 2, 6_000.0, 4.0, "A")  # $200/contract vs $50 budget
    d = GOV.evaluate(acct(), mkt(), es)
    assert d.verdict is Verdict.RESIZE_MICRO and d.suggested_micro_contracts == 2
    micro = resize_to_micro(es)
    assert (micro.symbol, micro.contracts) == ("MES", 20)
    d2 = GOV.evaluate(acct(), mkt(), micro)
    assert d2.verdict is Verdict.APPROVED and d2.contracts == 2 and d2.symbol == "MES"


def test_micro_that_does_not_fit_is_rejected():
    d = GOV.evaluate(acct(), mkt(), order(stop=40.0))  # $80/contract MNQ
    assert d.verdict is Verdict.REJECT


def test_resize_of_micro_is_identity():
    o = order()
    assert resize_to_micro(o) is o
