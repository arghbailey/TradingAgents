"""The Apex Risk Governor: deterministic Python that enforces every risk rule.

LLMs may propose a trade, but only this module decides whether it goes out. Nothing
in here reads model output. The verdict is computed from account state, the clock,
the calendar, the quote and the order geometry alone.

Rules implemented (tier numbers come from ``apex.config``; 50K values in brackets):

* Effective-equity sizing: max risk per trade = effective drawdown buffer x risk
  fraction, with the fraction clamped to [1%, 2%] ($25-$50 on a $2,500 buffer).
* Trailing drawdown: LEGACY ratchets on the intraday (unrealized) high-water mark;
  EOD ratchets only on the end-of-day realized balance. The threshold stops trailing
  for good once it reaches ``nominal + $100`` (the freeze level).
* Daily loss halt at the tier's daily_loss_limit (-$650): flatten and lock until the
  next session. Soft warning at 15% of the daily budget halves size. Within 15% of the
  circuit breaker, new entries are rejected.
* Consistency cap: no new positions once the day's profit >= max_single_day_profit ($900).
* Consecutive losses: 2 stop-outs -> 60-minute cooldown; 3 -> session shutdown.
* News lockout: no new orders, and flatten, from T-5 to T+5 minutes (inclusive) around
  FOMC/CPI/PPI/NFP/GDP.
* Session clock (America/New_York): no entries 09:30-09:45; prime 09:45-11:30;
  midday 11:30-13:30 at half size; 13:30-15:45 A+ setups only; 15:45-15:55 no new
  entries; mandatory flatten at 15:55 and nothing new after it.
* Spread gate: reject when the spread is wider than 2 ticks.
* Max contracts per tier. A mini order that cannot fit returns RESIZE_MICRO; the
  resizer converts it 10:1 to micros and the governor checks it again.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timedelta
from enum import StrEnum

from .calendar import ET, EconomicEvent
from .config import (
    DEFAULT_RISK_PARAMS,
    MICRO_PER_MINI,
    ApexAccountTier,
    EvaluationType,
    RiskParams,
    get_contract,
)


class Verdict(StrEnum):
    APPROVED = "APPROVED"
    RESIZE_MICRO = "RESIZE_MICRO"
    REJECT = "REJECT"


class SessionPhase(StrEnum):
    CLOSED = "CLOSED"              # before 09:30 or at/after 16:00
    OPENING_RANGE = "OPENING_RANGE"  # 09:30 <= t < 09:45: no entries
    PRIME = "PRIME"                # 09:45 <= t < 11:30: full size
    MIDDAY = "MIDDAY"              # 11:30 <= t < 13:30: half size
    AFTERNOON = "AFTERNOON"        # 13:30 <= t < 15:45: A+ setups only
    LATE = "LATE"                  # 15:45 <= t < 15:55: no new entries
    FLATTEN = "FLATTEN"            # 15:55 <= t < 16:00: mandatory flatten


def session_phase(now: datetime) -> SessionPhase:
    t = now.astimezone(ET).time() if now.tzinfo else now.time()
    if t < time(9, 30) or t >= time(16, 0):
        return SessionPhase.CLOSED
    if t < time(9, 45):
        return SessionPhase.OPENING_RANGE
    if t < time(11, 30):
        return SessionPhase.PRIME
    if t < time(13, 30):
        return SessionPhase.MIDDAY
    if t < time(15, 45):
        return SessionPhase.AFTERNOON
    if t < time(15, 55):
        return SessionPhase.LATE
    return SessionPhase.FLATTEN


def active_news_lockout(now: datetime, events: list[EconomicEvent],
                        params: RiskParams = DEFAULT_RISK_PARAMS) -> EconomicEvent | None:
    """The high-impact event whose [T-5, T+5] window (inclusive) contains ``now``."""
    window = timedelta(minutes=params.news_window_minutes)
    watched = {n.upper() for n in params.high_impact_events}
    for event in events:
        # A named release (FOMC, CPI, ...) always counts; anything else only if marked high.
        if event.name.upper() not in watched and event.impact.lower() != "high":
            continue
        if event.when - window <= now <= event.when + window:
            return event
    return None


# --------------------------------------------------------------------------- account


@dataclass
class AccountState:
    """Mutable account state. The governor reads it; the trade loop updates it."""

    tier: ApexAccountTier
    eval_type: EvaluationType
    balance: float                 # realized balance
    unrealized: float = 0.0
    high_water_mark: float = 0.0   # the balance the trailing threshold follows
    threshold: float = 0.0         # liquidation threshold (drawdown floor)
    frozen: bool = False           # the threshold has stopped trailing for good
    day_start_balance: float = 0.0
    consecutive_losses: int = 0
    last_stop_out: datetime | None = None
    session_locked: bool = False   # daily halt or 3-loss shutdown, until next session
    lock_reason: str = ""
    open_contracts: int = 0

    @classmethod
    def fresh(cls, tier: ApexAccountTier, eval_type: EvaluationType) -> AccountState:
        start = tier.nominal_size
        return cls(tier=tier, eval_type=EvaluationType.parse(eval_type), balance=start,
                   high_water_mark=start, threshold=start - tier.total_drawdown,
                   day_start_balance=start)

    # --- derived figures
    @property
    def equity(self) -> float:
        return self.balance + self.unrealized

    @property
    def daily_pnl(self) -> float:
        return self.equity - self.day_start_balance

    @property
    def effective_buffer(self) -> float:
        return max(0.0, self.equity - self.threshold)

    @property
    def failed(self) -> bool:
        return self.equity <= self.threshold

    # --- threshold ratchet
    def _ratchet(self, mark: float) -> None:
        if self.frozen:
            return
        if mark > self.high_water_mark:
            self.high_water_mark = mark
        freeze = self.tier.threshold_freeze_level
        candidate = self.high_water_mark - self.tier.total_drawdown
        if candidate >= freeze:
            self.threshold = freeze
            self.frozen = True
        elif candidate > self.threshold:
            self.threshold = candidate

    def mark_to_market(self, unrealized: float) -> None:
        """An intraday price update. LEGACY ratchets on it; EOD does not."""
        self.unrealized = unrealized
        if self.eval_type is EvaluationType.LEGACY:
            self._ratchet(self.equity)
        self._check_daily_halt()

    def realize(self, pnl: float, stopped_out: bool, at: datetime | None = None) -> None:
        """Close a position for ``pnl`` (unrealized goes to zero)."""
        self.unrealized = 0.0
        self.balance += pnl
        if self.eval_type is EvaluationType.LEGACY:
            self._ratchet(self.equity)
        if stopped_out and pnl < 0:
            self.consecutive_losses += 1
            self.last_stop_out = at
        elif pnl > 0:
            self.consecutive_losses = 0
        self._check_daily_halt()

    def end_of_day(self) -> None:
        """Session close: EOD accounts ratchet on the realized balance here."""
        if self.eval_type is EvaluationType.EOD:
            self._ratchet(self.balance)
        else:
            self._ratchet(self.equity)

    def new_session(self) -> None:
        self.day_start_balance = self.balance
        self.consecutive_losses = 0
        self.last_stop_out = None
        self.session_locked = False
        self.lock_reason = ""

    def _check_daily_halt(self) -> None:
        if self.daily_pnl <= -self.tier.daily_loss_limit and not self.session_locked:
            self.session_locked = True
            self.lock_reason = (f"daily loss limit hit ({self.daily_pnl:.2f} <= "
                                f"-{self.tier.daily_loss_limit:.2f}): flatten, locked until next session")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tier"] = self.tier.name
        d["eval_type"] = self.eval_type.value
        d["last_stop_out"] = self.last_stop_out.isoformat() if self.last_stop_out else None
        d.update(equity=self.equity, daily_pnl=self.daily_pnl, effective_buffer=self.effective_buffer)
        return d


# --------------------------------------------------------------------------- orders


@dataclass(frozen=True)
class OrderProposal:
    symbol: str
    direction: str            # "LONG" or "SHORT"
    contracts: int
    entry: float
    stop_points: float        # distance from entry to stop, in index points
    setup_grade: str = "A"    # "A+", "A", "B"


@dataclass(frozen=True)
class MarketContext:
    now: datetime
    spread_ticks: float
    events: tuple[EconomicEvent, ...] = ()


@dataclass
class RiskDecision:
    verdict: Verdict
    contracts: int = 0
    symbol: str = ""
    reasons: list[str] = field(default_factory=list)
    flatten_required: bool = False
    lock_session: bool = False
    max_risk_dollars: float = 0.0
    risk_per_contract: float = 0.0
    size_multiplier: float = 1.0
    suggested_micro_contracts: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict.value
        return d


def max_risk_per_trade(buffer: float, params: RiskParams = DEFAULT_RISK_PARAMS) -> float:
    fraction = min(max(params.risk_fraction, params.min_risk_fraction), params.max_risk_fraction)
    return max(0.0, buffer) * fraction


class ApexRiskGovernor:
    """Pure function of (account, market, order) -> RiskDecision."""

    def __init__(self, params: RiskParams = DEFAULT_RISK_PARAMS):
        self.params = params

    def evaluate(self, account: AccountState, market: MarketContext,
                 order: OrderProposal) -> RiskDecision:
        p, tier = self.params, account.tier
        reject = lambda *why, **kw: RiskDecision(  # noqa: E731
            Verdict.REJECT, 0, order.symbol, list(why), **kw)

        try:
            spec = get_contract(order.symbol)
        except ValueError as exc:
            return reject(str(exc))
        if order.direction not in ("LONG", "SHORT"):
            return reject(f"direction must be LONG or SHORT, got {order.direction!r}")
        if order.contracts < 1 or order.stop_points <= 0:
            return reject("order needs at least 1 contract and a positive stop distance")

        # Hard account states: flatten-and-lock conditions first.
        if account.failed:
            return reject(f"equity {account.equity:.2f} at/below trailing threshold "
                          f"{account.threshold:.2f}: account breached",
                          flatten_required=True, lock_session=True)
        if account.daily_pnl <= -tier.daily_loss_limit:
            return reject(f"daily loss {account.daily_pnl:.2f} reached limit -{tier.daily_loss_limit:.2f}: "
                          "flatten and lock until next session", flatten_required=True, lock_session=True)
        if account.session_locked:
            return reject(f"session locked: {account.lock_reason or 'until next session'}")
        if account.consecutive_losses >= p.shutdown_after_losses:
            return reject(f"{account.consecutive_losses} consecutive stop-outs: session shutdown",
                          flatten_required=True, lock_session=True)
        if account.consecutive_losses >= p.cooldown_after_losses and account.last_stop_out is not None:
            resume = account.last_stop_out + timedelta(minutes=p.cooldown_minutes)
            if market.now < resume:
                return reject(f"{account.consecutive_losses} consecutive stop-outs: cooldown until "
                              f"{resume.astimezone(ET).strftime('%H:%M')} ET")

        event = active_news_lockout(market.now, list(market.events), p)
        if event is not None:
            return reject(f"news lockout: {event.name} at {event.when.astimezone(ET).strftime('%H:%M')} ET "
                          f"(+/-{p.news_window_minutes} min): no new orders, flatten",
                          flatten_required=True)

        phase = session_phase(market.now)
        if phase is SessionPhase.FLATTEN:
            return reject("15:55 ET mandatory flatten: no new entries", flatten_required=True)
        if phase is SessionPhase.CLOSED:
            return reject("outside the 09:30-16:00 ET session: no entries")
        if phase is SessionPhase.OPENING_RANGE:
            return reject("09:30-09:45 ET opening range: no entries")
        if phase is SessionPhase.LATE:
            return reject("after 15:45 ET: no new entries before the 15:55 flatten")
        if phase is SessionPhase.AFTERNOON and order.setup_grade != "A+":
            return reject(f"13:30-15:45 ET accepts A+ setups only (got {order.setup_grade})")

        day_profit = account.daily_pnl
        if day_profit >= tier.max_single_day_profit:
            return reject(f"consistency cap: day profit {day_profit:.2f} >= "
                          f"{tier.max_single_day_profit:.2f} ({tier.consistency_cap_ratio:.0%} of target)")

        loss_used = max(0.0, -day_profit)
        remaining = tier.daily_loss_limit - loss_used
        if remaining <= p.circuit_proximity_fraction * tier.daily_loss_limit:
            return reject(f"within {p.circuit_proximity_fraction:.0%} of the daily circuit breaker "
                          f"(remaining {remaining:.2f} of {tier.daily_loss_limit:.2f})")

        if market.spread_ticks > p.max_spread_ticks:
            return reject(f"spread {market.spread_ticks:g} ticks > {p.max_spread_ticks} ticks")

        # Sizing.
        reasons: list[str] = []
        multiplier = 1.0
        if phase is SessionPhase.MIDDAY:
            multiplier *= p.midday_size_multiplier
            reasons.append("11:30-13:30 ET midday: half size")
        if loss_used >= p.soft_warning_fraction * tier.daily_loss_limit:
            multiplier *= p.soft_warning_size_multiplier
            reasons.append(f"soft warning: day loss {loss_used:.2f} >= "
                           f"{p.soft_warning_fraction:.0%} of daily budget: half size")

        budget = max_risk_per_trade(account.effective_buffer, p) * multiplier
        # A stop-out may not breach the daily limit or the circuit-breaker margin.
        budget = min(budget, remaining - p.circuit_proximity_fraction * tier.daily_loss_limit)
        risk_per_contract = order.stop_points * spec.point_value
        tier_cap = tier.max_contracts_micro if spec.is_micro else tier.max_contracts_mini
        cap = max(0, tier_cap - account.open_contracts)
        fit = math.floor(budget / risk_per_contract + 1e-9) if risk_per_contract > 0 else 0
        allowed = min(order.contracts, fit, cap)
        base = {"max_risk_dollars": round(budget, 2), "risk_per_contract": risk_per_contract,
                "size_multiplier": multiplier}

        if allowed >= 1:
            if allowed < order.contracts:
                reasons.append(f"sized down {order.contracts} -> {allowed} {spec.symbol} "
                               f"(risk budget ${budget:.2f}, ${risk_per_contract:.2f}/contract, tier cap {cap})")
            reasons.append(f"approved {allowed} {spec.symbol}: risk ${allowed * risk_per_contract:.2f} "
                           f"<= budget ${budget:.2f}")
            return RiskDecision(Verdict.APPROVED, allowed, spec.symbol, reasons, **base)

        if not spec.is_micro:
            micro = get_contract(spec.micro_symbol)
            micro_risk = order.stop_points * micro.point_value
            micro_cap = max(0, tier.max_contracts_micro - account.open_contracts)
            micro_fit = min(order.contracts * MICRO_PER_MINI,
                            math.floor(budget / micro_risk + 1e-9) if micro_risk > 0 else 0, micro_cap)
            if micro_fit >= 1:
                reasons.append(f"{spec.symbol} risk ${risk_per_contract:.2f}/contract exceeds budget "
                               f"${budget:.2f}: resize to {micro_fit} {micro.symbol}")
                return RiskDecision(Verdict.RESIZE_MICRO, 0, spec.symbol, reasons,
                                    suggested_micro_contracts=micro_fit, **base)

        reasons.append(f"no size fits: ${risk_per_contract:.2f}/contract vs budget ${budget:.2f}, "
                       f"tier cap {cap}")
        return RiskDecision(Verdict.REJECT, 0, spec.symbol, reasons, **base)


def resize_to_micro(order: OrderProposal) -> OrderProposal:
    """Convert a mini order to its micro equivalent at 10:1. Micros are returned unchanged."""
    spec = get_contract(order.symbol)
    if spec.is_micro:
        return order
    return OrderProposal(spec.micro_symbol, order.direction, order.contracts * MICRO_PER_MINI,
                         order.entry, order.stop_points, order.setup_grade)
