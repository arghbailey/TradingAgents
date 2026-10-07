"""Apex account tiers, contract specifications and risk parameters.

UNVERIFIED — check apextraderfunding.com before relying on any tier number below.

The tier table reproduces the Apex design document. Apex's own site refused an
automated fetch (HTTP 403) when this was written (2026-10), and third-party summaries
of Apex's March 2026 rule overhaul disagree with the design document. For example, they
list the 50K evaluation at a $2,000 drawdown, 6 contracts and a $1,000 daily loss limit,
and they list no 250K/300K evaluations. Treat every number here as a placeholder you
must confirm. Override a tier with ``ApexAccountTier(...)`` or ``register_tier``. The
rest of the code reads tiers only through ``get_tier``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class EvaluationType(StrEnum):
    """How the trailing drawdown threshold ratchets."""

    LEGACY = "LEGACY"  # intraday trailing: ratchets on the unrealized high-water mark
    EOD = "EOD"        # end-of-day trailing: ratchets only on the realized close balance

    @classmethod
    def parse(cls, value: str | EvaluationType) -> EvaluationType:
        if isinstance(value, EvaluationType):
            return value
        try:
            return cls(str(value).strip().upper())
        except ValueError:
            raise ValueError(f"eval type must be one of {[e.value for e in cls]}, got {value!r}") from None


@dataclass(frozen=True)
class ApexAccountTier:
    name: str
    nominal_size: float
    profit_target: float
    total_drawdown: float
    max_contracts_mini: int
    max_contracts_micro: int
    daily_loss_limit: float
    consistency_cap_ratio: float = 0.30

    @property
    def max_single_day_profit(self) -> float:
        """Consistency cap: the most one day may contribute toward the profit target."""
        return self.profit_target * self.consistency_cap_ratio

    @property
    def threshold_freeze_level(self) -> float:
        """The trailing threshold stops trailing once it reaches this balance."""
        return self.nominal_size + 100


# UNVERIFIED — check apextraderfunding.com. Values are from the design document.
_TIERS: dict[str, ApexAccountTier] = {
    t.name: t
    for t in (
        ApexAccountTier("25K", 25_000, 1_500, 1_500, 4, 40, 400),
        ApexAccountTier("50K", 50_000, 3_000, 2_500, 10, 100, 650),
        ApexAccountTier("100K", 100_000, 6_000, 3_000, 14, 140, 750),
        ApexAccountTier("150K", 150_000, 9_000, 5_000, 17, 170, 1_250),
        ApexAccountTier("250K", 250_000, 15_000, 6_500, 27, 270, 1_600),
        ApexAccountTier("300K", 300_000, 20_000, 7_500, 35, 350, 1_900),
    )
}


def get_tier(name: str) -> ApexAccountTier:
    key = str(name).strip().upper()
    if key not in _TIERS:
        raise ValueError(f"unknown Apex tier {name!r}; known: {', '.join(_TIERS)}")
    return _TIERS[key]


def register_tier(tier: ApexAccountTier) -> None:
    """Add or replace a tier, e.g. once you have confirmed Apex's current numbers."""
    _TIERS[tier.name.upper()] = tier


def tier_names() -> list[str]:
    return list(_TIERS)


@dataclass(frozen=True)
class ContractSpec:
    symbol: str
    point_value: float      # USD per index point per contract
    tick_size: float        # index points
    is_micro: bool
    micro_symbol: str       # the micro equivalent (itself for a micro)
    family: str             # "ES" or "NQ": which index the contract tracks

    @property
    def tick_value(self) -> float:
        return self.point_value * self.tick_size


CONTRACTS: dict[str, ContractSpec] = {
    "ES": ContractSpec("ES", 50.0, 0.25, False, "MES", "ES"),
    "MES": ContractSpec("MES", 5.0, 0.25, True, "MES", "ES"),
    "NQ": ContractSpec("NQ", 20.0, 0.25, False, "MNQ", "NQ"),
    "MNQ": ContractSpec("MNQ", 2.0, 0.25, True, "MNQ", "NQ"),
}

MICRO_PER_MINI = 10


def get_contract(symbol: str) -> ContractSpec:
    key = str(symbol).strip().upper()
    if key not in CONTRACTS:
        raise ValueError(f"unsupported symbol {symbol!r}; supported: {', '.join(CONTRACTS)}")
    return CONTRACTS[key]


@dataclass(frozen=True)
class RiskParams:
    """Deterministic risk knobs. Every LLM-independent rule reads these."""

    risk_fraction: float = 0.02            # clamped to [min_risk_fraction, max_risk_fraction]
    min_risk_fraction: float = 0.01
    max_risk_fraction: float = 0.02
    soft_warning_fraction: float = 0.15    # loss >= 15% of daily budget -> halve size
    circuit_proximity_fraction: float = 0.15  # remaining daily budget <= 15% -> reject
    max_spread_ticks: int = 2
    cooldown_after_losses: int = 2
    cooldown_minutes: int = 60
    shutdown_after_losses: int = 3
    news_window_minutes: int = 5
    high_impact_events: tuple[str, ...] = ("FOMC", "CPI", "PPI", "NFP", "GDP")
    midday_size_multiplier: float = 0.5
    soft_warning_size_multiplier: float = 0.5
    min_adx: float = 20.0
    max_debate_rounds: int = 2
    max_resize_iterations: int = 2


@dataclass(frozen=True)
class BracketParams:
    # Defaults from the design doc; mini contracts share their micro's point distances.
    stop_points: dict[str, float] = field(
        default_factory=lambda: {"MES": 4.0, "ES": 4.0, "MNQ": 16.0, "NQ": 16.0})
    c1_target_points: dict[str, float] = field(
        default_factory=lambda: {"MES": 2.5, "ES": 2.5, "MNQ": 12.0, "NQ": 12.0})
    # When set, stop = max(default stop, atr_stop_multiple * ATR).
    atr_stop_multiple: float | None = None
    legacy_scale_fraction: float = 0.70
    legacy_t1_r: float = 1.5
    legacy_breakeven_r: float = 1.0
    breakeven_offset_ticks: int = 1


@dataclass(frozen=True)
class SetupParams:
    # Double-bottom/top minimum pattern height, by contract family.
    dbdt_min_range_points: dict[str, float] = field(default_factory=lambda: {"ES": 10.0, "NQ": 30.0})
    max_bars_past_crossover: int = 3
    rsi_overbought: float = 80.0
    rsi_oversold: float = 20.0
    momo_min_rvol: float = 2.0
    momo_min_range_atr: float = 1.5
    ribbon_compression_atr: float = 0.15
    priority: tuple[str, ...] = ("DB_DT", "RMA", "TREND", "MOMO", "FFMA")


# Frozen, shared defaults (safe to use as argument defaults).
DEFAULT_RISK_PARAMS = RiskParams()
DEFAULT_BRACKET_PARAMS = BracketParams()
DEFAULT_SETUP_PARAMS = SetupParams()
