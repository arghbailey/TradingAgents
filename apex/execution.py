"""Bracket construction and the DRY-RUN execution sink.

There is no broker integration. Nothing here talks to Tradovate, NinjaTrader or
Rithmic, and no webhook fires. ``DryRunDispatcher`` writes the bracket payload as JSON
to disk and to the log, and that is all it does.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from .config import DEFAULT_BRACKET_PARAMS, BracketParams, EvaluationType, get_contract

logger = logging.getLogger(__name__)


def _round_tick(price: float, tick: float) -> float:
    return round(round(price / tick) * tick, 6)


def build_bracket(symbol: str, direction: str, contracts: int, entry: float, stop_points: float,
                  eval_type: EvaluationType, params: BracketParams = DEFAULT_BRACKET_PARAMS) -> dict:
    """The full bracket for an approved order. Pure and deterministic.

    * Limit entry at the 21 EMA retest (``entry``).
    * Stop ``stop_points`` away.
    * EOD accounts: C1 takes the fixed target (2.5 pts MES / 12 pts MNQ). After C1
      fills, C2's stop moves to breakeven + 1 tick and trails the 21 EMA.
    * LEGACY accounts: scale 70% out at T1 = 1.5R and move the stop to breakeven at 1R.
    """
    spec = get_contract(symbol)
    tick = spec.tick_size
    sign = 1 if direction == "LONG" else -1
    entry = _round_tick(entry, tick)
    stop = _round_tick(entry - sign * stop_points, tick)
    be_plus = _round_tick(entry + sign * params.breakeven_offset_ticks * tick, tick)
    payload = {
        "symbol": spec.symbol,
        "side": "BUY" if direction == "LONG" else "SELL",
        "direction": direction,
        "contracts": contracts,
        "entry": {"type": "LIMIT", "price": entry, "rationale": "21 EMA retest"},
        "stop": {"type": "STOP", "price": stop, "points": stop_points},
        "risk_dollars": round(stop_points * spec.point_value * contracts, 2),
        "eval_type": EvaluationType.parse(eval_type).value,
    }
    if EvaluationType.parse(eval_type) is EvaluationType.LEGACY:
        scale = max(1, round(contracts * params.legacy_scale_fraction)) if contracts > 1 else contracts
        t1 = _round_tick(entry + sign * params.legacy_t1_r * stop_points, tick)
        be_trigger = _round_tick(entry + sign * params.legacy_breakeven_r * stop_points, tick)
        payload["targets"] = [
            {"leg": "T1", "contracts": scale, "price": t1, "r_multiple": params.legacy_t1_r},
            {"leg": "RUNNER", "contracts": contracts - scale, "price": None,
             "management": "trail 21 EMA"},
        ]
        payload["stop_management"] = {
            "move_to_breakeven_at": be_trigger, "breakeven_r": params.legacy_breakeven_r,
            "breakeven_price": entry,
        }
    else:
        c1 = (contracts + 1) // 2
        c2 = contracts - c1
        target = params.c1_target_points[spec.symbol]
        payload["targets"] = [
            {"leg": "C1", "contracts": c1, "price": _round_tick(entry + sign * target, tick),
             "points": target},
        ]
        if c2:
            payload["targets"].append(
                {"leg": "C2", "contracts": c2, "price": None,
                 "management": "after C1 fills: stop to breakeven+1 tick, then trail 21 EMA"})
            payload["stop_management"] = {"after_c1_fill_stop_to": be_plus, "then": "trail 21 EMA"}
        else:
            payload["stop_management"] = {"note": "single contract: C1 only, no runner"}
    return payload


class DryRunDispatcher:
    """Writes the order payload to ``out_dir`` as JSON. Never sends anything anywhere."""

    mode = "dry-run"

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.dispatched: list[dict] = []

    def dispatch(self, payload: dict) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        path = self.out_dir / f"dryrun_{payload.get('symbol', 'NA')}_{stamp}.json"
        record = {"mode": self.mode, "sent_to_broker": False, "payload": payload}
        path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        self.dispatched.append(record)
        logger.info("DRY-RUN order (not sent): %s", json.dumps(payload, default=str))
        return path
