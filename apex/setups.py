"""Deterministic WSGTA setup detection. No LLM is involved here.

Five setups, evaluated on the last closed bar:

* RMA   - pullback in trend to the 21/30 EMA zone. Long when ema21 > ema65 and price
          holds above VWAP, the bar tags the 21 EMA and closes above the 30 EMA.
          Short mirrors it.
* FFMA  - exhaustion fade: RSI > 80 -> short, RSI < 20 -> long.
* TREND - compressed 9/15 ribbon (|ema9 - ema15| <= k * ATR) that has just crossed.
          The entry must come no more than 3 bars past the crossover.
* MOMO  - a momentum bar: RVOL >= 2, range >= 1.5 ATR, ADX >= 25, closing in its
          direction, and no more than 3 bars past the 9/15 crossover.
* DB_DT - double bottom / double top: two swing extremes within tolerance and a
          pattern height of at least 10 pts (ES family) / 30 pts (NQ family).

Setups are checked in ``SetupParams.priority`` order and the first match wins.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .config import DEFAULT_SETUP_PARAMS, SetupParams, get_contract
from .market import Bar, Indicators, compute_indicators

SETUPS = ("RMA", "FFMA", "TREND", "MOMO", "DB_DT")


@dataclass
class SetupSignal:
    setup: str | None                 # one of SETUPS, or None
    direction: str | None = None      # "LONG" / "SHORT"
    grade: str = "B"                  # "A+", "A", "B"
    confluence: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    bars_since_cross: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _bars_since_cross(fast: list[float], slow: list[float]) -> tuple[int | None, str | None]:
    """Bars since the last fast/slow crossover and which way it went."""
    for back in range(1, len(fast)):
        i = len(fast) - back
        prev = fast[i - 1] - slow[i - 1]
        curr = fast[i] - slow[i]
        if prev <= 0 < curr:
            return back - 1, "LONG"
        if prev >= 0 > curr:
            return back - 1, "SHORT"
    return None, None


def confluence_snapshot(ind: Indicators) -> dict:
    i = -1
    return {
        "close": round(ind.close[i], 2),
        "vwap": round(ind.vwap[i], 2),
        "ema21": round(ind.ema21[i], 2),
        "ema30": round(ind.ema30[i], 2),
        "ema65": round(ind.ema65[i], 2),
        "ema200": round(ind.ema200[i], 2),
        "adx": round(ind.adx[i], 2),
        "rvol": round(ind.rvol[i], 2),
        "rsi": round(ind.rsi[i], 2),
        "atr": round(ind.atr[i], 2),
    }


def grade(ind: Indicators, direction: str) -> str:
    """A+ needs 4 of 5 confluences aligned with the trade, A needs 3."""
    sign = 1 if direction == "LONG" else -1
    checks = [
        sign * (ind.close[-1] - ind.vwap[-1]) > 0,
        sign * (ind.ema21[-1] - ind.ema65[-1]) > 0,
        sign * (ind.ema65[-1] - ind.ema200[-1]) > 0,
        ind.adx[-1] >= 25,
        ind.rvol[-1] >= 1.5,
    ]
    n = sum(checks)
    return "A+" if n >= 4 else "A" if n >= 3 else "B"


def _rma(bars, ind, p) -> tuple[str, str] | None:
    b, e21, e30, e65 = bars[-1], ind.ema21[-1], ind.ema30[-1], ind.ema65[-1]
    if e21 > e65 and b.close > ind.vwap[-1] and b.low <= e21 and b.close >= e30:
        return "LONG", "RMA: uptrend pullback tagged the 21 EMA and held the 30 EMA"
    if e21 < e65 and b.close < ind.vwap[-1] and b.high >= e21 and b.close <= e30:
        return "SHORT", "RMA: downtrend pullback tagged the 21 EMA and held below the 30 EMA"
    return None


def _ffma(bars, ind, p) -> tuple[str, str] | None:
    r = ind.rsi[-1]
    if r > p.rsi_overbought:
        return "SHORT", f"FFMA: RSI {r:.1f} > {p.rsi_overbought:g}, fade"
    if r < p.rsi_oversold:
        return "LONG", f"FFMA: RSI {r:.1f} < {p.rsi_oversold:g}, fade"
    return None


def _trend(bars, ind, p) -> tuple[str, str] | None:
    since, direction = _bars_since_cross(ind.ema9, ind.ema15)
    if direction is None:
        return None
    compressed = abs(ind.ema9[-1] - ind.ema15[-1]) <= p.ribbon_compression_atr * max(ind.atr[-1], 1e-9)
    if not compressed:
        return None
    if since > p.max_bars_past_crossover:
        return None  # don't chase
    return direction, f"TREND: compressed 9/15 ribbon crossed {direction} {since} bar(s) ago"


def _momo(bars, ind, p) -> tuple[str, str] | None:
    b = bars[-1]
    rng = b.high - b.low
    if ind.rvol[-1] < p.momo_min_rvol or rng < p.momo_min_range_atr * ind.atr[-2] or ind.adx[-1] < 25:
        return None
    direction = "LONG" if b.close > b.open else "SHORT"
    since, cross_dir = _bars_since_cross(ind.ema9, ind.ema15)
    if since is None or cross_dir != direction or since > p.max_bars_past_crossover:
        return None  # don't chase
    return direction, f"MOMO: RVOL {ind.rvol[-1]:.1f}, range {rng:.2f} >= {p.momo_min_range_atr} ATR"


def _swing_points(values: list[float], kind: str, width: int = 2) -> list[int]:
    idx = []
    for i in range(width, len(values) - width):
        window = values[i - width:i + width + 1]
        if (kind == "low" and values[i] == min(window)) or (kind == "high" and values[i] == max(window)):
            idx.append(i)
    return idx


def _dbdt(bars, ind, p, symbol: str, lookback: int = 40) -> tuple[str, str] | None:
    family = get_contract(symbol).family
    min_range = p.dbdt_min_range_points[family]
    seg = bars[-lookback:]
    lows, highs = [b.low for b in seg], [b.high for b in seg]
    tol = 0.1 * min_range
    last = seg[-1]
    sl = _swing_points(lows, "low")
    if len(sl) >= 2:
        a, b = sl[-2], sl[-1]
        if b - a >= 5 and abs(lows[a] - lows[b]) <= tol:
            neck = max(highs[a:b + 1])
            if neck - min(lows[a], lows[b]) >= min_range and last.close > lows[b]:
                return "LONG", f"DB: double bottom {lows[a]:.2f}/{lows[b]:.2f}, height {neck - lows[b]:.2f}"
    sh = _swing_points(highs, "high")
    if len(sh) >= 2:
        a, b = sh[-2], sh[-1]
        if b - a >= 5 and abs(highs[a] - highs[b]) <= tol:
            neck = min(lows[a:b + 1])
            if max(highs[a], highs[b]) - neck >= min_range and last.close < highs[b]:
                return "SHORT", f"DT: double top {highs[a]:.2f}/{highs[b]:.2f}, height {highs[b] - neck:.2f}"
    return None


def detect_setup(bars: list[Bar], symbol: str, params: SetupParams = DEFAULT_SETUP_PARAMS) -> SetupSignal:
    if len(bars) < 30:
        return SetupSignal(None, reasons=[f"only {len(bars)} bars; need at least 30"])
    ind = compute_indicators(bars)
    snap = confluence_snapshot(ind)
    detectors = {
        "RMA": lambda: _rma(bars, ind, params),
        "FFMA": lambda: _ffma(bars, ind, params),
        "TREND": lambda: _trend(bars, ind, params),
        "MOMO": lambda: _momo(bars, ind, params),
        "DB_DT": lambda: _dbdt(bars, ind, params, symbol),
    }
    since, _ = _bars_since_cross(ind.ema9, ind.ema15)
    for name in params.priority:
        hit = detectors[name]()
        if hit:
            direction, why = hit
            return SetupSignal(name, direction, grade(ind, direction), snap, [why], since)
    return SetupSignal(None, confluence=snap, reasons=["no WSGTA setup on the last bar"],
                       bars_since_cross=since)
