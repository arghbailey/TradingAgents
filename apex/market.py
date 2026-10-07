"""Bars, indicators and market-data providers.

Indicators are plain Python so they are deterministic and dependency-free. No live
futures feed is wired in (see "Not yet built" in docs/APEX_AUTOMATION.md). Use
``CsvMarketData`` for recorded bars or ``FixtureMarketData`` offline.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Protocol, runtime_checkable

from .calendar import ET, at_et
from .config import get_contract


@dataclass(frozen=True)
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float
    tick_size: float

    @property
    def spread_ticks(self) -> float:
        return round((self.ask - self.bid) / self.tick_size, 6)


@runtime_checkable
class MarketDataProvider(Protocol):
    def bars(self, symbol: str, until: datetime) -> list[Bar]: ...
    def quote(self, symbol: str, at: datetime) -> Quote: ...


# ----------------------------------------------------------------------- indicators


def ema(values: list[float], period: int) -> list[float]:
    if not values:
        return []
    k = 2.0 / (period + 1)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def rsi(closes: list[float], period: int = 14) -> list[float]:
    """Wilder's RSI; the first ``period`` values are 50.0 (neutral warm-up)."""
    out = [50.0] * len(closes)
    if len(closes) <= period:
        return out
    gains = [max(0.0, closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    losses = [max(0.0, closes[i - 1] - closes[i]) for i in range(1, len(closes))]
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    for i in range(period, len(closes)):
        if i > period:
            avg_g = (avg_g * (period - 1) + gains[i - 1]) / period
            avg_l = (avg_l * (period - 1) + losses[i - 1]) / period
        if avg_g == 0 and avg_l == 0:
            out[i] = 50.0  # no movement at all: neutral, not overbought
        else:
            out[i] = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1 + avg_g / avg_l)
    return out


def true_ranges(bars: list[Bar]) -> list[float]:
    trs = []
    for i, b in enumerate(bars):
        if i == 0:
            trs.append(b.high - b.low)
        else:
            pc = bars[i - 1].close
            trs.append(max(b.high - b.low, abs(b.high - pc), abs(b.low - pc)))
    return trs


def _wilder(values: list[float], period: int) -> list[float]:
    out: list[float] = []
    for i, v in enumerate(values):
        if i < period:
            out.append(sum(values[: i + 1]) / (i + 1))
        else:
            out.append((out[-1] * (period - 1) + v) / period)
    return out


def atr(bars: list[Bar], period: int = 14) -> list[float]:
    return _wilder(true_ranges(bars), period)


def adx(bars: list[Bar], period: int = 14) -> list[float]:
    if len(bars) < 2:
        return [0.0] * len(bars)
    plus_dm, minus_dm = [0.0], [0.0]
    for i in range(1, len(bars)):
        up = bars[i].high - bars[i - 1].high
        down = bars[i - 1].low - bars[i].low
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
    tr = _wilder(true_ranges(bars), period)
    pdm, mdm = _wilder(plus_dm, period), _wilder(minus_dm, period)
    dx = []
    for t, p, m in zip(tr, pdm, mdm, strict=True):
        pdi = 100 * p / t if t else 0.0
        mdi = 100 * m / t if t else 0.0
        dx.append(100 * abs(pdi - mdi) / (pdi + mdi) if (pdi + mdi) else 0.0)
    return _wilder(dx, period)


def session_vwap(bars: list[Bar]) -> list[float]:
    """VWAP that resets at the start of each ET calendar day."""
    out, pv, vol, day = [], 0.0, 0.0, None
    for b in bars:
        d = b.ts.astimezone(ET).date()
        if d != day:
            pv, vol, day = 0.0, 0.0, d
        typical = (b.high + b.low + b.close) / 3
        pv += typical * b.volume
        vol += b.volume
        out.append(pv / vol if vol else b.close)
    return out


def rvol(bars: list[Bar], lookback: int = 20) -> list[float]:
    out = []
    for i, b in enumerate(bars):
        window = [x.volume for x in bars[max(0, i - lookback):i]]
        avg = sum(window) / len(window) if window else 0.0
        out.append(b.volume / avg if avg else 1.0)
    return out


@dataclass(frozen=True)
class Indicators:
    close: list[float]
    ema9: list[float]
    ema15: list[float]
    ema21: list[float]
    ema30: list[float]
    ema65: list[float]
    ema200: list[float]
    rsi: list[float]
    adx: list[float]
    atr: list[float]
    vwap: list[float]
    rvol: list[float]


def compute_indicators(bars: list[Bar]) -> Indicators:
    closes = [b.close for b in bars]
    return Indicators(
        close=closes, ema9=ema(closes, 9), ema15=ema(closes, 15), ema21=ema(closes, 21),
        ema30=ema(closes, 30), ema65=ema(closes, 65), ema200=ema(closes, 200),
        rsi=rsi(closes), adx=adx(bars), atr=atr(bars), vwap=session_vwap(bars), rvol=rvol(bars),
    )


# ------------------------------------------------------------------------ providers


def fixture_bars(symbol: str, day: date, until_hhmm: tuple[int, int] = (10, 5),
                 n_bars: int = 260) -> list[Bar]:
    """Deterministic 5-minute bars ending at ``until_hhmm`` ET on ``day``.

    A steady, gently noisy uptrend followed by a four-bar pullback that tags the
    21 EMA and holds the 30 EMA: a textbook RMA long. Prices are scaled to the
    contract family (NQ ~ 20,000, ES ~ 6,000).
    """
    family = get_contract(symbol).family
    base, step, pull = (20_000.0, 2.0, 1.1) if family == "NQ" else (6_000.0, 0.5, 1.1)
    end = at_et(day, *until_hhmm)
    start = end - timedelta(minutes=5 * (n_bars - 1))
    bars: list[Bar] = []
    price = base
    for i in range(n_bars):
        ts = start + timedelta(minutes=5 * i)
        wiggle = math.sin(i * 0.9) * step * 0.6
        if i < n_bars - 4:
            o = price
            c = price + step + wiggle * 0.3
        else:  # pullback toward the 21 EMA
            o = price
            c = price - step * pull
        h = max(o, c) + step * 0.4
        lo = min(o, c) - step * 0.4
        vol = 1_000 + 150 * math.sin(i * 0.37) ** 2
        bars.append(Bar(ts, round(o, 2), round(h, 2), round(lo, 2), round(c, 2), round(vol, 1)))
        price = c
    # Make the last bar tag the 21 EMA and close back above it.
    ind = compute_indicators(bars)
    last = bars[-1]
    tag = round(ind.ema21[-2] - step * 0.1, 2)
    bars[-1] = Bar(last.ts, last.open, last.high, min(last.low, tag), last.close, last.volume)
    return bars


class FixtureMarketData:
    """Offline market data: ``fixture_bars`` and a one-tick quote."""

    def __init__(self, day: date, until_hhmm: tuple[int, int] = (10, 5), spread_ticks: int = 1):
        self.day, self.until_hhmm, self.spread_ticks = day, until_hhmm, spread_ticks

    def bars(self, symbol: str, until: datetime) -> list[Bar]:
        return [b for b in fixture_bars(symbol, self.day, self.until_hhmm) if b.ts <= until]

    def quote(self, symbol: str, at: datetime) -> Quote:
        spec = get_contract(symbol)
        bars = self.bars(symbol, at)
        px = bars[-1].close if bars else 0.0
        return Quote(px, px + self.spread_ticks * spec.tick_size, spec.tick_size)


class CsvMarketData:
    """Recorded bars from CSV: ``ts,open,high,low,close,volume`` (ISO timestamps, ET if naive).

    One file serves every symbol it is asked for. The quote is synthesized as one
    tick wide unless ``spread_ticks`` says otherwise.
    """

    def __init__(self, path: str | Path, spread_ticks: float = 1):
        self.path, self.spread_ticks = Path(path), spread_ticks
        rows = list(csv.DictReader(self.path.open(encoding="utf-8")))
        self._bars = []
        for r in rows:
            ts = datetime.fromisoformat(r["ts"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=ET)
            self._bars.append(Bar(ts, float(r["open"]), float(r["high"]), float(r["low"]),
                                  float(r["close"]), float(r.get("volume") or 0)))

    def bars(self, symbol: str, until: datetime) -> list[Bar]:
        return [b for b in self._bars if b.ts <= until]

    def quote(self, symbol: str, at: datetime) -> Quote:
        spec = get_contract(symbol)
        bars = self.bars(symbol, at)
        px = bars[-1].close if bars else 0.0
        return Quote(px, px + self.spread_ticks * spec.tick_size, spec.tick_size)
