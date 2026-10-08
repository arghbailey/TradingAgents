"""Three NEW intraday strategy families on MNQ/MES 5m bars, under virtual Apex 50K
EOD eval economics: Opening Range Breakout (A), overnight-gap at the RTH open (B),
and ADX-regime VWAP mean reversion (C).

Self-contained by design (see the task that produced this module): does not import
``apex.backtest``/``apex.sweep``/``apex.setups`` since another agent edits those
concurrently. Reads bars directly from ``results/backtest_data/{symbol}_5m.csv`` with
pandas; reimplements its own small ATR/ADX/VWAP, cost model and walk-forward split
rather than depending on the volatile modules' internals.

Honesty rules (same shape as ``apex.sweep``):

1. The grid below (``GRID``) is fixed before any cell is run -- 56 cells/symbol,
   comfortably under the ~200 budget.
2. Bars are split by index into train (first 60%) and holdout (last 40%). A trade
   belongs to whichever side its *entry* bar index falls in (it may exit on bars
   after the split, same convention as ``apex.sweep.run_walk_forward``).
3. A cell "clears the bar" only if holdout net PnL > holdout buy-and-hold of 1 micro
   AND holdout trades >= 20. Verdicts are per symbol per family, scanning every cell
   in that family (not just the reported top-10).

Engine rules: signals read only closed 5m bars; fills are the next bar's open.
Costs: $1.04 round-trip commission once per trade, 1 tick of slippage against the
position on both the entry and exit fill. One micro contract, fixed size (no
governor/sizing integration -- see DEVIATIONS in the implementation report). Every
trade is intraday: forced flat at 15:55 ET (the last RTH 5m bar of the day).

A virtual 50K EOD eval account ($50,000 start, $2,000 trailing EOD drawdown, $1,000
daily loss stop) replays each cell's trades afterward purely to *report* breaches --
it does not feed back into trade generation.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from datetime import date as Date
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
DATA_DIR = Path("results/backtest_data")

# ----------------------------------------------------------------------- constants

TICK = {"MNQ": 0.25, "MES": 0.25}
POINT_VALUE = {"MNQ": 2.0, "MES": 5.0}  # tick value: MNQ $0.50, MES $1.25 per micro
COMMISSION_RT = 1.04  # round-trip, once per trade
SLIP_TICKS = 1  # per side

RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)
FLAT_TIME = time(15, 55)  # last RTH 5m bar; forced flat here or earlier

EVAL_START = 50_000.0
EVAL_TRAILING_DD = 2_000.0
EVAL_DAILY_LOSS = 1_000.0

MIN_HOLDOUT_TRADES = 20
WARMUP_BARS = 200  # bars before ATR/ADX/VWAP-stdev are trusted for signals

# ----------------------------------------------------------------------- grid (pre-registered)

GRID: dict[str, list[dict]] = {
    "ORB": [
        {"range_min": r, "retest": rt, "stop": s, "exit_mode": e}
        for r in (15, 30)
        for rt in (True, False)
        for s in (("opp", None), ("atr", 1.0), ("atr", 1.5))
        for e in ("R2", "time_flat")
    ],
    "GAP": [
        {"mode": m, "gap_pct": g, "atr_mult": a, "exit_mode": e}
        for m in ("fade", "follow")
        for g in (0.003, 0.005)
        for a in (1.0, 1.5)
        for e in ("gap_fill", "R2", "time_flat")
    ],
    "MEANREV": [
        {"adx_max": adx, "sigma": s, "atr_mult": a}
        for adx in (20, 25)
        for s in (1.5, 2.0)
        for a in (1.0, 1.5)
    ],
}


def grid_size() -> int:
    return sum(len(v) for v in GRID.values())


assert grid_size() <= 200, "strategies2 grid grew past the pre-registered <=200 cells/symbol budget"


def _label(params: dict) -> str:
    parts = []
    for k, v in params.items():
        if isinstance(v, tuple):
            v = f"{v[0]}{v[1]}" if v[1] is not None else v[0]
        parts.append(f"{k}={v}")
    return ",".join(parts)


# ----------------------------------------------------------------------- data loading


def load_bars(symbol: str) -> pd.DataFrame:
    path = DATA_DIR / f"{symbol}_5m.csv"
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert(ET)
    df = df.sort_values("ts").reset_index(drop=True)
    df["et_date"] = df["ts"].dt.date
    df["et_time"] = df["ts"].dt.time
    return df


# ----------------------------------------------------------------------- indicators (causal)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1, skipna=True)


def wilder_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def wilder_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_ = wilder_atr(df, period)
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_
    denom = (plus_di + minus_di).replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / denom).fillna(0.0)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP that resets at each ET calendar day boundary."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = typical * df["volume"]
    cum_pv = pv.groupby(df["et_date"]).cumsum()
    cum_vol = df["volume"].groupby(df["et_date"]).cumsum().replace(0, np.nan)
    return cum_pv / cum_vol


def with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr14"] = wilder_atr(df, 14)
    df["adx14"] = wilder_adx(df, 14)
    df["vwap"] = session_vwap(df)
    df["vwap_dev"] = df["close"] - df["vwap"]
    df["vwap_dev_std"] = df["vwap_dev"].rolling(20, min_periods=20).std()
    df["is_rth"] = (df["et_time"] >= RTH_OPEN) & (df["et_time"] < RTH_CLOSE)
    return df


# ----------------------------------------------------------------------- Family A: ORB helpers


def _add_minutes(t: time, minutes: int) -> time:
    return (datetime(2000, 1, 1, t.hour, t.minute) + timedelta(minutes=minutes)).time()


def orb_range(idxs: np.ndarray, times: np.ndarray, highs: np.ndarray, lows: np.ndarray,
              range_min: int) -> tuple[float, float, int] | None:
    """(range_high, range_low, start_pos) from bars in ``idxs`` (one day's RTH positions,
    time-ascending) with ``et_time < 09:30 + range_min``. ``start_pos`` is the first
    position in ``idxs`` at or after the range window closes."""
    range_end = _add_minutes(RTH_OPEN, range_min)
    mask = times[idxs] < range_end
    if not mask.any() or mask.all():
        return None
    range_bars = idxs[mask]
    return float(highs[range_bars].max()), float(lows[range_bars].min()), int(mask.sum())


# ----------------------------------------------------------------------- shared exit scan


def _scan_exit_rth(idxs: np.ndarray, start_pos: int, highs: np.ndarray, lows: np.ndarray,
                   closes: np.ndarray, direction: str, stop_price: float,
                   target_price: float | None) -> tuple[int, float, str] | None:
    """First bar (from ``start_pos`` in ``idxs``, one day's RTH positions) whose range
    reaches the stop or target; both in the same bar resolves to the stop (conservative).
    The last position in ``idxs`` is the day's final RTH bar (15:55) -- reaching it with
    neither hit forces a flat exit at its close."""
    sign = 1 if direction == "LONG" else -1
    for pos in range(start_pos, len(idxs)):
        i = idxs[pos]
        hit_stop = (lows[i] <= stop_price) if sign == 1 else (highs[i] >= stop_price)
        hit_target = target_price is not None and (
            (highs[i] >= target_price) if sign == 1 else (lows[i] <= target_price))
        if hit_stop:
            return int(i), stop_price, "STOP"
        if hit_target:
            return int(i), target_price, "TARGET"
        if pos == len(idxs) - 1:
            return int(i), float(closes[i]), "TIME"
    return None


def _favorable_target(entry_price: float, direction: str, target_price: float | None) -> float | None:
    """Drop a target that sits behind entry in the losing direction (e.g. a "gap fill"
    target for a trend-follow trade) rather than let it fire as a bogus instant exit."""
    if target_price is None:
        return None
    sign = 1 if direction == "LONG" else -1
    return target_price if sign * (target_price - entry_price) > 0 else None


def _apply_slippage(price: float, tick: float, direction: str, side: str) -> float:
    sign = 1 if direction == "LONG" else -1
    tick_sign = 1 if side == "entry" else -1
    return price + sign * tick_sign * SLIP_TICKS * tick


@dataclass
class Trade:
    day: Date
    direction: str
    entry_idx: int
    entry_price: float
    exit_idx: int
    exit_price: float
    reason: str
    pnl: float


def _make_trade(day: Date, direction: str, entry_i: int, entry_raw: float, exit_i: int,
                exit_raw: float, reason: str, tick: float, pv: float) -> Trade:
    entry_price = _apply_slippage(entry_raw, tick, direction, "entry")
    exit_price = _apply_slippage(exit_raw, tick, direction, "exit")
    sign = 1 if direction == "LONG" else -1
    gross = sign * (exit_price - entry_price) * pv
    pnl = round(gross - COMMISSION_RT, 2)
    return Trade(day, direction, entry_i, entry_price, exit_i, exit_price, reason, pnl)


# ----------------------------------------------------------------------- Family A: ORB


def _orb_day_trades(day: Date, idxs: np.ndarray, times: np.ndarray, opens: np.ndarray,
                    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, atr: np.ndarray,
                    params: dict, tick: float, pv: float) -> list[Trade]:
    rng = orb_range(idxs, times, highs, lows, params["range_min"])
    if rng is None:
        return []
    range_high, range_low, start_pos = rng
    retest = params["retest"]
    stop_kind, stop_mult = params["stop"]
    exit_mode = params["exit_mode"]

    state = {"LONG": "idle", "SHORT": "idle"}
    taken = {"LONG": False, "SHORT": False}
    trades: list[Trade] = []
    j = start_pos
    while j < len(idxs) - 1:
        i = idxs[j]
        if i < WARMUP_BARS:
            j += 1
            continue
        fired = None
        for direction in ("LONG", "SHORT"):
            if taken[direction]:
                continue
            broke = (closes[i] > range_high) if direction == "LONG" else (closes[i] < range_low)
            st = state[direction]
            if st == "idle" and broke:
                state[direction] = "fire" if not retest else "broken"
            elif st == "broken":
                touched = (lows[i] <= range_high) if direction == "LONG" else (highs[i] >= range_low)
                if touched:
                    state[direction] = "confirm"
            elif st == "confirm" and broke:
                state[direction] = "fire"
            if state[direction] == "fire":
                fired = direction
                break
        if fired is not None:
            direction = fired
            taken[direction] = True
            state[direction] = "done"
            entry_i = int(idxs[j + 1])
            entry_raw = float(opens[entry_i])
            sign = 1 if direction == "LONG" else -1
            if stop_kind == "opp":
                stop_price = range_low if direction == "LONG" else range_high
            else:
                stop_price = entry_raw - sign * stop_mult * float(atr[i])
            stop_dist = abs(entry_raw - stop_price)
            target_price = entry_raw + sign * 2 * stop_dist if exit_mode == "R2" else None
            result = _scan_exit_rth(idxs, j + 1, highs, lows, closes, direction, stop_price, target_price)
            if result is None:
                j += 1
                continue
            exit_i, exit_raw, reason = result
            trades.append(_make_trade(day, direction, entry_i, entry_raw, exit_i, exit_raw,
                                      reason, tick, pv))
            j = int(np.searchsorted(idxs, exit_i)) + 1
            continue
        j += 1
    return trades


# ----------------------------------------------------------------------- Family B: gap


def build_gap_table(df: pd.DataFrame) -> dict[Date, tuple[float | None, float | None]]:
    """day -> (prior RTH close, gap_pct), where ``gap_pct`` compares today's RTH open
    against the *previous trading day's* RTH close -- the previous entry in this table,
    not whatever bar happens to sit right before it in the raw series (which, for the
    first bar of a day, is an overnight Globex bar straddling the 18:00 reopen)."""
    rth = df[df["is_rth"]]
    g = rth.groupby("et_date", sort=True)
    opens = g["open"].first()
    closes = g["close"].last()
    out: dict[Date, tuple[float | None, float | None]] = {}
    prev_close: float | None = None
    for day in opens.index:
        o = float(opens[day])
        if prev_close is not None and prev_close != 0:
            out[day] = (prev_close, (o - prev_close) / prev_close)
        else:
            out[day] = (None, None)
        prev_close = float(closes[day])
    return out


def _gap_day_trades(day: Date, idxs: np.ndarray, prior_close: float | None, gap_pct: float | None,
                    opens: np.ndarray, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                    atr: np.ndarray, params: dict, tick: float, pv: float) -> list[Trade]:
    if gap_pct is None or len(idxs) < 2 or idxs[0] < WARMUP_BARS:
        return []
    if abs(gap_pct) < params["gap_pct"]:
        return []
    gap_up = gap_pct > 0
    if params["mode"] == "fade":
        direction = "SHORT" if gap_up else "LONG"
    else:
        direction = "LONG" if gap_up else "SHORT"

    i = int(idxs[0])  # signal reads the first RTH (09:30) bar, fully closed
    entry_i = int(idxs[1])
    entry_raw = float(opens[entry_i])
    sign = 1 if direction == "LONG" else -1
    stop_price = entry_raw - sign * params["atr_mult"] * float(atr[i])
    stop_dist = abs(entry_raw - stop_price)

    exit_mode = params["exit_mode"]
    if exit_mode == "gap_fill":
        target_price = prior_close
    elif exit_mode == "R2":
        target_price = entry_raw + sign * 2 * stop_dist
    else:
        target_price = None
    target_price = _favorable_target(entry_raw, direction, target_price)

    result = _scan_exit_rth(idxs, 1, highs, lows, closes, direction, stop_price, target_price)
    if result is None:
        return []
    exit_i, exit_raw, reason = result
    return [_make_trade(day, direction, entry_i, entry_raw, exit_i, exit_raw, reason, tick, pv)]


# ----------------------------------------------------------------------- Family C: mean reversion


def _meanrev_day_trades(day: Date, idxs: np.ndarray, opens: np.ndarray, highs: np.ndarray,
                        lows: np.ndarray, closes: np.ndarray, atr: np.ndarray, adx: np.ndarray,
                        vwap: np.ndarray, dev: np.ndarray, dev_std: np.ndarray, params: dict,
                        tick: float, pv: float) -> list[Trade]:
    trades: list[Trade] = []
    pos = 0
    n = len(idxs)
    while pos < n - 1:
        i = int(idxs[pos])
        sd = dev_std[i]
        if (i >= WARMUP_BARS and not math.isnan(sd) and sd > 0
                and adx[i] < params["adx_max"] and abs(dev[i]) > params["sigma"] * sd):
            direction = "SHORT" if dev[i] > 0 else "LONG"
            entry_i = int(idxs[pos + 1])
            entry_raw = float(opens[entry_i])
            sign = 1 if direction == "LONG" else -1
            stop_price = entry_raw - sign * params["atr_mult"] * float(atr[i])
            target_price = _favorable_target(entry_raw, direction, float(vwap[i]))
            result = _scan_exit_rth(idxs, pos + 1, highs, lows, closes, direction, stop_price, target_price)
            if result is None:
                pos += 1
                continue
            exit_i, exit_raw, reason = result
            trades.append(_make_trade(day, direction, entry_i, entry_raw, exit_i, exit_raw,
                                      reason, tick, pv))
            pos = int(np.searchsorted(idxs, exit_i)) + 1
            continue
        pos += 1
    return trades


# ----------------------------------------------------------------------- virtual Apex 50K EOD eval


@dataclass
class EvalAccount:
    """Read-only replay of a trade sequence through a virtual Apex 50K EOD eval
    account: $2,000 trailing drawdown ratcheting only on the realized end-of-day
    balance, $1,000 daily loss stop checked after each trade. Purely observational --
    it does not feed back into which trades a strategy takes."""

    balance: float = EVAL_START
    hwm_eod: float = EVAL_START
    day: Date | None = None
    day_start_balance: float = EVAL_START
    breached: bool = False
    breach_type: str | None = None
    breach_date: Date | None = None

    def _roll_day(self, day: Date) -> None:
        if self.day is not None and day != self.day:
            self._checkpoint_eod()
        if self.day != day:
            self.day_start_balance = self.balance
            self.day = day

    def _checkpoint_eod(self) -> None:
        self.hwm_eod = max(self.hwm_eod, self.balance)
        floor = self.hwm_eod - EVAL_TRAILING_DD
        if not self.breached and self.balance < floor:
            self.breached, self.breach_type, self.breach_date = True, "TRAILING_DD", self.day

    def realize(self, pnl: float, day: Date) -> None:
        self._roll_day(day)
        self.balance += pnl
        if not self.breached and (self.day_start_balance - self.balance) > EVAL_DAILY_LOSS:
            self.breached, self.breach_type, self.breach_date = True, "DAILY_LOSS", day

    def finalize(self) -> None:
        self._checkpoint_eod()


def eval_replay(trades: list[Trade]) -> tuple[bool, str | None]:
    acct = EvalAccount()
    for t in sorted(trades, key=lambda t: t.exit_idx):
        acct.realize(t.pnl, t.day)
    acct.finalize()
    return acct.breached, acct.breach_type


# ----------------------------------------------------------------------- cells / reporting


def buy_and_hold(df_slice: pd.DataFrame, pv: float) -> float:
    gross = (float(df_slice["close"].iat[-1]) - float(df_slice["open"].iat[0])) * pv
    return round(gross - COMMISSION_RT, 2)


@dataclass
class CellResult:
    symbol: str
    family: str
    params: str
    train_trades: int
    train_net_pnl: float
    train_bh: float
    train_breached: bool
    train_breach_type: str | None
    holdout_trades: int
    holdout_net_pnl: float
    holdout_bh: float
    holdout_breached: bool
    holdout_breach_type: str | None

    @property
    def clears_bar(self) -> bool:
        return self.holdout_net_pnl > self.holdout_bh and self.holdout_trades >= MIN_HOLDOUT_TRADES


def build_cell(symbol: str, family: str, params: dict, trades: list[Trade], df: pd.DataFrame,
              split_idx: int, pv: float) -> CellResult:
    train = [t for t in trades if t.entry_idx < split_idx]
    holdout = [t for t in trades if t.entry_idx >= split_idx]
    train_bh = buy_and_hold(df.iloc[:split_idx], pv)
    holdout_bh = buy_and_hold(df.iloc[split_idx:], pv)
    tb, tbt = eval_replay(train)
    hb, hbt = eval_replay(holdout)
    return CellResult(
        symbol, family, _label(params),
        len(train), round(sum(t.pnl for t in train), 2), train_bh, tb, tbt,
        len(holdout), round(sum(t.pnl for t in holdout), 2), holdout_bh, hb, hbt,
    )


def run_symbol(symbol: str) -> list[CellResult]:
    df = with_indicators(load_bars(symbol))
    n = len(df)
    split_idx = int(n * 0.6)
    times = df["et_time"].to_numpy()
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    atr = df["atr14"].to_numpy(dtype=float)
    adx = df["adx14"].to_numpy(dtype=float)
    vwap = df["vwap"].to_numpy(dtype=float)
    dev = df["vwap_dev"].to_numpy(dtype=float)
    dev_std = df["vwap_dev_std"].to_numpy(dtype=float)
    tick, pv = TICK[symbol], POINT_VALUE[symbol]

    rth = df[df["is_rth"]]
    day_to_positions: dict[Date, np.ndarray] = {
        d: grp.index.to_numpy() for d, grp in rth.groupby("et_date")
    }
    gap_table = build_gap_table(df)

    results: list[CellResult] = []
    for family, cells in GRID.items():
        for params in cells:
            trades: list[Trade] = []
            for day, idxs in day_to_positions.items():
                if family == "ORB":
                    trades += _orb_day_trades(day, idxs, times, opens, highs, lows, closes, atr,
                                              params, tick, pv)
                elif family == "GAP":
                    prior_close, gap_pct = gap_table.get(day, (None, None))
                    trades += _gap_day_trades(day, idxs, prior_close, gap_pct, opens, highs, lows,
                                              closes, atr, params, tick, pv)
                else:
                    trades += _meanrev_day_trades(day, idxs, opens, highs, lows, closes, atr, adx,
                                                  vwap, dev, dev_std, params, tick, pv)
            results.append(build_cell(symbol, family, params, trades, df, split_idx, pv))
    return results


def write_csv(cells: list[CellResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "family", "params", "train_trades", "train_net_pnl", "train_bh",
                   "train_breached", "train_breach_type", "holdout_trades", "holdout_net_pnl",
                   "holdout_bh", "holdout_breached", "holdout_breach_type", "clears_bar"])
        for c in cells:
            w.writerow([c.symbol, c.family, c.params, c.train_trades, c.train_net_pnl, c.train_bh,
                       c.train_breached, c.train_breach_type, c.holdout_trades, c.holdout_net_pnl,
                       c.holdout_bh, c.holdout_breached, c.holdout_breach_type, c.clears_bar])


def verdict_for_family(cells: list[CellResult]) -> str:
    for c in cells:
        if c.clears_bar:
            return f"EDGE_FOUND({c.params})"
    return "NO_EDGE"


def render_markdown(all_cells: dict[str, list[CellResult]]) -> str:
    lines = [
        "# Apex strategies2 sweep: Opening Range Breakout / Gap / VWAP mean reversion",
        "",
        f"Grid size: {grid_size()} cells/symbol (pre-registered budget: <=200).",
        "",
        "**Multiple-comparisons caveat:** the top-10 train cells per symbol are the "
        "best of many tries; holdout, not train rank, decides the verdict.",
        "",
    ]
    for symbol, cells in all_cells.items():
        lines += [f"## {symbol}", "", "### Top-10 train cells, with holdout", "",
                 "| Family | Params | Train trades | Train PnL | Train B&H | Holdout trades | "
                 "Holdout PnL | Holdout B&H | Clears bar |",
                 "|---|---|---|---|---|---|---|---|---|"]
        top = sorted(cells, key=lambda c: c.train_net_pnl, reverse=True)[:10]
        for c in top:
            lines.append(
                f"| {c.family} | {c.params} | {c.train_trades} | {c.train_net_pnl:.2f} | "
                f"{c.train_bh:.2f} | {c.holdout_trades} | {c.holdout_net_pnl:.2f} | "
                f"{c.holdout_bh:.2f} | {'YES' if c.clears_bar else 'no'} |"
            )
        lines += ["", "### Verdicts (scanning every cell in the family, not just the top-10)", ""]
        for family in GRID:
            fam_cells = [c for c in cells if c.family == family]
            lines.append(f"- **{family}**: {verdict_for_family(fam_cells)}")
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="apex.strategies2")
    ap.add_argument("--symbol", choices=["MNQ", "MES", "all"], default="all")
    ap.add_argument("--out-dir", default="results/backtest")
    args = ap.parse_args(argv)

    symbols = ["MNQ", "MES"] if args.symbol == "all" else [args.symbol]
    out_dir = Path(args.out_dir)
    all_cells: dict[str, list[CellResult]] = {}
    for symbol in symbols:
        cells = run_symbol(symbol)
        all_cells[symbol] = cells
        write_csv(cells, out_dir / f"strat2_{symbol}.csv")
        verdicts = {fam: verdict_for_family([c for c in cells if c.family == fam]) for fam in GRID}
        print(f"{symbol}: {len(cells)} cells -> {verdicts}")

    report_path = out_dir / "strat2_report.md"
    report_path.write_text(render_markdown(all_cells), encoding="utf-8")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
