"""Pre-registered walk-forward parameter sweep: does ANY WSGTA variant beat
buy-and-hold out-of-sample, or does none?

Anti-cherry-picking rules (see docs/APEX_AUTOMATION.md and the task that produced
this module):

1. The grid below is fixed before any cell is run. Timeframes, session filters,
   regime filters and per-setup knob ranges are declared as data, not discovered.
2. Each symbol's bars are split by time into train (first 60%) and holdout (last
   40%). Selection (which cells look good) happens on train only; every cell's
   train result is written to CSV, and only the top-10 train cells per symbol get
   their holdout result reported.
3. A cell "clears the bar" only if holdout net PnL > holdout buy-and-hold of 1
   micro AND holdout trades >= 20. One clearing cell is enough to call EDGE_FOUND;
   zero is NO_EDGE.
4. Indicators and signals are computed once per (timeframe, setup knob variant) --
   they're causal recurrences over the whole bar series, so slicing where trades
   are *allowed to open* (via ``run_walk_forward``'s ``i_start``/``i_end``) is
   equivalent to recomputing from scratch on a prefix, and a lot cheaper. Session
   and regime filters are independent of setup params, so they reuse one indicator
   pass per timeframe.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import time
from pathlib import Path

from .backtest import MIN_BARS_FOR_SIGNAL, buy_and_hold_pnl, compute_signals, load_bars, run_walk_forward
from .calendar import ET
from .config import DEFAULT_SETUP_PARAMS, SetupParams
from .market import Bar, Indicators, compute_indicators
from .setups import SETUPS, SetupSignal

# ----------------------------------------------------------------------- grid

TIMEFRAMES: dict[str, int | None] = {"1h": 1, "2h": 2, "4h": 4, "1d": None}  # None = calendar day
# 5m-sourced sweep resamples causally off 5m bars instead of 1h: 1 bar = 5 min, so
# 15m/30m/1h are 3/6/12 native bars apiece. See apex.backtest.load_bars(data="5m").
TIMEFRAMES_5M: dict[str, int | None] = {"5m": 1, "15m": 3, "30m": 6, "1h": 12}
TIMEFRAMES_BY_DATA: dict[str, dict[str, int | None]] = {"1h": TIMEFRAMES, "5m": TIMEFRAMES_5M}
SESSIONS = ("all", "rth", "rth_first2h")
REGIMES = ("none", "adx25", "ema200_long_only")


def _setup_param_grid() -> dict[str, list[tuple[str, SetupParams]]]:
    """2-3 values per knob, pulled from ``apex.config.SetupParams`` field names."""
    p = DEFAULT_SETUP_PARAMS
    return {
        "RMA": [("default", p)],
        "FFMA": [
            (f"rsi_overbought={ob},rsi_oversold={os}", replace(p, rsi_overbought=ob, rsi_oversold=os))
            for ob in (75.0, 80.0, 85.0) for os in (15.0, 20.0, 25.0)
        ],
        "TREND": [
            (f"ribbon_compression_atr={c},max_bars_past_crossover={m}",
             replace(p, ribbon_compression_atr=c, max_bars_past_crossover=m))
            for c in (0.10, 0.15, 0.20) for m in (2, 3, 5)
        ],
        "MOMO": [
            (f"momo_min_rvol={r},momo_min_range_atr={a}",
             replace(p, momo_min_rvol=r, momo_min_range_atr=a))
            for r in (1.5, 2.0, 2.5) for a in (1.0, 1.5, 2.0)
        ],
        "DB_DT": [
            (f"dbdt_scale={s}",
             replace(p, dbdt_min_range_points={k: v * s for k, v in p.dbdt_min_range_points.items()}))
            for s in (0.7, 1.0, 1.3)
        ],
    }


def grid_size_per_symbol(data: str = "1h") -> int:
    knobs = sum(len(v) for v in _setup_param_grid().values())
    return knobs * len(TIMEFRAMES_BY_DATA[data]) * len(SESSIONS) * len(REGIMES)


assert grid_size_per_symbol("1h") <= 1500, "sweep grid grew past the pre-registered <=1500 cells/symbol budget"
assert grid_size_per_symbol("5m") <= 1500, "sweep grid grew past the pre-registered <=1500 cells/symbol budget"
assert set(_setup_param_grid()) == set(SETUPS)

# ----------------------------------------------------------------------- resampling


def _agg_chunk(chunk: list[Bar]) -> Bar:
    return Bar(chunk[0].ts, chunk[0].open, max(b.high for b in chunk), min(b.low for b in chunk),
              chunk[-1].close, sum(b.volume for b in chunk))


def resample_count(bars: list[Bar], n: int) -> list[Bar]:
    """Every ``n`` consecutive bars become one. Causal: bucket ``k`` only ever reads
    bars already known by the time bucket ``k`` closes; a trailing partial bucket is
    dropped rather than filled in with bars that don't exist yet."""
    return [_agg_chunk(bars[i:i + n]) for i in range(0, (len(bars) // n) * n, n)]


def resample_daily(bars: list[Bar]) -> list[Bar]:
    """One bar per ET calendar day. Causal for the same reason: each day's bucket
    is built only from that day's already-seen bars."""
    out: list[Bar] = []
    chunk: list[Bar] = []
    cur = None
    for b in bars:
        d = b.ts.astimezone(ET).date()
        if cur is not None and d != cur:
            out.append(_agg_chunk(chunk))
            chunk = []
        chunk.append(b)
        cur = d
    if chunk:
        out.append(_agg_chunk(chunk))
    return out


def resample(bars: list[Bar], timeframe: str, data: str = "1h") -> list[Bar]:
    n = TIMEFRAMES_BY_DATA[data][timeframe]
    return bars if n == 1 else (resample_daily(bars) if n is None else resample_count(bars, n))


# ----------------------------------------------------------------------- filters


def _session_ok(bar: Bar, session: str) -> bool:
    if session == "all":
        return True
    t = bar.ts.astimezone(ET).time()
    if session == "rth":
        return time(9, 30) <= t < time(16, 0)
    if session == "rth_first2h":
        return time(9, 30) <= t < time(11, 30)
    raise ValueError(f"unknown session filter {session!r}")


def _regime_ok(ind: Indicators, i: int, direction: str, regime: str) -> bool:
    if regime == "none":
        return True
    if regime == "adx25":
        return ind.adx[i] >= 25
    if regime == "ema200_long_only":
        return direction == "LONG" and ind.close[i] > ind.ema200[i]
    raise ValueError(f"unknown regime filter {regime!r}")


def filter_signals(bars: list[Bar], ind: Indicators, signals: list[SetupSignal | None],
                   session: str, regime: str) -> list[SetupSignal | None]:
    out = list(signals)
    for i, sig in enumerate(signals):
        if sig is None or sig.direction is None:
            continue
        if not (_session_ok(bars[i], session) and _regime_ok(ind, i, sig.direction, regime)):
            out[i] = None
    return out


# ----------------------------------------------------------------------- walk-forward split


def split_index(n: int, frac: float = 0.6) -> int:
    """Index where train ends and holdout begins: the first ``frac`` of bars are
    train, the rest holdout. No overlap -- trades may only open at or after this
    index on the holdout side, and strictly before it on the train side."""
    return max(MIN_BARS_FOR_SIGNAL + 1, min(n - 1, int(n * frac)))


# ----------------------------------------------------------------------- cells


@dataclass
class SweepCell:
    symbol: str
    setup: str
    timeframe: str
    session: str
    regime: str
    params: str
    train_trades: int
    train_net_pnl: float
    train_bh: float
    holdout_trades: int
    holdout_net_pnl: float
    holdout_bh: float

    @property
    def clears_bar(self) -> bool:
        return self.holdout_net_pnl > self.holdout_bh and self.holdout_trades >= 20


def run_sweep(symbol: str, tier_name: str = "50K", eval_type: str = "EOD",
             data: str = "1h") -> list[SweepCell]:
    bars_native = load_bars(symbol, data=data)
    param_grid = _setup_param_grid()
    cells: list[SweepCell] = []

    for tf in TIMEFRAMES_BY_DATA[data]:
        bars_tf = resample(bars_native, tf, data=data)
        if len(bars_tf) < MIN_BARS_FOR_SIGNAL + 20:
            continue
        ind_tf = compute_indicators(bars_tf)
        train_end = split_index(len(bars_tf))

        for setup_name, variants in param_grid.items():
            for label, params in variants:
                signals = compute_signals(symbol, bars_tf, params=params)
                for session in SESSIONS:
                    for regime in REGIMES:
                        filtered = filter_signals(bars_tf, ind_tf, signals, session, regime)
                        train_trades, _, _ = run_walk_forward(
                            symbol, setup_name, bars_tf, tier_name, eval_type,
                            signals=filtered, i_start=MIN_BARS_FOR_SIGNAL, i_end=train_end)
                        holdout_trades, _, _ = run_walk_forward(
                            symbol, setup_name, bars_tf, tier_name, eval_type,
                            signals=filtered, i_start=train_end, i_end=len(bars_tf))
                        cells.append(SweepCell(
                            symbol, setup_name, tf, session, regime, label,
                            len(train_trades), round(sum(t.pnl for t in train_trades), 2),
                            buy_and_hold_pnl(bars_tf[:train_end], symbol),
                            len(holdout_trades), round(sum(t.pnl for t in holdout_trades), 2),
                            buy_and_hold_pnl(bars_tf[train_end:], symbol),
                        ))
    return cells


# ----------------------------------------------------------------------- reporting


def top_train_cells(cells: list[SweepCell], n: int = 10) -> list[SweepCell]:
    return sorted(cells, key=lambda c: c.train_net_pnl, reverse=True)[:n]


def verdict(top: list[SweepCell]) -> str:
    for c in top:
        if c.clears_bar:
            return (f"EDGE_FOUND({c.setup}/{c.timeframe}/{c.session}/{c.regime}/{c.params})")
    return "NO_EDGE"


def write_csv(cells: list[SweepCell], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "setup", "timeframe", "session", "regime", "params",
                   "train_trades", "train_net_pnl", "train_bh",
                   "holdout_trades", "holdout_net_pnl", "holdout_bh", "clears_bar"])
        for c in cells:
            w.writerow([c.symbol, c.setup, c.timeframe, c.session, c.regime, c.params,
                       c.train_trades, c.train_net_pnl, c.train_bh,
                       c.holdout_trades, c.holdout_net_pnl, c.holdout_bh, c.clears_bar])


def render_markdown(symbol: str, cells: list[SweepCell], data: str = "1h") -> str:
    top = top_train_cells(cells)
    lines = [
        f"# Apex WSGTA sweep: {symbol} ({data} bars)",
        "",
        f"Grid size: {len(cells)} cells (pre-registered budget: {grid_size_per_symbol(data)}/symbol).",
        "",
        "**Multiple-comparisons caveat:** the top-10 train cells are the best of "
        f"{len(cells)} tries; expect some of that train edge to be best-of-N luck rather "
        "than a real effect. Holdout, not train rank, is what decides the verdict.",
        "",
        "## Top-10 train cells, with holdout",
        "",
        "| Setup | TF | Session | Regime | Params | Train trades | Train PnL | Train B&H | "
        "Holdout trades | Holdout PnL | Holdout B&H | Clears bar |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in top:
        lines.append(
            f"| {c.setup} | {c.timeframe} | {c.session} | {c.regime} | {c.params} | "
            f"{c.train_trades} | {c.train_net_pnl:.2f} | {c.train_bh:.2f} | "
            f"{c.holdout_trades} | {c.holdout_net_pnl:.2f} | {c.holdout_bh:.2f} | "
            f"{'YES' if c.clears_bar else 'no'} |"
        )
    lines += ["", f"**Verdict: {verdict(top)}**", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="apex.sweep")
    ap.add_argument("--symbol", choices=["MNQ", "MES", "all"], default="all")
    ap.add_argument("--tier", default="50K")
    ap.add_argument("--eval-type", default="EOD")
    ap.add_argument("--data", choices=["1h", "5m"], default="1h", help="bar source")
    ap.add_argument("--out-dir", default="results/backtest")
    args = ap.parse_args(argv)

    symbols = ["MNQ", "MES"] if args.symbol == "all" else [args.symbol]
    out_dir = Path(args.out_dir)
    prefix = "sweep5m" if args.data == "5m" else "sweep"
    for symbol in symbols:
        cells = run_sweep(symbol, args.tier, args.eval_type, data=args.data)
        write_csv(cells, out_dir / f"{prefix}_{symbol}.csv")
        md = render_markdown(symbol, cells, data=args.data)
        (out_dir / f"{prefix}_{symbol}.md").write_text(md, encoding="utf-8")
        print(f"{symbol}: {len(cells)} cells -> {verdict(top_train_cells(cells))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
