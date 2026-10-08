"""Apex futures backtest harness: do any of the 5 WSGTA setups beat buy-and-hold,
and can they pass an Apex evaluation?

Bar-by-bar walk-forward on 1h bars. A signal is read off the last *closed* bar and
filled at the next bar's open (no lookahead). Every entry is routed through the real
``ApexRiskGovernor`` for sizing/approval, so session-clock, daily-loss and trailing-DD
rules all apply exactly as they do live. See ``docs/APEX_AUTOMATION.md`` ("Backtest
harness for Apex rules" was previously unbuilt).

Simplifications (see DEVIATIONS in the implementation report):

* One exit per trade (stop or fixed target, whichever the bar range reaches first;
  both-in-one-bar is scored as the stop, the conservative assumption). The live
  C1/C2 partial-fill bracket in ``apex.execution`` is not replayed here.
* No intrabar mark-to-market: a trade's P&L realizes atomically at the exit bar,
  so LEGACY intraday trailing-DD ratcheting on unrealized equity is not modeled
  (only EOD realized-balance ratcheting is exercised in practice).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from .calendar import ET
from .config import (
    DEFAULT_BRACKET_PARAMS,
    DEFAULT_RISK_PARAMS,
    DEFAULT_SETUP_PARAMS,
    EvaluationType,
    SetupParams,
    get_contract,
    get_tier,
)
from .market import Bar, CsvMarketData, Indicators, compute_indicators
from .risk import AccountState, ApexRiskGovernor, MarketContext, OrderProposal, SessionPhase, Verdict, session_phase
from .setups import SETUPS, SetupSignal, _swing_points, detect_setup

# ----------------------------------------------------------------------- constants

# $1.04 round-trip commission per micro contract (MNQ/MES), charged once per trade.
COMMISSION_PER_MICRO_RT = 1.04
# 1 tick of slippage per side (entry and exit), against the trade's direction.
SLIPPAGE_TICKS_PER_SIDE = 1

# Tick value sanity (point_value * tick_size), verified against apex.config.CONTRACTS:
#   MNQ: $2.00/pt * 0.25 = $0.50/tick/micro
#   MES: $5.00/pt * 0.25 = $1.25/tick/micro
TICK_VALUE = {sym: get_contract(sym).tick_value for sym in ("MNQ", "MES")}

YF_SYMBOLS = {"MNQ": "MNQ=F", "MES": "MES=F"}
CACHE_DIR = Path("results/backtest_data")
MIN_BARS_FOR_SIGNAL = 30

# ----------------------------------------------------------------------- data


def cache_path(symbol: str, data: str = "1h") -> Path:
    return CACHE_DIR / f"{symbol}_{data}.csv"


def fetch_and_cache(symbol: str, period: str = "730d") -> Path:
    """Download 1h bars from yfinance and cache them as CSV (``apex.market`` format)."""
    import yfinance as yf

    path = cache_path(symbol)
    if path.exists():
        return path
    hist = yf.Ticker(YF_SYMBOLS[symbol]).history(period=period, interval="1h")
    if hist.empty:
        raise RuntimeError(f"yfinance returned no 1h bars for {YF_SYMBOLS[symbol]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts", "open", "high", "low", "close", "volume"])
        for ts, row in hist.iterrows():
            w.writerow([ts.isoformat(), row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]])
    return path


def load_bars(symbol: str, refresh: bool = False, data: str = "1h") -> list[Bar]:
    """``data="1h"`` fetches/caches yfinance bars as before. ``data="5m"`` reads the
    real, pre-pulled ``results/backtest_data/<SYM>_5m.csv`` (see 5m_SOURCE.md) --
    there is no yfinance 5m source, so ``refresh`` only applies to ``"1h"``.
    """
    if data == "1h":
        if refresh:
            p = cache_path(symbol, data)
            if p.exists():
                p.unlink()
        path = fetch_and_cache(symbol)
    else:
        path = cache_path(symbol, data)
        if not path.exists():
            raise FileNotFoundError(f"no cached {data} bars for {symbol} at {path}")
    far_future = datetime(2100, 1, 1, tzinfo=ET)
    return CsvMarketData(path).bars(symbol, until=far_future)


# ----------------------------------------------------------------------- trades


@dataclass
class Trade:
    setup: str
    symbol: str
    direction: str
    contracts: int
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime
    exit_price: float
    exit_reason: str  # "STOP" or "TARGET"
    pnl: float


def _apply_slippage(price: float, tick: float, direction: str, side: str) -> float:
    """Worsen ``price`` by one tick against the position.

    LONG pays more to enter (+tick) and receives less to exit (-tick); SHORT mirrors it.
    """
    sign = 1 if direction == "LONG" else -1
    tick_sign = 1 if side == "entry" else -1
    return price + sign * tick_sign * SLIPPAGE_TICKS_PER_SIDE * tick


def _scan_exit(bars: list[Bar], start_idx: int, direction: str, stop_price: float,
               target_price: float) -> tuple[int, float, str] | None:
    """First bar from ``start_idx`` whose range reaches the stop or target.

    Both reached in the same bar resolves to the stop (conservative).
    """
    sign = 1 if direction == "LONG" else -1
    for i in range(start_idx, len(bars)):
        b = bars[i]
        hit_stop = (b.low <= stop_price) if sign == 1 else (b.high >= stop_price)
        hit_target = (b.high >= target_price) if sign == 1 else (b.low <= target_price)
        if hit_stop:
            return i, stop_price, "STOP"
        if hit_target:
            return i, target_price, "TARGET"
    return None


def _bars_since_cross_at(fast: list[float], slow: list[float], c: int) -> tuple[int | None, str | None]:
    """``apex.setups._bars_since_cross`` indexed at absolute position ``c`` instead of
    on a truncated list, so the caller doesn't have to slice (and copy) on every bar."""
    for back in range(1, c + 1):
        i = c - back + 1
        prev = fast[i - 1] - slow[i - 1]
        curr = fast[i] - slow[i]
        if prev <= 0 < curr:
            return back - 1, "LONG"
        if prev >= 0 > curr:
            return back - 1, "SHORT"
    return None, None


def _signal_at(bars: list[Bar], ind: Indicators, c: int, symbol: str,
               p: SetupParams) -> SetupSignal:
    """``apex.setups.detect_setup``, indexed at ``c`` against indicators precomputed
    once for the whole series instead of recomputed from scratch every bar.

    Every WSGTA indicator (EMA/RSI/ATR/ADX/VWAP/RVOL) is a causal recurrence over
    bars[0:i], so ``compute_indicators(bars)[i]`` and
    ``compute_indicators(bars[: i + 1])[-1]`` are identical -- this only changes how
    the value is looked up, not what's computed. Equivalence with ``detect_setup`` is
    checked in ``tests/test_apex_backtest.py``.
    """
    b = bars[c]
    snap = {
        "close": round(ind.close[c], 2), "vwap": round(ind.vwap[c], 2),
        "ema21": round(ind.ema21[c], 2), "ema30": round(ind.ema30[c], 2),
        "ema65": round(ind.ema65[c], 2), "ema200": round(ind.ema200[c], 2),
        "adx": round(ind.adx[c], 2), "rvol": round(ind.rvol[c], 2),
        "rsi": round(ind.rsi[c], 2), "atr": round(ind.atr[c], 2),
    }

    def rma() -> tuple[str, str] | None:
        e21, e30, e65 = ind.ema21[c], ind.ema30[c], ind.ema65[c]
        if e21 > e65 and b.close > ind.vwap[c] and b.low <= e21 and b.close >= e30:
            return "LONG", "RMA: uptrend pullback tagged the 21 EMA and held the 30 EMA"
        if e21 < e65 and b.close < ind.vwap[c] and b.high >= e21 and b.close <= e30:
            return "SHORT", "RMA: downtrend pullback tagged the 21 EMA and held below the 30 EMA"
        return None

    def ffma() -> tuple[str, str] | None:
        r = ind.rsi[c]
        if r > p.rsi_overbought:
            return "SHORT", f"FFMA: RSI {r:.1f} > {p.rsi_overbought:g}, fade"
        if r < p.rsi_oversold:
            return "LONG", f"FFMA: RSI {r:.1f} < {p.rsi_oversold:g}, fade"
        return None

    def trend() -> tuple[str, str] | None:
        since, direction = _bars_since_cross_at(ind.ema9, ind.ema15, c)
        if direction is None:
            return None
        compressed = abs(ind.ema9[c] - ind.ema15[c]) <= p.ribbon_compression_atr * max(ind.atr[c], 1e-9)
        if not compressed or since > p.max_bars_past_crossover:
            return None
        return direction, f"TREND: compressed 9/15 ribbon crossed {direction} {since} bar(s) ago"

    def momo() -> tuple[str, str] | None:
        rng = b.high - b.low
        if ind.rvol[c] < p.momo_min_rvol or rng < p.momo_min_range_atr * ind.atr[c - 1] or ind.adx[c] < 25:
            return None
        direction = "LONG" if b.close > b.open else "SHORT"
        since, cross_dir = _bars_since_cross_at(ind.ema9, ind.ema15, c)
        if since is None or cross_dir != direction or since > p.max_bars_past_crossover:
            return None
        return direction, f"MOMO: RVOL {ind.rvol[c]:.1f}, range {rng:.2f} >= {p.momo_min_range_atr} ATR"

    def dbdt(lookback: int = 40) -> tuple[str, str] | None:
        family = get_contract(symbol).family
        min_range = p.dbdt_min_range_points[family]
        seg = bars[max(0, c - lookback + 1): c + 1]
        lows, highs = [x.low for x in seg], [x.high for x in seg]
        tol = 0.1 * min_range
        last = seg[-1]
        sl = _swing_points(lows, "low")
        if len(sl) >= 2:
            a, bb = sl[-2], sl[-1]
            if bb - a >= 5 and abs(lows[a] - lows[bb]) <= tol:
                neck = max(highs[a:bb + 1])
                if neck - min(lows[a], lows[bb]) >= min_range and last.close > lows[bb]:
                    return "LONG", f"DB: double bottom {lows[a]:.2f}/{lows[bb]:.2f}"
        sh = _swing_points(highs, "high")
        if len(sh) >= 2:
            a, bb = sh[-2], sh[-1]
            if bb - a >= 5 and abs(highs[a] - highs[bb]) <= tol:
                neck = min(lows[a:bb + 1])
                if max(highs[a], highs[bb]) - neck >= min_range and last.close < highs[bb]:
                    return "SHORT", f"DT: double top {highs[a]:.2f}/{highs[bb]:.2f}"
        return None

    detectors = {"RMA": rma, "FFMA": ffma, "TREND": trend, "MOMO": momo, "DB_DT": dbdt}
    since, _ = _bars_since_cross_at(ind.ema9, ind.ema15, c)
    for name in p.priority:
        hit = detectors[name]()
        if hit:
            direction, why = hit
            sign = 1 if direction == "LONG" else -1
            checks = [
                sign * (ind.close[c] - ind.vwap[c]) > 0, sign * (ind.ema21[c] - ind.ema65[c]) > 0,
                sign * (ind.ema65[c] - ind.ema200[c]) > 0, ind.adx[c] >= 25, ind.rvol[c] >= 1.5,
            ]
            n_align = sum(checks)
            g = "A+" if n_align >= 4 else "A" if n_align >= 3 else "B"
            return SetupSignal(name, direction, g, snap, [why], since)
    return SetupSignal(None, confluence=snap, reasons=["no WSGTA setup on the last bar"],
                       bars_since_cross=since)


def compute_signals(symbol: str, bars: list[Bar],
                    params: SetupParams = DEFAULT_SETUP_PARAMS) -> list[SetupSignal | None]:
    """The WSGTA signal at each bar index, reading only that bar and everything before
    it -- the no-lookahead contract. Index < MIN_BARS_FOR_SIGNAL is ``None``. Indicators
    are computed once for the whole series (they're all causal) and shared across every
    setup's walk-forward, which is what keeps a 700-day 1h backtest fast."""
    if len(bars) < MIN_BARS_FOR_SIGNAL:
        return [None] * len(bars)
    ind = compute_indicators(bars)
    sigs: list[SetupSignal | None] = [None] * len(bars)
    for i in range(MIN_BARS_FOR_SIGNAL, len(bars)):
        sigs[i] = _signal_at(bars, ind, i, symbol, params)
    return sigs


def run_walk_forward(symbol: str, setup_name: str, bars: list[Bar], tier_name: str,
                     eval_type: str,
                     signals: list[SetupSignal | None] | None = None,
                     i_start: int | None = None, i_end: int | None = None,
                     flatten_sink: list[datetime] | None = None,
                     ) -> tuple[list[Trade], AccountState, str | None]:
    """Bar-by-bar walk-forward for one setup/symbol. Returns closed trades, the
    long-running account they were traded through (sizing, DD and daily-loss are all
    governed live), and the date the account first breached (if any; once breached the
    governor rejects every further order on its own, same as live).

    ``i_start``/``i_end`` restrict which bar indices may open a *new* trade (default:
    the whole series) without truncating ``bars`` itself, so a trade opened near the
    end of a window can still exit on bars after it. Used by ``apex.sweep`` to run a
    fresh account over a train or holdout slice of one causal signal series.

    ``flatten_sink``, if given, collects the fill timestamp of every signal bar that
    lands in the governor's 15:55-16:00 ET mandatory-flatten window (the rule is
    always rejected there; this just confirms real bar timestamps actually fall in
    that 5-minute-wide window -- on 1h bars, which land on the hour, they never do).
    """
    spec = get_contract(symbol)
    tier = get_tier(tier_name)
    account = AccountState.fresh(tier, EvaluationType.parse(eval_type))
    governor = ApexRiskGovernor(DEFAULT_RISK_PARAMS)
    stop_pts = DEFAULT_BRACKET_PARAMS.stop_points[symbol]
    target_pts = DEFAULT_BRACKET_PARAMS.c1_target_points[symbol]
    signals = signals if signals is not None else compute_signals(symbol, bars)

    trades: list[Trade] = []
    blown_date: str | None = None
    current_day: date | None = None
    i = i_start if i_start is not None else MIN_BARS_FOR_SIGNAL
    n = (i_end if i_end is not None else len(bars))
    n = min(n, len(bars))
    while i < n - 1:
        day = bars[i].ts.date()
        if current_day is not None and day != current_day:
            account.end_of_day()
            account.new_session()
        current_day = day

        sig = signals[i]  # signal reads only bars up to and including i (no lookahead)
        if sig is None or sig.setup != setup_name or sig.direction is None:
            i += 1
            continue

        fill_bar = bars[i + 1]
        if flatten_sink is not None and session_phase(fill_bar.ts) is SessionPhase.FLATTEN:
            flatten_sink.append(fill_bar.ts)
        entry_raw = fill_bar.open
        order = OrderProposal(symbol, sig.direction, tier.max_contracts_micro, entry_raw,
                              stop_pts, sig.grade)
        market = MarketContext(now=fill_bar.ts, spread_ticks=1.0)
        decision = governor.evaluate(account, market, order)
        if decision.verdict is not Verdict.APPROVED or decision.contracts < 1:
            i += 1
            continue

        contracts = decision.contracts
        entry = _apply_slippage(entry_raw, spec.tick_size, sig.direction, "entry")
        sign = 1 if sig.direction == "LONG" else -1
        stop_price = entry_raw - sign * stop_pts
        target_price = entry_raw + sign * target_pts

        hit = _scan_exit(bars, i + 1, sig.direction, stop_price, target_price)
        if hit is None:
            break  # ran off the end of the data still open: stop the walk here
        exit_idx, exit_raw, reason = hit
        exit_price = _apply_slippage(exit_raw, spec.tick_size, sig.direction, "exit")

        gross = sign * (exit_price - entry) * spec.point_value * contracts
        commission = COMMISSION_PER_MICRO_RT * contracts
        pnl = gross - commission

        account.open_contracts = contracts
        account.realize(pnl, stopped_out=(reason == "STOP"), at=bars[exit_idx].ts)
        account.open_contracts = 0
        if account.failed and blown_date is None:
            blown_date = bars[exit_idx].ts.date().isoformat()

        trades.append(Trade(setup_name, symbol, sig.direction, contracts, fill_bar.ts, entry,
                            bars[exit_idx].ts, exit_price, reason, round(pnl, 2)))
        i = exit_idx + 1  # flat until the exit bar closes

    if current_day is not None:
        account.end_of_day()
    return trades, account, blown_date


# ----------------------------------------------------------------------- metrics


def buy_and_hold_pnl(bars: list[Bar], symbol: str) -> float:
    """Net P&L of buying 1 micro at the first bar's open and holding to the last close."""
    spec = get_contract(symbol)
    gross = (bars[-1].close - bars[0].open) * spec.point_value
    return round(gross - COMMISSION_PER_MICRO_RT, 2)


def eval_window_fraction(trades: list[Trade], tier_name: str, eval_type: str,
                         window_days: int = 30) -> tuple[float, int]:
    """Fraction of rolling ``window_days`` windows (one per trading day with >=1 trade
    exit) in which a fresh account hits the profit target before breaching trailing DD.

    Reuses the trade P&Ls from the full walk-forward (sizing already governed); it does
    not re-run the governor at the smaller fresh-account buffer size each window.
    """
    if not trades:
        return 0.0, 0
    tier = get_tier(tier_name)
    et = EvaluationType.parse(eval_type)
    days = sorted({t.exit_ts.date() for t in trades})
    hits = 0
    for start in days:
        end = start + timedelta(days=window_days)
        window = [t for t in trades if start <= t.exit_ts.date() < end]
        if not window:
            continue
        acct = AccountState.fresh(tier, et)
        current_day: date | None = None
        reached_target = False
        breached = False
        for t in window:
            d = t.exit_ts.date()
            if current_day is not None and d != current_day:
                acct.end_of_day()
                acct.new_session()
            current_day = d
            acct.realize(t.pnl, stopped_out=(t.exit_reason == "STOP"), at=t.exit_ts)
            if acct.failed:
                breached = True
                break
            if acct.balance - tier.nominal_size >= tier.profit_target:
                reached_target = True
                break
        if reached_target and not breached:
            hits += 1
    windows_with_trades = sum(1 for start in days
                              if any(start <= t.exit_ts.date() < start + timedelta(days=window_days)
                                    for t in trades))
    return (hits / windows_with_trades if windows_with_trades else 0.0), windows_with_trades


@dataclass
class BacktestResult:
    symbol: str
    setup: str
    trades: int
    net_pnl: float
    win_rate: float
    eval_blown: bool
    eval_blown_date: str | None
    buy_hold_pnl: float
    beats_bh: bool
    eval_pass_fraction: float
    eval_windows: int
    flatten_events: int = 0


def backtest_symbol_setup(symbol: str, setup_name: str, bars: list[Bar], tier_name: str,
                          eval_type: str,
                          signals: list[SetupSignal | None] | None = None) -> BacktestResult:
    flatten_sink: list[datetime] = []
    trades, account, blown_date = run_walk_forward(symbol, setup_name, bars, tier_name, eval_type,
                                                    signals=signals, flatten_sink=flatten_sink)
    net = round(sum(t.pnl for t in trades), 2)
    wins = sum(1 for t in trades if t.pnl > 0)
    win_rate = round(wins / len(trades), 4) if trades else 0.0
    bh = buy_and_hold_pnl(bars, symbol)
    frac, nwin = eval_window_fraction(trades, tier_name, eval_type)
    return BacktestResult(symbol, setup_name, len(trades), net, win_rate, account.failed,
                          blown_date, bh, net > bh, frac, nwin, len(flatten_sink))


# ----------------------------------------------------------------------- CLI


def _render_markdown(results: list[BacktestResult], tier_name: str, eval_type: str) -> str:
    lines = [
        f"# Apex WSGTA backtest ({tier_name}, {eval_type})",
        "",
        "| Symbol | Setup | Trades | Net PnL | Win rate | Eval blown | Buy&Hold PnL | Beats B&H | Eval pass frac | Windows | Flatten events |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.symbol} | {r.setup} | {r.trades} | {r.net_pnl:.2f} | {r.win_rate:.2%} | "
            f"{'YES ' + (r.eval_blown_date or '') if r.eval_blown else 'no'} | {r.buy_hold_pnl:.2f} | "
            f"{'YES' if r.beats_bh else 'no'} | {r.eval_pass_fraction:.2%} | {r.eval_windows} | {r.flatten_events} |"
        )
    return "\n".join(lines) + "\n"


def _print_table(results: list[BacktestResult]) -> None:
    header = (f"{'Symbol':<7}{'Setup':<8}{'Trades':>7}{'NetPnL':>10}{'WinRate':>9}{'Blown':>7}"
             f"{'B&H':>10}{'BeatsBH':>9}{'EvalPass':>10}{'Flatten':>9}")
    print(header)
    for r in results:
        print(f"{r.symbol:<7}{r.setup:<8}{r.trades:>7}{r.net_pnl:>10.2f}{r.win_rate:>9.2%}"
              f"{('Y' if r.eval_blown else 'n'):>7}{r.buy_hold_pnl:>10.2f}"
              f"{('Y' if r.beats_bh else 'n'):>9}{r.eval_pass_fraction:>10.2%}{r.flatten_events:>9}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="apex.backtest")
    ap.add_argument("--symbol", choices=["MNQ", "MES", "all"], default="all")
    ap.add_argument("--setup", choices=[*SETUPS, "all"], default="all")
    ap.add_argument("--tier", default="50K")
    ap.add_argument("--eval-type", default="EOD")
    ap.add_argument("--data", choices=["1h", "5m"], default="1h", help="bar source")
    ap.add_argument("--refresh", action="store_true", help="re-download cached bars (1h only)")
    ap.add_argument("--out-dir", default="results/backtest")
    args = ap.parse_args(argv)

    symbols = ["MNQ", "MES"] if args.symbol == "all" else [args.symbol]
    setup_names = list(SETUPS) if args.setup == "all" else [args.setup]

    results: list[BacktestResult] = []
    for symbol in symbols:
        bars = load_bars(symbol, refresh=args.refresh, data=args.data)
        signals = compute_signals(symbol, bars)  # one indicator pass, shared by every setup
        for setup_name in setup_names:
            results.append(backtest_symbol_setup(symbol, setup_name, bars, args.tier, args.eval_type,
                                                  signals=signals))

    _print_table(results)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    report_path = out_dir / f"report_{args.data}_{args.tier}_{args.eval_type}_{stamp}.md"
    report_path.write_text(_render_markdown(results, args.tier, args.eval_type), encoding="utf-8")
    print(f"\nwrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
