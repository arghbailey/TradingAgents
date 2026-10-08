"""Apex Institutional TradingAgents Automation: one WSGTA decision cycle. DRY-RUN ONLY.

    python run_apex_automation.py --mode dry-run --tier 50K --eval-type EOD --offline --verbose

``--offline`` uses scripted LLMs, fixture bars, a fixture calendar and fixture
news/sentiment: no keys and no network. Without it, the LLM roles are built from
the configured profile (see apex/llm.py and .env.apex.example), and bars come from
``--bars-csv`` because no live futures feed is wired in.

No order is ever sent to a broker. The only execution sink writes the bracket
payload JSON under ``--out``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

ALLOWED_MODES = ("dry-run",)


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Apex WSGTA automation (DRY-RUN ONLY).")
    p.add_argument("--mode", required=True, help="must be 'dry-run'; live modes do not exist")
    p.add_argument("--tier", default="50K", help="Apex tier: 25K, 50K, 100K, 150K")
    p.add_argument("--eval-type", default="EOD", help="EOD or LEGACY")
    p.add_argument("--symbol", default="MNQ", help="MNQ, MES, NQ or ES")
    p.add_argument("--date", default=None, help="trade date YYYY-MM-DD (default: today)")
    p.add_argument("--time", default="10:05", help="ET wall-clock time to evaluate at, HH:MM")
    p.add_argument("--vault", default="vault", help="vault root (default ./vault)")
    p.add_argument("--out", default="results/apex_dryrun", help="where dry-run order JSON goes")
    p.add_argument("--profile", default=None, help="LLM profile: local (Ollama) or cloud")
    p.add_argument("--bars-csv", default=None, help="CSV of bars (ts,open,high,low,close,volume)")
    p.add_argument("--calendar-json", default=None, help="JSON list of economic events")
    p.add_argument("--account-json", default=None,
                   help='JSON with account figures, e.g. {"balance": 50200, "threshold": 47700}')
    p.add_argument("--offline", action="store_true", help="fake LLMs + fixture data; no keys")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _load_account(args, tier, eval_type):
    from apex.risk import AccountState

    acct = AccountState.fresh(tier, eval_type)
    if args.account_json:
        data = json.loads(Path(args.account_json).read_text(encoding="utf-8"))
        for key, value in data.items():
            if key in ("tier", "eval_type") or not hasattr(acct, key):
                raise SystemExit(f"error: unknown or read-only account field {key!r}")
            if key == "last_stop_out" and value:
                value = datetime.fromisoformat(value)
            setattr(acct, key, value)
    return acct


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.mode not in ALLOWED_MODES:
        print(f"error: --mode {args.mode!r} is not allowed. This automation is DRY-RUN ONLY; "
              f"the only accepted mode is 'dry-run'. There is no live or broker mode.", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    from dotenv import load_dotenv

    from apex.calendar import ET, StaticCalendar, at_et, fixture_calendar
    from apex.config import EvaluationType, get_contract, get_tier
    from apex.execution import DryRunDispatcher
    from apex.graph import ApexDeps, run_cycle
    from apex.llm import (
        FixtureNews,
        FixtureSentiment,
        JevSentimentStub,
        LLMSentiment,
        build_role_llms,
        offline_llms,
        resolve_roles,
    )
    from apex.market import CsvMarketData, FixtureMarketData
    from apex.vault import Vault

    load_dotenv(".env.apex")
    try:
        tier = get_tier(args.tier)
        eval_type = EvaluationType.parse(args.eval_type)
        symbol = get_contract(args.symbol).symbol
        day = date.fromisoformat(args.date) if args.date else datetime.now(ET).date()
        hh, mm = (int(x) for x in args.time.split(":"))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    now = at_et(day, hh, mm)

    import os

    if args.offline:
        llms = offline_llms()
        market = FixtureMarketData(day, (hh, mm))
        calendar = StaticCalendar.from_json(args.calendar_json) if args.calendar_json else fixture_calendar(day)
        news, sentiment = FixtureNews(), FixtureSentiment()
    else:
        if not args.bars_csv:
            print("error: no live futures feed is built yet; pass --bars-csv or use --offline",
                  file=sys.stderr)
            return 2
        roles = resolve_roles(args.profile or os.environ.get("APEX_LLM_PROFILE", "local"))
        llms = build_role_llms(roles)
        market = CsvMarketData(args.bars_csv)
        calendar = StaticCalendar.from_json(args.calendar_json) if args.calendar_json else StaticCalendar()
        news = FixtureNews()  # no live news feed yet; see docs "Not yet built"
        sentiment = (LLMSentiment(llms["sentiment"], news) if llms.get("sentiment") is not None
                     else JevSentimentStub())

    deps = ApexDeps(
        account=_load_account(args, tier, eval_type), market=market, calendar=calendar, news=news,
        sentiment=sentiment, llms=llms, vault=Vault(args.vault), dispatcher=DryRunDispatcher(args.out),
        now=now,
    )
    final = run_cycle(deps, symbol, day.isoformat())

    print("=" * 64)
    print(f"APEX WSGTA DRY-RUN | {symbol} | {tier.name} {eval_type.value} | {now.isoformat()}")
    print("=" * 64)
    if args.verbose:
        for m in final.get("messages", []):
            print(f"[{getattr(m, 'name', None) or 'msg'}] {m.content}")
        print("-" * 64)
    summary = {k: final.get(k) for k in (
        "active_setup", "setup_direction", "setup_grade", "setup_confluence", "atr_bracket",
        "news_blackout_active", "sentiment_score", "debate_round", "consensus_rating",
        "effective_buffer", "daily_pnl", "risk_verdict", "risk_reasons", "order_symbol",
        "order_action", "order_contracts", "terminal_reason", "dispatch_path")}
    print(json.dumps(summary, indent=2, default=str))
    if final.get("order_payload"):
        print("-" * 64)
        print("Bracket payload (DRY-RUN, not sent to any broker):")
        print(json.dumps(final["order_payload"], indent=2, default=str))
    print("-" * 64)
    print(f"Vault: {json.dumps(final.get('vault_result'), default=str)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
