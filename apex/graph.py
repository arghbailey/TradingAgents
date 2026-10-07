"""The WSGTA LangGraph.

    START -> vault_preflight_reader
          -> [wsgta_technicals | macro_news_gate | social_sentiment]   (parallel)
          -> gatekeeper_router
               news blackout / daily halt -> circuit_breaker_halt
               no setup / ADX < 20        -> terminal_no_trade
               otherwise                  -> bull_researcher
    bull_researcher -> bear_researcher -> research_manager
          actionable rating  -> apex_risk_governor
          debate_round < 2   -> bull_researcher
          otherwise          -> terminal_no_trade
    apex_risk_governor
          APPROVED                          -> portfolio_manager
          RESIZE_MICRO (under the loop cap) -> contract_resizer -> apex_risk_governor
          otherwise                         -> terminal_no_trade
    portfolio_manager -> execution_dispatcher (DRY-RUN) -> vault_post_trade_reflector -> END
    terminal_no_trade / circuit_breaker_halt            -> vault_post_trade_reflector -> END

Every gate is decided by code. LLMs write narrative (technical read, macro summary,
bull/bear arguments, risk and PM commentary) and propose a rating. The rating is
parsed strictly, and it must agree with the deterministic setup direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from .calendar import EconomicCalendar
from .config import (
    DEFAULT_RISK_PARAMS,
    BracketParams,
    EvaluationType,
    RiskParams,
    SetupParams,
    get_contract,
)
from .execution import DryRunDispatcher, build_bracket
from .llm import NewsProvider, SentimentProvider, clamp_sentiment, parse_rating, text_of
from .market import MarketDataProvider
from .risk import (
    AccountState,
    ApexRiskGovernor,
    MarketContext,
    OrderProposal,
    Verdict,
    active_news_lockout,
    resize_to_micro,
)
from .setups import detect_setup
from .state import ACTIONABLE_RATINGS, WSGTAState
from .vault import Vault

NODES = (
    "vault_preflight_reader", "wsgta_technicals", "macro_news_gate", "social_sentiment",
    "gatekeeper_router", "bull_researcher", "bear_researcher", "research_manager",
    "apex_risk_governor", "contract_resizer", "portfolio_manager", "execution_dispatcher",
    "terminal_no_trade", "circuit_breaker_halt", "vault_post_trade_reflector",
)


@dataclass
class ApexDeps:
    """Everything the graph touches, injected so tests can swap any piece."""

    account: AccountState
    market: MarketDataProvider
    calendar: EconomicCalendar
    news: NewsProvider
    sentiment: SentimentProvider
    llms: dict[str, Any]
    vault: Vault
    dispatcher: DryRunDispatcher
    now: datetime
    governor: Any = None
    risk_params: RiskParams = field(default_factory=RiskParams)
    bracket_params: BracketParams = field(default_factory=BracketParams)
    setup_params: SetupParams = field(default_factory=SetupParams)
    requested_contracts: int = 2

    def __post_init__(self):
        if self.governor is None:
            self.governor = ApexRiskGovernor(self.risk_params)
        if getattr(self.dispatcher, "mode", None) != "dry-run":
            raise ValueError("only a dry-run dispatcher is allowed")


def _say(deps: ApexDeps, role: str, prompt: str) -> AIMessage | None:
    llm = deps.llms.get(role)
    if llm is None:
        return None
    return AIMessage(content=text_of(llm.invoke(prompt)), name=role)


def _msgs(*m) -> list:
    return [x for x in m if x is not None]


# ------------------------------------------------------------------ routing (pure)


def route_gatekeeper(state: WSGTAState, params: RiskParams = DEFAULT_RISK_PARAMS) -> str:
    if state.get("news_blackout_active") or state.get("daily_loss_halt"):
        return "circuit_breaker_halt"
    adx = (state.get("setup_confluence") or {}).get("adx", 0.0)
    if not state.get("active_setup") or adx < params.min_adx:
        return "terminal_no_trade"
    return "bull_researcher"


def route_research(state: WSGTAState, params: RiskParams = DEFAULT_RISK_PARAMS) -> str:
    if state.get("consensus_rating") in ACTIONABLE_RATINGS:
        return "apex_risk_governor"
    if state.get("debate_round", 0) < params.max_debate_rounds:
        return "bull_researcher"
    return "terminal_no_trade"


def route_governor(state: WSGTAState, params: RiskParams = DEFAULT_RISK_PARAMS) -> str:
    verdict = state.get("risk_verdict")
    if verdict == Verdict.APPROVED.value:
        return "portfolio_manager"
    if verdict == Verdict.RESIZE_MICRO.value and state.get("resize_iterations", 0) < params.max_resize_iterations:
        return "contract_resizer"
    return "terminal_no_trade"


# ------------------------------------------------------------------------- graph


def build_wsgta_graph(deps: ApexDeps):
    rp = deps.risk_params

    def vault_preflight_reader(state: WSGTAState) -> dict:
        acct = deps.account
        ctx = deps.vault.preflight(state["symbol"], acct.tier.name)
        halt = (acct.session_locked or acct.failed
                or acct.daily_pnl <= -acct.tier.daily_loss_limit)
        return {
            "vault_context": ctx,
            "now": deps.now.isoformat(),
            "eval_type": acct.eval_type.value,
            "effective_buffer": round(acct.effective_buffer, 2),
            "daily_pnl": round(acct.daily_pnl, 2),
            "daily_loss_halt": bool(halt),
            "consistency_cap_breached": acct.daily_pnl >= acct.tier.max_single_day_profit,
            "debate_round": 0, "debate_history": [], "resize_iterations": 0,
            "order_symbol": state["symbol"], "order_requested_contracts": deps.requested_contracts,
        }

    def wsgta_technicals(state: WSGTAState) -> dict:
        symbol = state["symbol"]
        bars = deps.market.bars(symbol, deps.now)
        sig = detect_setup(bars, symbol, deps.setup_params)
        conf = sig.confluence
        bp = deps.bracket_params
        stop = bp.stop_points[symbol]
        if bp.atr_stop_multiple and conf.get("atr"):
            stop = max(stop, bp.atr_stop_multiple * conf["atr"])
        msg = _say(deps, "technical",
                   f"Summarize the {symbol} technical picture for a futures day trader. "
                   f"Setup={sig.setup} {sig.direction} grade={sig.grade}; indicators={conf}. "
                   "Narrative only; do not propose sizing.")
        return {
            "active_setup": sig.setup, "setup_direction": sig.direction, "setup_grade": sig.grade,
            "setup_reasons": sig.reasons,
            "setup_confluence": {k: conf.get(k) for k in
                                 ("vwap", "ema21", "ema30", "ema65", "ema200", "adx", "rvol", "rsi", "close")},
            "atr_bracket": {"atr": conf.get("atr"), "stop_points": stop,
                            "c1_target_points": bp.c1_target_points[symbol]},
            "messages": _msgs(msg),
        }

    def macro_news_gate(state: WSGTAState) -> dict:
        events = deps.calendar.events_for(deps.now.date())
        hit = active_news_lockout(deps.now, events, rp)
        heads = deps.news.headlines(state["symbol"], state["trade_date"])
        msg = _say(deps, "macro_news",
                   f"Macro briefing for {state['symbol']} on {state['trade_date']}. "
                   f"Scheduled: {[e.to_dict() for e in events]}. Headlines: {heads}. Narrative only.")
        return {
            "news_blackout_active": hit is not None,
            "blackout_event": hit.name if hit else None,
            "macro_summary": msg.content if msg else "; ".join(heads),
            "messages": _msgs(msg),
        }

    def social_sentiment(state: WSGTAState) -> dict:
        try:
            score = clamp_sentiment(deps.sentiment.score(state["symbol"], state["trade_date"]))
        except NotImplementedError:
            score = 0.0
        return {"sentiment_score": score}

    def gatekeeper_router(state: WSGTAState) -> dict:
        return {}

    def _debate_prompt(state, side: str) -> str:
        return (f"You are the {side} researcher for {state['symbol']} futures. Setup "
                f"{state.get('active_setup')} {state.get('setup_direction')} "
                f"(grade {state.get('setup_grade')}), confluence {state.get('setup_confluence')}, "
                f"sentiment {state.get('sentiment_score')}, macro: {state.get('macro_summary')}. "
                f"Debate so far: {state.get('debate_history')}. Argue the {side} case briefly.")

    def bull_researcher(state: WSGTAState) -> dict:
        msg = _say(deps, "bull", _debate_prompt(state, "bull"))
        text = msg.content if msg else "(no bull model configured)"
        return {"debate_history": [*state.get("debate_history", []), f"BULL: {text}"],
                "messages": _msgs(msg)}

    def bear_researcher(state: WSGTAState) -> dict:
        msg = _say(deps, "bear", _debate_prompt(state, "bear"))
        text = msg.content if msg else "(no bear model configured)"
        return {"debate_history": [*state.get("debate_history", []), f"BEAR: {text}"],
                "messages": _msgs(msg)}

    def research_manager(state: WSGTAState) -> dict:
        msg = _say(deps, "research_manager",
                   "Judge this debate and end with one line 'RATING: <STRONG_BUY|BUY|NEUTRAL|SELL|"
                   "STRONG_SELL>'.\n" + "\n".join(state.get("debate_history", [])))
        rating = parse_rating(msg.content if msg else "")
        direction = state.get("setup_direction")
        # Deterministic guard: the rating may not contradict the detected setup.
        if (rating in ("BUY", "STRONG_BUY") and direction != "LONG") or \
                (rating in ("SELL", "STRONG_SELL") and direction != "SHORT"):
            rating = "NEUTRAL"
        return {"consensus_rating": rating, "debate_round": state.get("debate_round", 0) + 1,
                "messages": _msgs(msg)}

    def apex_risk_governor(state: WSGTAState) -> dict:
        rating = state.get("consensus_rating")
        order = OrderProposal(
            symbol=state.get("order_symbol") or state["symbol"],
            direction="LONG" if rating in ("BUY", "STRONG_BUY") else "SHORT",
            contracts=int(state.get("order_requested_contracts") or deps.requested_contracts),
            entry=float((state.get("setup_confluence") or {}).get("ema21") or 0.0),
            stop_points=float(state["atr_bracket"]["stop_points"]),
            setup_grade=state.get("setup_grade", "B"),
        )
        quote = deps.market.quote(order.symbol, deps.now)
        mctx = MarketContext(deps.now, quote.spread_ticks, tuple(deps.calendar.events_for(deps.now.date())))
        decision = deps.governor.evaluate(deps.account, mctx, order)
        verdict = decision.verdict.value if isinstance(decision.verdict, Verdict) else str(decision.verdict)
        # Narrative only: the model sees the verdict and cannot change it.
        msg = _say(deps, "risk_narrative",
                   f"Explain this risk decision for the trader in two sentences: {verdict}; "
                   f"reasons: {decision.reasons}.")
        return {"risk_verdict": verdict, "risk_reasons": list(decision.reasons),
                "order_contracts": decision.contracts, "messages": _msgs(msg)}

    def contract_resizer(state: WSGTAState) -> dict:
        order = OrderProposal(state.get("order_symbol") or state["symbol"], "LONG",
                              int(state.get("order_requested_contracts") or deps.requested_contracts),
                              0.0, 1.0)
        micro = resize_to_micro(order)
        return {"order_symbol": micro.symbol, "order_requested_contracts": micro.contracts,
                "resize_iterations": state.get("resize_iterations", 0) + 1}

    def portfolio_manager(state: WSGTAState) -> dict:
        rating = state["consensus_rating"]
        direction = "LONG" if rating in ("BUY", "STRONG_BUY") else "SHORT"
        symbol = state.get("order_symbol") or state["symbol"]
        payload = build_bracket(symbol, direction, int(state["order_contracts"]),
                                float(state["setup_confluence"]["ema21"]),
                                float(state["atr_bracket"]["stop_points"]),
                                EvaluationType.parse(state["eval_type"]), deps.bracket_params)
        payload.update(setup=state.get("active_setup"), rating=rating, trade_date=state["trade_date"],
                       governor_reasons=state.get("risk_reasons", []))
        msg = _say(deps, "portfolio_manager",
                   f"Comment on this approved dry-run bracket (you cannot change it): {payload}")
        return {"order_action": payload["side"], "order_payload": payload, "messages": _msgs(msg)}

    def execution_dispatcher(state: WSGTAState) -> dict:
        path = deps.dispatcher.dispatch(state["order_payload"])
        return {"dispatch_path": str(path)}

    def terminal_no_trade(state: WSGTAState) -> dict:
        if state.get("risk_verdict"):
            why = f"risk governor: {state['risk_verdict']}: {'; '.join(state.get('risk_reasons') or [])}"
            if state["risk_verdict"] == Verdict.RESIZE_MICRO.value:
                why = f"resize loop cap ({rp.max_resize_iterations}) reached; " + why
        elif state.get("consensus_rating") and state.get("active_setup"):
            why = f"no actionable consensus after {state.get('debate_round', 0)} debate round(s)"
        elif not state.get("active_setup"):
            why = "no WSGTA setup: " + "; ".join(state.get("setup_reasons") or [])
        else:
            why = f"ADX {state.get('setup_confluence', {}).get('adx')} below {rp.min_adx:g}"
        return {"order_action": "NO_TRADE", "order_payload": None, "order_contracts": 0,
                "terminal_reason": why}

    def circuit_breaker_halt(state: WSGTAState) -> dict:
        if state.get("news_blackout_active"):
            why = f"news blackout ({state.get('blackout_event')}): flatten, no new orders"
        else:
            why = "daily loss halt / session lock: flatten, locked until next session"
            deps.account.session_locked = True
        return {"order_action": "HALT_FLATTEN", "order_payload": None, "order_contracts": 0,
                "terminal_reason": why}

    def vault_post_trade_reflector(state: WSGTAState) -> dict:
        record = {k: state.get(k) for k in (
            "symbol", "trade_date", "now", "active_setup", "setup_direction", "setup_grade",
            "setup_confluence", "atr_bracket", "news_blackout_active", "blackout_event",
            "sentiment_score", "debate_round", "consensus_rating", "eval_type", "effective_buffer",
            "daily_pnl", "daily_loss_halt", "consistency_cap_breached", "risk_verdict",
            "risk_reasons", "order_symbol", "order_action", "order_contracts", "order_payload",
            "dispatch_path", "terminal_reason")}
        record["debate_history"] = state.get("debate_history", [])
        record["account"] = deps.account.to_dict()
        record["mode"] = "dry-run"
        return {"vault_result": deps.vault.reflect(record)}

    g = StateGraph(WSGTAState)
    for name, fn in [
        ("vault_preflight_reader", vault_preflight_reader), ("wsgta_technicals", wsgta_technicals),
        ("macro_news_gate", macro_news_gate), ("social_sentiment", social_sentiment),
        ("gatekeeper_router", gatekeeper_router), ("bull_researcher", bull_researcher),
        ("bear_researcher", bear_researcher), ("research_manager", research_manager),
        ("apex_risk_governor", apex_risk_governor), ("contract_resizer", contract_resizer),
        ("portfolio_manager", portfolio_manager), ("execution_dispatcher", execution_dispatcher),
        ("terminal_no_trade", terminal_no_trade), ("circuit_breaker_halt", circuit_breaker_halt),
        ("vault_post_trade_reflector", vault_post_trade_reflector),
    ]:
        g.add_node(name, fn)

    fan_out = ["wsgta_technicals", "macro_news_gate", "social_sentiment"]
    g.add_edge(START, "vault_preflight_reader")
    for n in fan_out:
        g.add_edge("vault_preflight_reader", n)
    g.add_edge(fan_out, "gatekeeper_router")
    g.add_conditional_edges("gatekeeper_router", lambda s: route_gatekeeper(s, rp),
                            ["circuit_breaker_halt", "terminal_no_trade", "bull_researcher"])
    g.add_edge("bull_researcher", "bear_researcher")
    g.add_edge("bear_researcher", "research_manager")
    g.add_conditional_edges("research_manager", lambda s: route_research(s, rp),
                            ["apex_risk_governor", "bull_researcher", "terminal_no_trade"])
    g.add_conditional_edges("apex_risk_governor", lambda s: route_governor(s, rp),
                            ["portfolio_manager", "contract_resizer", "terminal_no_trade"])
    g.add_edge("contract_resizer", "apex_risk_governor")
    g.add_edge("portfolio_manager", "execution_dispatcher")
    g.add_edge("execution_dispatcher", "vault_post_trade_reflector")
    g.add_edge("terminal_no_trade", "vault_post_trade_reflector")
    g.add_edge("circuit_breaker_halt", "vault_post_trade_reflector")
    g.add_edge("vault_post_trade_reflector", END)
    return g.compile()


def run_cycle(deps: ApexDeps, symbol: str, trade_date: str, recursion_limit: int = 60) -> WSGTAState:
    get_contract(symbol)  # validate early
    graph = build_wsgta_graph(deps)
    return graph.invoke({"symbol": symbol.upper(), "trade_date": trade_date, "messages": []},
                        {"recursion_limit": recursion_limit})
