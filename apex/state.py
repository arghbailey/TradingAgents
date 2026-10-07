"""WSGTAState: the state carried through the WSGTA LangGraph."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages

SetupName = Literal["RMA", "FFMA", "TREND", "MOMO", "DB_DT"]
Rating = Literal["STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"]
ACTIONABLE_RATINGS = ("STRONG_BUY", "BUY", "SELL", "STRONG_SELL")


class SetupConfluence(TypedDict, total=False):
    vwap: float
    ema21: float
    ema30: float
    ema65: float
    ema200: float
    adx: float
    rvol: float


class WSGTAState(TypedDict, total=False):
    # identity
    symbol: str
    trade_date: str
    now: str                         # ISO timestamp (ET) the cycle is evaluated at
    messages: Annotated[list, add_messages]

    # technicals (deterministic)
    active_setup: SetupName | None
    setup_direction: str | None
    setup_grade: str
    setup_reasons: list[str]
    setup_confluence: SetupConfluence
    atr_bracket: dict                # {"atr": x, "stop_points": y, "c1_target_points": z}

    # macro / sentiment
    news_blackout_active: bool
    blackout_event: str | None
    macro_summary: str
    sentiment_score: float           # clamped to [-1, 1]

    # debate
    debate_history: list[str]
    debate_round: int
    consensus_rating: Rating

    # account / risk (deterministic)
    eval_type: str
    effective_buffer: float
    daily_pnl: float
    daily_loss_halt: bool
    consistency_cap_breached: bool
    risk_verdict: str | None
    risk_reasons: list[str]
    resize_iterations: int

    # order
    order_symbol: str
    order_requested_contracts: int
    order_action: str | None         # BUY / SELL / NO_TRADE / HALT_FLATTEN
    order_contracts: int
    order_payload: dict | None
    dispatch_path: str | None

    # memory
    vault_context: dict
    terminal_reason: str | None
    vault_result: dict
