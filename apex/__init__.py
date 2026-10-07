"""Apex Institutional TradingAgents Automation (DRY-RUN ONLY).

A futures day-trading layer built on top of TradingAgents:

* ``apex.config``   - Apex account tiers, contract specs, risk parameters.
* ``apex.risk``     - the deterministic Apex Risk Governor (code decides, LLMs only propose).
* ``apex.market``   - bars, indicators and an offline fixture feed.
* ``apex.setups``   - deterministic WSGTA setup detection (RMA, FFMA, TREND, MOMO, DB/DT).
* ``apex.calendar`` - injectable high-impact economic calendar.
* ``apex.llm``      - role -> (provider, model) mapping built on TradingAgents' LLM clients.
* ``apex.vault``    - Obsidian / Karpathy-wiki memory vault.
* ``apex.state`` / ``apex.graph`` - the LangGraph "WSGTA" StateGraph.
* ``apex.execution`` - the dry-run execution sink. There is no broker integration.
"""

DRY_RUN_ONLY = True

__all__ = ["DRY_RUN_ONLY"]
