import os
import sys
from datetime import datetime
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph

def main():
    ticker = "AAPL"
    # Use recent trading date
    trade_date = "2026-09-23"

    print(f"=== Starting TradingAgents Analysis for {ticker} (Date: {trade_date}) ===")

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "ollama"
    # Using local Ollama models
    config["deep_think_llm"] = "qwen2.5-coder:14b"
    config["quick_think_llm"] = "qwen2.5-coder:7b"
    config["backend_url"] = "http://localhost:11434/v1"
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    config["results_dir"] = os.path.abspath("./results")

    print(f"Provider: {config['llm_provider']}")
    print(f"Deep thinking model: {config['deep_think_llm']}")
    print(f"Quick thinking model: {config['quick_think_llm']}")
    print("Initializing agents graph...")

    # We select market and fundamentals analysts for a reliable initial run
    ta = TradingAgentsGraph(
        selected_analysts=("market", "fundamentals"),
        debug=True,
        config=config,
    )

    print("\nRunning multi-agent analysis pipeline...")
    final_state, decision = ta.propagate(ticker, trade_date)

    print("\n" + "=" * 50)
    print(f"=== FINAL RECOMMENDATION FOR {ticker} ===")
    print("=" * 50)
    print(f"Decision Signal: {decision}\n")

    if final_state.get("final_trade_decision"):
        print("--- Final Trade Decision ---")
        print(final_state["final_trade_decision"])
        print()

    if final_state.get("trader_investment_plan"):
        print("--- Trader Investment Plan ---")
        print(final_state["trader_investment_plan"])
        print()

    # Save reports
    report_paths = ta.save_reports(final_state, ticker)
    print(f"\nFull report written to: {report_paths}")

if __name__ == "__main__":
    main()
