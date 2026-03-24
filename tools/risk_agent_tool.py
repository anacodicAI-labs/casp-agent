from __future__ import annotations

from tools._common import tool


@tool
def risk_agent_tool(features_json: str, risk_queries_json: str) -> str:
    from agents.risk import run_risk_agent

    return run_risk_agent(features_json, risk_queries_json)

