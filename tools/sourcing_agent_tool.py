from __future__ import annotations

from tools._common import tool


@tool
def sourcing_agent_tool(features_json: str, risk_assessment_json: str, sourcing_queries_json: str) -> str:
    from agents.sourcing import run_sourcing_agent

    return run_sourcing_agent(features_json, risk_assessment_json, sourcing_queries_json)

