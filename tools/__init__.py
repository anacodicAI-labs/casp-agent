"""
LLM tool surface (Strands `@tool`).

Industry-standard intent:
- `tools/` contains `@tool` wrappers that an LLM agent can decide to call.
- Internal helper/plumbing should be imported from `utils/` (non-LLM layer).

This `__init__.py` intentionally keeps imports lazy to avoid circular imports
between `tools/` and `utils/`.
"""

from __future__ import annotations


def get_risk_agent_tools():
    # Risk Agent tools: decision points for LLM (atomic calls)
    from tools.weather_api_tool import weather_api_tool
    from tools.news_api_tool import news_api_tool
    from tools.web_search_tool import web_search_tool
    from tools.calculate_risk_score_tool import calculate_risk_score_tool

    return [weather_api_tool, news_api_tool, web_search_tool, calculate_risk_score_tool]


def get_sourcing_agent_tools():
    # Sourcing Agent tools: decision points for LLM (atomic calls)
    from tools.distance_api_tool import distance_api_tool
    from tools.routes_lookup_tool import routes_lookup_tool
    from tools.web_search_tool import web_search_tool
    from tools.get_carrier_options_tool import get_carrier_options_tool

    return [distance_api_tool, routes_lookup_tool, web_search_tool, get_carrier_options_tool]


def get_gather_tools():
    # Phase 1 (gather) tools only: extract features + risk + sourcing — NO optimization/carbon.
    # Keeps human-in-the-loop review meaningful before Phase 2 runs.
    from tools.extract_features_tool import extract_features_tool
    from tools.risk_agent_tool import risk_agent_tool
    from tools.sourcing_agent_tool import sourcing_agent_tool

    return [extract_features_tool, risk_agent_tool, sourcing_agent_tool]


def get_optimize_tools():
    # Phase 2 tools only: deterministic optimization + carbon analysis
    from tools.run_optimization_tool import run_optimization_tool
    from tools.carbon_analysis_tool import carbon_analysis_tool

    return [run_optimization_tool, carbon_analysis_tool]


def get_all_tools():
    # Orchestrator agent tools: decision points for LLM (atomic calls)
    from tools.extract_features_tool import extract_features_tool
    from tools.risk_agent_tool import risk_agent_tool
    from tools.sourcing_agent_tool import sourcing_agent_tool
    from tools.run_optimization_tool import run_optimization_tool
    from tools.carbon_analysis_tool import carbon_analysis_tool

    return [extract_features_tool, risk_agent_tool, sourcing_agent_tool, run_optimization_tool, carbon_analysis_tool]


__all__ = ["get_all_tools", "get_gather_tools", "get_optimize_tools", "get_risk_agent_tools", "get_sourcing_agent_tools"]

