"""
Risk Agent as a Strands LLM agent: gathers weather/news from API + web search (union),
fuses conflicting data, then calls calculate_risk_score_tool.
Exposed to the orchestrator via risk_agent_tool(features_json, risk_queries_json).
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

_CODE_DIR = Path(__file__).resolve().parent.parent
if os.getcwd() != str(_CODE_DIR):
    os.chdir(_CODE_DIR)

RISK_AGENT_PROMPT = """You are a supply chain risk assessment agent. You reason about delivery risk by gathering data from multiple sources and fusing them.

Your tools:
1. **weather_api_tool(city)** – Get weather from OpenWeatherMap API for a city. Returns weather_condition (clear, rainy, cold, hot, foggy, stormy).
2. **news_api_tool(query)** – Search news (NewsAPI) for disruption context. Input a query like "supply chain disruption India" or "port strike Mumbai". Returns list of articles.
3. **web_search_tool(queries)** – Run web search. Input: single query string OR JSON array of query strings (e.g. ["Mumbai weather today", "Mumbai port strike 2025"]). Returns snippets per query.
4. **calculate_risk_score_tool(package_type, weather_condition, risk_factors_json, route_dict_json)** – Compute risk using the early-warning model. weather_condition is your fused canonical weather; risk_factors_json is a JSON array of strings (risk factors you identified); route_dict_json is a JSON object with at least delivery_partner, package_type, vehicle_type, delivery_mode, region, weather_condition, distance_km, package_weight_kg, delivery_rating. This tool returns a deterministic `min_buffer_days` floor from risk level.

Workflow:
1. **Gather from BOTH APIs and web search (union)** – Always call weather_api_tool for origin and destination, news_api_tool for disruption, and web_search_tool with the provided risk_queries list.
2. **Fuse and check consistency** – Compare weather from API vs web search; decide the canonical weather_condition and final list of risk_factors (union of API news + web findings).
3. **Call calculate_risk_score_tool** – Pass package_type, fused weather_condition, risk_factors as JSON array, and a minimal route_dict (include defaults where needed).
4. **Return** – Reply with the risk assessment as JSON: risk_level, risk_score, risk_factors, delay_probability, recommended_buffer_days, buffer_rationale, warnings, alert_required.

Buffer recommendation policy:
- Determine `recommended_buffer_days` by reasoning over:
  - delay_probability,
  - weather at origin/destination,
  - package criticality (pharmacy/groceries are more conservative),
  - route distance (long-haul adds variance),
  - disruption signals from news/web.
- Guidelines:
  - 0 days: delay_prob < 5% and no weather/disruption concerns
  - 1 day: delay_prob 5–15% OR adverse weather OR long route (>500 km)
  - 2 days: delay_prob 15–30% OR severe weather OR active disruption
  - 3 days: delay_prob > 30% OR critical package with any risk signal
  - For critical packages (pharmacy, groceries), add +1 day safety margin.
- Always include `buffer_rationale` as a concise explanation.
- Ensure recommended_buffer_days is at least the tool-provided min_buffer_days.

Validation after all tools return (before final JSON):
- delay_probability must be in [0.0, 1.0]. If outside range, cap it and add
  "validation_flag": "delay_probability_capped".
- recommended_buffer_days must be consistent with risk_level:
  LOW: 0–1, MEDIUM: 1–2, HIGH: >=2, CRITICAL: >=3.
  If HIGH/CRITICAL has 0 buffer days, flag ANOMALY_BUFFER_TOO_LOW.
- weather_condition in your JSON must match weather_api_tool output, not assumptions.
  If weather used web-search fallback, include "weather_source": "web_search_fallback".
- Tool errors must not be silently ignored; include them in risk_factors.

Your final message must be valid JSON so the orchestrator can parse it."""

_risk_agent = None


def _enforce_buffer_floor(obj: dict) -> dict:
    """Ensure final buffer recommendation respects deterministic safety floor."""
    buffer_map = {"CRITICAL": 2, "HIGH": 1, "MEDIUM": 1, "LOW": 0}
    level = str(obj.get("risk_level") or "LOW").upper()
    min_buffer = int(obj.get("min_buffer_days", buffer_map.get(level, 0)))
    llm_buffer = obj.get("recommended_buffer_days", 0)
    try:
        llm_buffer = int(llm_buffer)
    except (TypeError, ValueError):
        llm_buffer = 0
    final_buffer = max(min_buffer, llm_buffer)
    obj["recommended_buffer_days"] = final_buffer
    if "buffer_rationale" not in obj or not str(obj.get("buffer_rationale")).strip():
        obj["buffer_rationale"] = (
            f"Recommended buffer is max(min_buffer_days={min_buffer}, "
            f"llm_recommendation={llm_buffer}) for risk level {level}."
        )
    return obj


def get_risk_agent(region_name: Optional[str] = None, model_id: Optional[str] = None):
    global _risk_agent
    if _risk_agent is not None:
        return _risk_agent
    try:
        from strands import Agent
        from strands.models import BedrockModel
    except ImportError as e:
        raise ImportError("Strands and Bedrock required for Risk Agent. pip install strands-agents boto3") from e
    from tools import get_risk_agent_tools

    region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
    model_id = model_id or os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    bedrock_model = BedrockModel(
        model_id=model_id,
        region_name=region_name,
        temperature=0.0,
    )
    _risk_agent = Agent(
        model=bedrock_model,
        system_prompt=RISK_AGENT_PROMPT,
        tools=get_risk_agent_tools(),
    )
    return _risk_agent


def run_risk_agent(features_json: str, risk_queries_json: str, region_name: Optional[str] = None, model_id: Optional[str] = None) -> str:
    from tools.stream_events import set_stream_scope

    agent = get_risk_agent(region_name=region_name, model_id=model_id)
    try:
        features = json.loads(features_json)
    except json.JSONDecodeError:
        features = {}
    try:
        queries = json.loads(risk_queries_json) if risk_queries_json.strip() else []
    except json.JSONDecodeError:
        queries = [risk_queries_json] if risk_queries_json.strip() else []
    if not isinstance(queries, list):
        queries = [queries]
    origin = str(features.get("origin") or "mumbai").lower()
    destination = str(features.get("destination") or "delhi").lower()
    package_type = (features.get("package_type") or "clothing").lower()
    message = (
        f"Assess risk for this shipment.\n\n"
        f"Features: origin={origin}, destination={destination}, package_type={package_type}. "
        f"Full features: {json.dumps(features)}\n\n"
        f"Use these search queries to gather data (call web_search_tool with this list): {json.dumps(queries)}\n\n"
        f"Call weather_api_tool for '{origin}' and '{destination}', news_api_tool for disruption, "
        f"web_search_tool with the queries above. Then fuse API + web results and call calculate_risk_score_tool. "
        f"Return the risk assessment as JSON."
    )
    set_stream_scope("risk")
    try:
        response = agent(message)
    finally:
        set_stream_scope("")
    text = str(response)
    try:
        obj = json.loads(text)
        return json.dumps(_enforce_buffer_floor(obj))
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if "risk_level" in obj or "risk_factors" in obj:
                return json.dumps(_enforce_buffer_floor(obj))
        except json.JSONDecodeError:
            pass
    return text

