"""
Sourcing Agent as a Strands LLM agent: uses distance, routes, web search, carrier options;
reasons about best carrier given risk.
Exposed to the orchestrator via sourcing_agent_tool(features_json, risk_assessment_json, sourcing_queries_json).
Returns carrier_options JSON for run_optimization_tool.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

_CODE_DIR = Path(__file__).resolve().parent.parent
if os.getcwd() != str(_CODE_DIR):
    os.chdir(_CODE_DIR)

SOURCING_AGENT_PROMPT = """You are a supply chain sourcing agent. You reason about the best carrier for a shipment given risk and constraints.

Your tools:
1. **distance_api_tool(origin, destination)** – Get road distance in km (OpenRouteService API). Returns distance_km and source.
2. **routes_lookup_tool(origin, destination)** – Get route info from local routes or cascade (distance_km, region, is_metro_to_metro).
3. **web_search_tool(queries)** – Run web search. Input: single query string OR JSON array of query strings (e.g. pricing queries). Returns snippets per query.
4. **get_carrier_options_tool(features_json, risk_assessment_json)** – Get carrier options (cost, on_time_pct, carbon, route) for the given features and risk. Returns list of options.

Workflow:
1. Call distance_api_tool and routes_lookup_tool for origin and destination to get distance/route context.
2. Call web_search_tool with the provided sourcing_queries (e.g. carrier rates, cold chain pricing) to gather external context.
3. Call get_carrier_options_tool(features_json, risk_assessment_json) to get the full list of carrier options.
4. Reason about which carrier best fits the shipment given risk_level, package_type (e.g. pharmacy needs high SLA), and cost/carbon trade-offs.
5. **Get industry benchmark**: Use web_search_tool to search for typical market rates for this shipment. Search: "delivery cost {package_type} {distance}km {weight}kg India". Use the results to estimate the industry benchmark cost (INR). Compare your recommended cost to this benchmark and set efficiency ("Below benchmark" or "Above benchmark") and efficiency_percentage (e.g. "15% below benchmark").
6. **Return** – Your final message MUST end with a valid JSON block (no extra text after it) containing:
  - "carrier_options": the full list of options you received from get_carrier_options_tool (do not modify it)
  - "recommendation": your short text recommendation (e.g. "Recommend BlueDart EV Van: 99% on-time, meets pharmacy SLA")
  - "industry_benchmark": estimated market rate in INR from web search (number or null)
  - "efficiency": "Below benchmark" or "Above benchmark" (if benchmark available)
  - "efficiency_percentage": e.g. "15% below benchmark" (if benchmark available)
  - "benchmark_reasoning": brief explanation (1-2 sentences) of how the benchmark was derived and why the recommendation compares so (must be present even if industry_benchmark is null; then explain fallback).

Example final block:
```json
{"carrier_options": [...], "recommendation": "Recommend BlueDart: 99% on-time, meets pharmacy SLA", "industry_benchmark": 2100, "efficiency": "Below benchmark", "efficiency_percentage": "12% below benchmark"}
```

Validation after get_carrier_options_tool (before final JSON):
- Vehicle vs distance feasibility:
  bike/ev bike/scooter are appropriate for <80 km.
  If distance_km > 200 and vehicle is bike/scooter/ev bike, flag "vehicle_range_mismatch" and prefer van/truck in recommendation.
  ev van: 80–400 km. van/truck: any distance.
- Cost plausibility:
  bike short-haul <₹500,
  van 1400km ~₹1500–2500,
  truck 1400km ~₹3000–6000.
  Cost <₹100 for 1400km => ANOMALY_COST_TOO_LOW.
- on_time_pct must be within 0–100 for every carrier option; exclude out-of-range options.
- Add "validation_summary": "ok" or list of flags in final JSON.

The orchestrator will parse carrier_options from this block to pass to the next step."""

_sourcing_agent = None


def get_sourcing_agent(region_name: Optional[str] = None, model_id: Optional[str] = None):
    global _sourcing_agent
    if _sourcing_agent is not None:
        return _sourcing_agent
    try:
        from strands import Agent
        from strands.models import BedrockModel
    except ImportError as e:
        raise ImportError("Strands and Bedrock required for Sourcing Agent. pip install strands-agents boto3") from e
    from tools import get_sourcing_agent_tools

    region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
    model_id = model_id or os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    bedrock_model = BedrockModel(
        model_id=model_id,
        region_name=region_name,
        temperature=0.0,
    )
    _sourcing_agent = Agent(
        model=bedrock_model,
        system_prompt=SOURCING_AGENT_PROMPT,
        tools=get_sourcing_agent_tools(),
    )
    return _sourcing_agent


def _extract_carrier_options_from_response(response: str) -> Optional[list]:
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response)
    if m:
        try:
            data = json.loads(m.group(1))
            if "carrier_options" in data:
                return data["carrier_options"]
        except json.JSONDecodeError:
            pass
    matches = list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response))
    for m in reversed(matches):
        try:
            data = json.loads(m.group(0))
            if "carrier_options" in data and isinstance(data["carrier_options"], list):
                return data["carrier_options"]
        except json.JSONDecodeError:
            continue
    return None


def _extract_full_sourcing_response_from_response(response: str) -> Optional[dict]:
    """
    Extract the final JSON object from an LLM response.

    Expected keys include at least: carrier_options (list).
    For older/partial outputs, we still try to return a dict.
    """
    # Prefer fenced JSON object blocks.
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response)
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, dict) and "carrier_options" in data:
                return data
        except json.JSONDecodeError:
            pass

    # Fallback: find the last JSON-ish object containing carrier_options.
    matches = list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response))
    for m in reversed(matches):
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict) and "carrier_options" in data:
                return data
        except json.JSONDecodeError:
            continue

    # Backward compatibility: if we only got carrier_options, wrap it.
    options = _extract_carrier_options_from_response(response)
    if options is not None:
        return {"carrier_options": options}
    return None


def run_sourcing_agent(
    features_json: str,
    risk_assessment_json: str,
    sourcing_queries_json: str,
    region_name: Optional[str] = None,
    model_id: Optional[str] = None,
) -> str:
    from tools.stream_events import set_stream_scope

    agent = get_sourcing_agent(region_name=region_name, model_id=model_id)
    try:
        queries = json.loads(sourcing_queries_json) if sourcing_queries_json.strip() else []
    except json.JSONDecodeError:
        queries = [sourcing_queries_json] if sourcing_queries_json.strip() else []
    if not isinstance(queries, list):
        queries = [queries]
    message = (
        f"Get carrier options and recommend the best carrier for this shipment.\n\n"
        f"Features: {features_json}\n\n"
        f"Risk assessment: {risk_assessment_json}\n\n"
        f"Use these search queries for pricing/context (call web_search_tool with this list): {json.dumps(queries)}\n\n"
        f"Call distance_api_tool and routes_lookup_tool for origin/destination, web_search_tool with the queries, "
        f"then get_carrier_options_tool(features_json, risk_assessment_json). "
        f"Reason about the best carrier given risk and end with a JSON block containing carrier_options, recommendation, and benchmark fields."
    )
    set_stream_scope("sourcing")
    try:
        response = agent(message)
    finally:
        set_stream_scope("")
    full = _extract_full_sourcing_response_from_response(str(response))
    if full is not None:
        return json.dumps(full)
    # Fallback: Python sourcing logic (no LLM JSON block)
    from agents._instance import get_orchestrator
    from utils.extraction_tools import extract_features_from_dict

    orch = get_orchestrator()
    features = json.loads(features_json)
    risk = json.loads(risk_assessment_json) if risk_assessment_json else {}
    features = extract_features_from_dict(features)
    options = orch.sourcing_service.get_carrier_options(features, risk)
    out = []
    for o in options:
        out.append(
            {
                "carrier": o.get("carrier"),
                "carrier_code": o.get("carrier_code"),
                "predicted_cost": o.get("predicted_cost"),
                "predicted_on_time_pct": o.get("predicted_on_time_pct"),
                "total_carbon_gco2": o.get("total_carbon_gco2"),
                "meets_sla": o.get("meets_sla"),
                "route": o.get("route"),
            }
        )
    # No LLM output block was recoverable; return a partial block for compatibility.
    return json.dumps(
        {
            "carrier_options": out,
            "recommendation": None,
            "industry_benchmark": None,
            "efficiency": None,
            "efficiency_percentage": None,
            "benchmark_reasoning": None,
        }
    )

