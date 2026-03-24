"""
Strands orchestrator agent: Bedrock LLM decides which tools to call and in what order.
Coordinates risk, sourcing, optimization, and carbon analysis.
"""

import os
import queue
from pathlib import Path
from typing import Any, Optional

_CODE_DIR = Path(__file__).resolve().parent.parent
if os.getcwd() != str(_CODE_DIR):
    os.chdir(_CODE_DIR)

ORCHESTRATOR_PROMPT = """You are a supply chain optimization orchestrator that coordinates specialized agents and tools. You read and understand the user query and decide which tools or agents to call.

Your tools:
1. **extract_features_tool(query)** – Extract shipment features from natural language (e.g. "insulin Mumbai to Delhi 150km"). Returns features + defaults_used.
2. **risk_agent_tool(features_json, risk_queries_json)** – Assess delivery risk via the Risk Agent (LLM). It gathers weather/news from API + web search (union), fuses conflicting data, and returns risk_level, risk_factors, delay_probability, recommended_buffer_days. You MUST pass risk_queries_json: a JSON array of adaptive search queries. Generate these from the user request and features (origin, destination, package_type). Examples: ["Mumbai weather today", "Delhi weather today", "Mumbai port strike 2025", "India logistics disruption", "Mumbai port status today"].
3. **sourcing_agent_tool(features_json, risk_assessment_json, sourcing_queries_json)** – Get carrier options via the Sourcing Agent (LLM). Returns either:
   - a JSON list of carrier options (legacy), or
   - a JSON object containing `carrier_options` plus `recommendation`, `industry_benchmark`, `efficiency`, `efficiency_percentage`, and `benchmark_reasoning` (new).
   You MUST pass sourcing_queries_json: a JSON array of adaptive search queries (e.g. ["Delhivery rates Mumbai Delhi 2025", "BlueDart pharmacy shipping rates", "cold chain logistics Mumbai Delhi cost"]).
4. **run_optimization_tool(features_json, carrier_options_json)** – Run route optimization. Use after sourcing_agent_tool. carrier_options_json is the return value of sourcing_agent_tool (list or dict). Returns best route, cost, on-time, carbon, governance (and passes through LLM benchmark fields if present).
5. **carbon_analysis_tool(carrier_options_json, package_type, optimization_result_json)** – Carbon and governance analysis (Python tool, no LLM). Use carrier options and optimization result.

Guidelines:
- Coordinate the pipeline by calling tools in the order that makes sense for the query and dependencies.
- Ensure all required context is gathered before optimization.
- For risk_agent_tool, generate 4–6 adaptive risk queries (weather, disruptions, port/lane events) from origin, destination, and package_type.
- For sourcing_agent_tool, generate 2–4 adaptive pricing/sourcing queries.
- Always summarize the outcome: recommended carrier, cost, on-time probability, carbon, risk level, and any governance notes.
- Package types are classified semantically (e.g. insulin → pharmacy). Use extracted features as-is unless the user asks to change them.

Validation after extract_features_tool:
- Is distance_km plausible? Mumbai→Delhi ~1400km, Mumbai→Pune ~150km, Delhi→Kolkata ~1500km, Chennai→Bangalore ~350km.
  If a cross-state route shows distance_km < 100, flag as ANOMALY_DISTANCE.
- Is package_type semantically correct? "insulin"→pharmacy, "shirt"→clothing.
  If defaults_used contains package_type but query has a specific product, flag as ANOMALY_PACKAGE_TYPE.
- Is package_weight_kg plausible for package_type?
- Add anomalies to your final summary under "validation_flags". Do not stop pipeline; flag and continue.

Final pipeline sanity check after all tools:
- CASP range check: pharmacy ~6.67×10⁻⁴, clothing ~14.67×10⁻⁴ (dataset-average references). >50% deviation from expected range = flag.
- Carbon vs distance/vehicle check: EV van 1400km with λ=2.5 is ~525,000 gCO2; bike 1400km with λ=1.0 is ~70,000 gCO2.
  Flag major deviations.
- Ensure cost > 0 and on_time within 0–100.
- Flag any zero/default where a real value is expected.
- Include "pipeline_sanity_flags" in final summary JSON/text."""

_orchestrator_agent = None


def get_orchestrator_agent(region_name: Optional[str] = None, model_id: Optional[str] = None):
    global _orchestrator_agent
    if _orchestrator_agent is not None:
        return _orchestrator_agent
    try:
        from strands import Agent
        from strands.models import BedrockModel
    except ImportError as e:
        raise ImportError(
            "Strands and Bedrock are required for orchestration. Install: pip install strands-agents boto3"
        ) from e
    from tools import get_all_tools

    region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
    model_id = model_id or os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    bedrock_model = BedrockModel(
        model_id=model_id,
        region_name=region_name,
        temperature=0.0,
    )
    _orchestrator_agent = Agent(
        model=bedrock_model,
        system_prompt=ORCHESTRATOR_PROMPT,
        tools=get_all_tools(),
    )
    return _orchestrator_agent


def run_orchestrator(message: str, region_name: Optional[str] = None, model_id: Optional[str] = None) -> str:
    agent = get_orchestrator_agent(region_name=region_name, model_id=model_id)
    return agent(message)


GATHER_PROMPT = """You are a supply chain intelligence gatherer. Extract shipment features, assess delivery risk, and find carrier options.

Your tools:
1. **extract_features_tool(query)** – Extract shipment features from natural language. Returns features + defaults_used.
2. **risk_agent_tool(features_json, risk_queries_json)** – Assess delivery risk via the Risk Agent (LLM). Gathers weather/news/web signals and returns risk_level, risk_factors, delay_probability, recommended_buffer_days. Generate 4–6 adaptive risk queries from origin, destination, and package_type.
3. **sourcing_agent_tool(features_json, risk_assessment_json, sourcing_queries_json)** – Get carrier options via the Sourcing Agent (LLM). Returns carrier_options, recommendation, industry_benchmark. Generate 2–4 pricing/sourcing queries.

Call all three tools in order. A human will review the results before optimization runs.

Validation after extract_features_tool:
- Is distance_km plausible? Mumbai→Delhi ~1400km, Mumbai→Pune ~150km, Delhi→Kolkata ~1500km, Chennai→Bangalore ~350km.
  If a cross-state route shows distance_km < 100, flag as ANOMALY_DISTANCE.
- Is package_type semantically correct? "insulin"→pharmacy, "shirt"→clothing.
  If defaults_used contains package_type but query has a specific product, flag as ANOMALY_PACKAGE_TYPE.
- Is package_weight_kg plausible for package_type?
- Add anomalies to your final summary under "validation_flags". Do not stop pipeline; flag and continue."""


def get_streaming_orchestrator_agent(
    event_queue: "queue.Queue[dict[str, Any]]",
    region_name: Optional[str] = None,
    model_id: Optional[str] = None,
    phase: str = "gather",
):
    """
    Create a fresh orchestrator agent with Strands hooks that emit tool lifecycle events.
    """
    try:
        from strands import Agent
        from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent
        from strands.models import BedrockModel
    except ImportError as e:
        raise ImportError(
            "Strands and Bedrock are required for orchestration. Install: pip install strands-agents boto3"
        ) from e
    from tools import get_all_tools, get_gather_tools

    region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
    model_id = model_id or os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    bedrock_model = BedrockModel(
        model_id=model_id,
        region_name=region_name,
        temperature=0.0,
    )
    tools = get_gather_tools() if phase == "gather" else get_all_tools()
    prompt = GATHER_PROMPT if phase == "gather" else ORCHESTRATOR_PROMPT
    agent = Agent(
        model=bedrock_model,
        system_prompt=prompt,
        tools=tools,
    )

    def _tool_name(tool_use: Any) -> str:
        if isinstance(tool_use, dict):
            if isinstance(tool_use.get("name"), str):
                return tool_use["name"]
            if isinstance(tool_use.get("toolName"), str):
                return tool_use["toolName"]
            if isinstance(tool_use.get("tool"), dict):
                nested = tool_use["tool"]
                return str(nested.get("name") or nested.get("toolName") or "")
        return ""

    def _tool_input(tool_use: Any) -> dict[str, Any]:
        if isinstance(tool_use, dict):
            payload = tool_use.get("input")
            if isinstance(payload, dict):
                return payload
            payload = tool_use.get("toolInput")
            if isinstance(payload, dict):
                return payload
        return {}

    def on_tool_start(event: BeforeToolCallEvent):
        try:
            event_queue.put(
                {
                    "current_tool_use": {
                        "name": _tool_name(getattr(event, "tool_use", None)),
                        "input": _tool_input(getattr(event, "tool_use", None)),
                    }
                }
            )
        except Exception:
            pass

    def on_tool_done(event: AfterToolCallEvent):
        try:
            event_queue.put(
                {
                    "tool_result": {
                        "name": _tool_name(getattr(event, "tool_use", None)),
                        "result": getattr(event, "result", None),
                    }
                }
            )
        except Exception:
            pass

    agent.add_hook(on_tool_start, BeforeToolCallEvent)
    agent.add_hook(on_tool_done, AfterToolCallEvent)
    return agent


def run_orchestrator_stream(
    message: str,
    event_queue: "queue.Queue[dict[str, Any]]",
    region_name: Optional[str] = None,
    model_id: Optional[str] = None,
    phase: str = "gather",
) -> str:
    """
    Run the orchestrator with tool hooks; events are emitted to event_queue.
    phase='gather' uses only tools 1-3 (extract/risk/sourcing).
    phase='full' uses all 5 tools.
    """
    agent = get_streaming_orchestrator_agent(event_queue, region_name=region_name, model_id=model_id, phase=phase)
    return agent(message)


if __name__ == "__main__":
    import sys

    msg = sys.argv[1] if len(sys.argv) > 1 else "Optimize delivery for insulin from Mumbai to Delhi, 150 km."
    print(run_orchestrator(msg))

