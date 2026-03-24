"""Supervisor wrapper for phased LLM-first logistics with deterministic fallback."""

from __future__ import annotations

import json
import queue
import re
import threading
from typing import Any, Dict, Generator, List, Optional

from agents._instance import get_orchestrator
from utils.extraction_tools import (
    enrich_extracted_features,
    extract_features_from_dict,
    extract_from_query_and_merge_defaults,
    extract_with_smart_llm,
)


def run_expert_consultation(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run control-tower pipeline (LLM-first with deterministic fallback):
    Risk Desk -> Sourcing Desk -> Planning baseline/rerun -> Sustainability Desk -> Synthesis.
    """
    orch = get_orchestrator()
    normalized = extract_features_from_dict(features)
    if not normalized.get("weather_origin"):
        normalized["weather_origin"] = normalized.get("weather_condition", "clear")
    if not normalized.get("weather_destination"):
        normalized["weather_destination"] = normalized.get("weather_condition", "clear")
    return orch.run_integrated_pipeline(normalized)


def _parse_sourcing_out(s: str) -> Dict[str, Any]:
    try:
        obj = json.loads(s or "{}")
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def _extract_json_from_value(value: Any) -> Any:
    """
    Best-effort extraction of JSON payloads from Strands tool result objects.
    """
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    match = re.search(r"\{.*\}|\[.*\]", s, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def _extract_payload_from_tool_result(result_obj: Any) -> Any:
    """
    Normalize different ToolResult shapes to the underlying JSON/text payload.
    """
    if isinstance(result_obj, dict):
        content = result_obj.get("content")
        if isinstance(content, list) and content:
            for block in content:
                if isinstance(block, dict):
                    if "json" in block:
                        return block["json"]
                    if "text" in block:
                        parsed = _extract_json_from_value(block["text"])
                        if parsed is not None:
                            return parsed
                        return block["text"]
        for key in ("json", "result", "output"):
            if key in result_obj:
                return result_obj[key]
    return result_obj


def run_chat_gather(query: str) -> Dict[str, Any]:
    """
    Phase 1: gather context through agents/tools, then return fully editable context.
    """
    query = (query or "").strip()
    features, defaults_used = extract_from_query_and_merge_defaults(query, use_smart_llm=True)
    orch = get_orchestrator()
    normalized = extract_features_from_dict(features)

    package_type = (normalized.get("package_type") or "clothing").lower()
    origin = str(normalized.get("origin") or "mumbai").lower()
    destination = str(normalized.get("destination") or "delhi").lower()

    weather_for_risk = normalized.get("weather_destination") or normalized.get("weather_origin") or normalized.get("weather_condition")
    risk_assessment = orch.risk_service.assess(
        origin=origin,
        destination=destination,
        weather_condition=weather_for_risk,
        package_type=package_type,
        distance_km=normalized.get("distance_km"),
        package_weight_kg=normalized.get("package_weight_kg"),
    )
    risk_llm_status = "unavailable"
    try:
        from agents.risk import run_risk_agent

        risk_queries = [
            f"{origin} weather today",
            f"{destination} weather today",
            f"{origin} {destination} logistics disruption",
        ]
        llm_risk = json.loads(run_risk_agent(json.dumps(normalized), json.dumps(risk_queries)))
        if isinstance(llm_risk, dict):
            risk_llm_status = "available"
            if llm_risk.get("recommended_buffer_days") is not None:
                risk_assessment["recommended_buffer_days"] = max(
                    int(risk_assessment.get("recommended_buffer_days", 0) or 0),
                    int(llm_risk.get("recommended_buffer_days", 0) or 0),
                )
            if llm_risk.get("buffer_rationale"):
                risk_assessment["buffer_rationale"] = llm_risk.get("buffer_rationale")
            if isinstance(llm_risk.get("risk_factors"), list):
                risk_assessment["risk_factors"] = llm_risk.get("risk_factors")
    except Exception:
        pass

    carriers: List[Dict[str, Any]] = []
    benchmark = None
    recommendation = None
    benchmark_reasoning = None
    sourcing_llm_status = "unavailable"
    try:
        from agents.sourcing import run_sourcing_agent

        dist = normalized.get("distance_km") or 150
        weight = normalized.get("package_weight_kg") or 25
        queries = [
            f"delivery cost {package_type} {dist}km {weight}kg India",
            f"cold chain {package_type} delivery cost {origin} to {destination}",
        ]
        sout = _parse_sourcing_out(
            run_sourcing_agent(
                features_json=json.dumps(normalized),
                risk_assessment_json=json.dumps(risk_assessment),
                sourcing_queries_json=json.dumps(queries),
            )
        )
        raw = sout.get("carrier_options")
        if isinstance(raw, list):
            carriers = raw
            sourcing_llm_status = "available"
        benchmark = sout.get("industry_benchmark")
        recommendation = sout.get("recommendation")
        benchmark_reasoning = sout.get("benchmark_reasoning")
    except Exception:
        pass

    if not carriers:
        carriers = orch.sourcing_service.get_carrier_options(normalized, risk_assessment)

    return {
        "phase": "gather",
        "mode": "llm_first_with_deterministic_fallback",
        "llm_status": {
            "risk": risk_llm_status,
            "sourcing": sourcing_llm_status,
        },
        "context": {
            "features": normalized,
            "defaults_used": defaults_used,
            "risk": risk_assessment,
            "carrier_options": carriers,
            "benchmark": benchmark,
            "recommendation": recommendation,
            "benchmark_reasoning": benchmark_reasoning,
        },
    }


def run_chat_gather_stream(query: str) -> Generator[Dict[str, Any], None, None]:
    """
    Stream gather-phase events in Strands-like shape for progressive frontend UX.
    Emits lifecycle/tool/data events and final gathered context.
    """
    query = (query or "").strip()
    yield {"init_event_loop": True}
    yield {"start_event_loop": True, "data": "Supervisor intake started"}

    # Keep context incrementally as real tool results arrive.
    gathered: Dict[str, Any] = {
        "features": {},
        "defaults_used": [],
        "risk": {},
        "carrier_options": [],
        "benchmark": None,
        "recommendation": None,
        "benchmark_reasoning": None,
    }
    llm_status = {"risk": "unavailable", "sourcing": "unavailable"}

    q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
    result_holder: Dict[str, Any] = {}

    def _run_agent():
        try:
            from agents.orchestrator import run_orchestrator_stream
            from tools.stream_events import set_stream_emitter, set_stream_scope

            set_stream_scope("")
            set_stream_emitter(lambda event: q.put(event))
            result_holder["output"] = run_orchestrator_stream(query, q)
        except Exception as exc:
            result_holder["error"] = str(exc)
        finally:
            try:
                from tools.stream_events import set_stream_emitter, set_stream_scope

                set_stream_emitter(None)
                set_stream_scope("")
            except Exception:
                pass
            q.put(None)

    t = threading.Thread(target=_run_agent, daemon=True)
    t.start()

    while True:
        evt = q.get()
        if evt is None:
            break
        if not isinstance(evt, dict):
            continue

        if evt.get("current_tool_use"):
            yield evt
            continue

        if evt.get("subtool_event"):
            yield evt
            continue

        if evt.get("tool_result"):
            tool = evt["tool_result"].get("name")
            raw_result = evt["tool_result"].get("result")
            payload = _extract_payload_from_tool_result(raw_result)
            parsed = _extract_json_from_value(payload)
            if parsed is None:
                parsed = payload

            # tool_result event for timeline/logging
            yield {
                "tool_result": {
                    "name": tool,
                    "result": parsed if isinstance(parsed, (dict, list, str, int, float, bool)) else str(parsed),
                }
            }

            if tool == "extract_features_tool":
                if isinstance(parsed, dict):
                    if isinstance(parsed.get("features"), dict):
                        gathered["features"] = extract_features_from_dict(parsed.get("features") or {})
                    elif parsed:
                        gathered["features"] = extract_features_from_dict(parsed)
                    if isinstance(parsed.get("defaults_used"), list):
                        gathered["defaults_used"] = parsed["defaults_used"]
                    if gathered["features"] and not gathered["features"].get("weather_origin"):
                        gathered["features"]["weather_origin"] = gathered["features"].get("weather_condition", "clear")
                    if gathered["features"] and not gathered["features"].get("weather_destination"):
                        gathered["features"]["weather_destination"] = gathered["features"].get("weather_condition", "clear")
                    yield {
                        "partial_context": {
                            "features": gathered["features"],
                            "defaults_used": gathered["defaults_used"],
                        }
                    }
            elif tool == "risk_agent_tool":
                llm_status["risk"] = "available"
                risk_obj = parsed if isinstance(parsed, dict) else _extract_json_from_value(parsed)
                if isinstance(risk_obj, dict):
                    gathered["risk"] = risk_obj
                    yield {
                        "partial_context": {
                            "risk": gathered["risk"],
                            "llm_status": {"risk": llm_status["risk"]},
                        }
                    }
            elif tool == "sourcing_agent_tool":
                llm_status["sourcing"] = "available"
                sout = parsed if isinstance(parsed, dict) else _parse_sourcing_out(str(parsed or ""))
                carriers = sout.get("carrier_options")
                if isinstance(carriers, list):
                    gathered["carrier_options"] = carriers
                elif isinstance(parsed, list):
                    gathered["carrier_options"] = parsed
                gathered["benchmark"] = sout.get("industry_benchmark")
                gathered["recommendation"] = sout.get("recommendation")
                gathered["benchmark_reasoning"] = sout.get("benchmark_reasoning")
                yield {
                    "partial_context": {
                        "carrier_options": gathered["carrier_options"],
                        "benchmark": gathered["benchmark"],
                        "recommendation": gathered["recommendation"],
                        "benchmark_reasoning": gathered["benchmark_reasoning"],
                        "llm_status": {"sourcing": llm_status["sourcing"]},
                    }
                }

    if result_holder.get("error"):
        # transparent fallback if orchestrator could not run
        yield {"data": f"LLM unavailable — using deterministic dataset fallback: {result_holder['error']}"}
        fallback = run_chat_gather(query)
        fallback["llm_status"] = {
            "risk": fallback.get("llm_status", {}).get("risk", "unavailable"),
            "sourcing": fallback.get("llm_status", {}).get("sourcing", "unavailable"),
        }
        yield {"result": fallback}
        return

    final = {
        "phase": "gather",
        "mode": "llm_first_with_deterministic_fallback",
        "llm_status": llm_status,
        "context": {
            "features": gathered["features"],
            "defaults_used": gathered["defaults_used"],
            "risk": gathered["risk"],
            "carrier_options": gathered["carrier_options"],
            "benchmark": gathered["benchmark"],
            "recommendation": gathered["recommendation"],
            "benchmark_reasoning": gathered["benchmark_reasoning"],
        },
    }
    yield {"result": final}


def _build_result_from_full_stream(
    gathered: Dict[str, Any],
    optimization_result: Optional[Dict[str, Any]],
    carbon_result: Optional[Dict[str, Any]],
    synthesis_text: str,
) -> Dict[str, Any]:
    opt = optimization_result or {}
    breakdown = opt.get("breakdown") if isinstance(opt.get("breakdown"), dict) else {}
    carbon = carbon_result or {}
    recommendation = (
        gathered.get("recommendation")
        or opt.get("llm_recommendation")
        or (opt.get("best_route") or {}).get("delivery_partner")
        or "No recommendation"
    )
    return {
        "control_tower_supervisor": "active",
        "execution_mode": "agentic_llm",
        "llm_status": "available",
        "supervisor_summary": synthesis_text or "Control Tower synthesis completed.",
        "recommendation": recommendation,
        "cost": breakdown.get("cost"),
        "on_time_probability": breakdown.get("predicted_on_time"),
        "transport_carbon": breakdown.get("total_carbon_gco2"),
        "ai_carbon": carbon.get("ai_carbon_gco2", 0),
        "total_carbon": carbon.get("total_carbon_gco2"),
        "casp_score": carbon.get("casp_score", opt.get("casp_score")),
        "casp_tier": carbon.get("casp_tier"),
        "carbon_roi": carbon.get("carbon_roi"),
        "risk_level": (gathered.get("risk") or {}).get("risk_level"),
        "risk_factors": (gathered.get("risk") or {}).get("risk_factors", []),
        "warnings": (gathered.get("risk") or {}).get("warnings", []),
        "recommended_buffer_days": (gathered.get("risk") or {}).get("recommended_buffer_days"),
        "buffer_rationale": (gathered.get("risk") or {}).get("buffer_rationale"),
        "governance": carbon.get("governance", opt.get("governance", {})),
        "greenest_viable": carbon.get("greenest_viable", ""),
        "carrier_options": gathered.get("carrier_options", []),
        "optimization_result": opt,
        "carbon_analysis": carbon,
        "benchmark_cost_inr": gathered.get("benchmark"),
        "benchmark_reasoning": gathered.get("benchmark_reasoning"),
        "efficiency": opt.get("efficiency"),
        "efficiency_percentage": opt.get("efficiency_percentage"),
    }


def run_chat_full_stream(query: str) -> Generator[Dict[str, Any], None, None]:
    """
    Single-stream pipeline (tools 1-5): extract -> risk -> sourcing -> optimize -> carbon -> synthesis.
    Emits current_tool_use/tool_result/subtool_event and a final {result}.
    """
    query = (query or "").strip()
    yield {"init_event_loop": True}
    yield {"start_event_loop": True, "data": "Full pipeline started"}

    gathered: Dict[str, Any] = {
        "features": {},
        "defaults_used": [],
        "risk": {},
        "carrier_options": [],
        "benchmark": None,
        "recommendation": None,
        "benchmark_reasoning": None,
    }
    optimization_result: Optional[Dict[str, Any]] = None
    carbon_result: Optional[Dict[str, Any]] = None
    q: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue()
    result_holder: Dict[str, Any] = {}

    def _run_agent():
        try:
            from agents.orchestrator import run_orchestrator_stream
            from tools.stream_events import set_stream_emitter, set_stream_scope

            set_stream_scope("")
            set_stream_emitter(lambda event: q.put(event))
            result_holder["output"] = run_orchestrator_stream(query, q, phase="full")
        except Exception as exc:
            result_holder["error"] = str(exc)
        finally:
            try:
                from tools.stream_events import set_stream_emitter, set_stream_scope

                set_stream_emitter(None)
                set_stream_scope("")
            except Exception:
                pass
            q.put(None)

    t = threading.Thread(target=_run_agent, daemon=True)
    t.start()

    while True:
        evt = q.get()
        if evt is None:
            break
        if not isinstance(evt, dict):
            continue

        if evt.get("current_tool_use") or evt.get("subtool_event"):
            yield evt
            continue

        if evt.get("tool_result"):
            tool = evt["tool_result"].get("name")
            raw_result = evt["tool_result"].get("result")
            payload = _extract_payload_from_tool_result(raw_result)
            parsed = _extract_json_from_value(payload)
            if parsed is None:
                parsed = payload

            yield {
                "tool_result": {
                    "name": tool,
                    "result": parsed if isinstance(parsed, (dict, list, str, int, float, bool)) else str(parsed),
                }
            }

            if tool == "extract_features_tool" and isinstance(parsed, dict):
                if isinstance(parsed.get("features"), dict):
                    gathered["features"] = extract_features_from_dict(parsed.get("features") or {})
                elif parsed:
                    gathered["features"] = extract_features_from_dict(parsed)
                if isinstance(parsed.get("defaults_used"), list):
                    gathered["defaults_used"] = parsed["defaults_used"]
                yield {"partial_context": {"features": gathered["features"], "defaults_used": gathered["defaults_used"]}}
            elif tool == "risk_agent_tool":
                risk_obj = parsed if isinstance(parsed, dict) else _extract_json_from_value(parsed)
                if isinstance(risk_obj, dict):
                    gathered["risk"] = risk_obj
                    yield {"partial_context": {"risk": gathered["risk"]}}
            elif tool == "sourcing_agent_tool":
                sout = parsed if isinstance(parsed, dict) else _parse_sourcing_out(str(parsed or ""))
                carriers = sout.get("carrier_options")
                if isinstance(carriers, list):
                    gathered["carrier_options"] = carriers
                elif isinstance(parsed, list):
                    gathered["carrier_options"] = parsed
                gathered["benchmark"] = sout.get("industry_benchmark")
                gathered["recommendation"] = sout.get("recommendation")
                gathered["benchmark_reasoning"] = sout.get("benchmark_reasoning")
                yield {
                    "partial_context": {
                        "carrier_options": gathered["carrier_options"],
                        "benchmark": gathered["benchmark"],
                        "recommendation": gathered["recommendation"],
                        "benchmark_reasoning": gathered["benchmark_reasoning"],
                    }
                }
            elif tool == "run_optimization_tool":
                optimization_result = parsed if isinstance(parsed, dict) else _extract_json_from_value(parsed)
            elif tool == "carbon_analysis_tool":
                carbon_result = parsed if isinstance(parsed, dict) else _extract_json_from_value(parsed)

    if result_holder.get("error"):
        yield {"data": f"LLM unavailable — using deterministic dataset fallback: {result_holder['error']}"}
        fallback_gather = run_chat_gather(query)
        yield {"result": run_chat_optimize(fallback_gather.get("context") or {})}
        return

    synthesis_text = str(result_holder.get("output") or "").strip()
    final = _build_result_from_full_stream(gathered, optimization_result, carbon_result, synthesis_text)
    yield {"result": final}


def run_chat_optimize_stream(context: Optional[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
    """
    SSE optimize-only stream for edited context.
    Emits tool lifecycle events for optimization and carbon stages, then final {result}.
    """
    yield {"current_tool_use": {"name": "run_optimization_tool", "input": {}}}
    result = run_chat_optimize(context)
    opt_result = result.get("optimization_result") if isinstance(result, dict) else None
    if isinstance(opt_result, dict):
        yield {"tool_result": {"name": "run_optimization_tool", "result": opt_result}}
    yield {"current_tool_use": {"name": "carbon_analysis_tool", "input": {}}}
    carbon_result = result.get("carbon_analysis") if isinstance(result, dict) else None
    if isinstance(carbon_result, dict):
        yield {"tool_result": {"name": "carbon_analysis_tool", "result": carbon_result}}
    yield {"result": result}


def run_chat_optimize(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Phase 2: run full optimization from edited gathered context.
    """
    ctx = context or {}
    features = extract_features_from_dict(ctx.get("features") or {})
    risk = ctx.get("risk") or {}
    carriers = ctx.get("carrier_options") or []

    # Apply user-edited risk hints back into feature set.
    if isinstance(risk, dict):
        if risk.get("weather_condition"):
            features["weather_condition"] = str(risk.get("weather_condition")).lower().strip()
    if features.get("weather_destination"):
        features["weather_condition"] = str(features.get("weather_destination")).lower().strip()
    elif features.get("weather_origin"):
        features["weather_condition"] = str(features.get("weather_origin")).lower().strip()
    benchmark = ctx.get("benchmark")
    recommendation = ctx.get("recommendation")
    benchmark_reasoning = ctx.get("benchmark_reasoning")
    orch = get_orchestrator()
    return orch.optimize_from_gathered_context(
        features=features,
        risk_assessment=risk if isinstance(risk, dict) else {},
        carrier_options=carriers if isinstance(carriers, list) else [],
        benchmark=benchmark,
        recommendation=recommendation,
        benchmark_reasoning=benchmark_reasoning,
    )

