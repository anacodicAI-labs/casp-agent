from __future__ import annotations

import json
from copy import deepcopy

from tools._common import _get_orchestrator, _json_safe, tool


def _priority_for_package(package_type: str) -> str:
    p = (package_type or "").strip().lower()
    return "carbon" if p in {"clothing", "cosmetics"} else "cost"


@tool
def run_optimization_tool(features_json: str, carrier_options_json: str) -> str:
    features = json.loads(features_json)
    carrier_data = json.loads(carrier_options_json)
    # Sourcing-agent output can be either:
    # - legacy: a list[carrier_option]
    # - new: dict {carrier_options, recommendation, industry_benchmark, ...}
    sourcing_block = None
    if isinstance(carrier_data, dict) and "carrier_options" in carrier_data:
        sourcing_block = carrier_data
        carrier_options = carrier_data.get("carrier_options") or []
    else:
        carrier_options = carrier_data

    orch = _get_orchestrator()
    from utils.extraction_tools import extract_features_from_dict

    features = extract_features_from_dict(features)
    package_type = (features.get("package_type") or "clothing").lower()
    priority = _priority_for_package(package_type)
    route_options = [opt["route"] for opt in carrier_options if opt.get("route")]
    if not route_options:
        return json.dumps({"error": "No route options available"})

    baseline_result = orch.sourcing_service.run_optimization(
        package_type=package_type,
        route_options=route_options,
        cost_predictor=orch.cost_predictor,
        on_time_predictor=orch.on_time_predictor,
        origin=features.get("origin"),
        destination=features.get("destination"),
        priority=priority,
    )

    constraints = {}
    if sourcing_block and isinstance(sourcing_block.get("constraints_proposed"), dict):
        constraints.update(sourcing_block["constraints_proposed"])
    if package_type in {"pharmacy", "groceries"} and "min_reliability_pct" not in constraints:
        constraints["min_reliability_pct"] = 99.0

    buffer_days = float(constraints.get("buffer_days", 0) or 0)
    uplift_pct = min(5.0, max(0.0, buffer_days * 1.0))
    min_reliability = constraints.get("min_reliability_pct")

    def constrained_on_time(route):
        base = float(orch.on_time_predictor(route))
        adjusted = min(100.0, base + uplift_pct)
        if min_reliability is not None and adjusted < float(min_reliability):
            return adjusted
        return adjusted

    result = orch.stakes_optimizer.optimize_route(
        package_type=package_type,
        route_options=route_options,
        cost_predictor=orch.cost_predictor,
        on_time_predictor=constrained_on_time if (uplift_pct > 0 or min_reliability is not None) else orch.on_time_predictor,
        origin=features.get("origin"),
        destination=features.get("destination"),
        priority=priority,
    )

    best_route = result.get("best_route", route_options[0])
    delay_prob = orch.ews.predict_delay_probability(best_route)
    risk = orch.ews.calculate_risk_score(package_type, delay_prob, best_route)
    result["early_warning"] = risk
    result["governance"] = orch.governance.generate_policy_recommendations(
        package_type,
        result,
        simulation_context={
            "route_options": route_options,
            "cost_predictor": orch.cost_predictor,
            "on_time_predictor": orch.on_time_predictor,
            "origin": features.get("origin"),
            "destination": features.get("destination"),
            "priority": priority,
        },
    )
    b0 = baseline_result.get("breakdown", {})
    b1 = result.get("breakdown", {})
    final_snapshot = deepcopy(result)
    result["plan_comparison"] = {
        "baseline_plan": baseline_result,
        "final_plan": final_snapshot,
        "deltas": {
            "cost_pct": (((float(b1.get("cost", 0) or 0) - float(b0.get("cost", 0) or 0)) / float(b0.get("cost", 1) or 1)) * 100.0)
            if float(b0.get("cost", 0) or 0) != 0
            else 0.0,
            "on_time_pct": (
                ((float(b1.get("predicted_on_time", 0) or 0) - float(b0.get("predicted_on_time", 0) or 0)) / float(b0.get("predicted_on_time", 1) or 1))
                * 100.0
            )
            if float(b0.get("predicted_on_time", 0) or 0) != 0
            else 0.0,
            "carbon_pct": (
                ((float(b1.get("total_carbon_gco2", 0) or 0) - float(b0.get("total_carbon_gco2", 0) or 0)) / float(b0.get("total_carbon_gco2", 1) or 1))
                * 100.0
            )
            if float(b0.get("total_carbon_gco2", 0) or 0) != 0
            else 0.0,
        },
    }
    result["expert_packets"] = {
        "planning_expert_packet": {
            "desk": "Pricing & Planning Desk",
            "status": "ok",
            "finding": "Baseline and constrained rerun computed.",
            "constraints_proposed": constraints,
            "evidence": ["run_optimization_tool", "stakes_optimizer"],
            "confidence": 0.9,
            "insufficient_evidence": False,
            "fallback_used": False,
            "notes": "Tool path generated baseline+rerun.",
        }
    }
    if sourcing_block:
        # Pass through LLM advice/benchmark for frontend visibility.
        result["llm_recommendation"] = sourcing_block.get("recommendation")
        result["industry_benchmark"] = sourcing_block.get("industry_benchmark")
        result["efficiency"] = sourcing_block.get("efficiency")
        result["efficiency_percentage"] = sourcing_block.get("efficiency_percentage")
        result["benchmark_reasoning"] = sourcing_block.get("benchmark_reasoning")
    return json.dumps(_json_safe(result))

