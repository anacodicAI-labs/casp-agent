from __future__ import annotations

import json

from tools._common import _get_orchestrator, _json_safe, tool


@tool
def calculate_risk_score_tool(package_type: str, weather_condition: str, risk_factors_json: str, route_dict_json: str) -> str:
    """
    Compute risk level/score using the orchestrator's early-warning system.
    Returns a JSON string.
    """

    orch = _get_orchestrator()
    ews = orch.ews
    try:
        route_dict = json.loads(route_dict_json) if route_dict_json else {}
        risk_factors = json.loads(risk_factors_json) if risk_factors_json else []
        if not isinstance(risk_factors, list):
            risk_factors = [risk_factors] if risk_factors else []

        route_dict.setdefault("weather_condition", weather_condition or "clear")
        route_dict.setdefault("package_type", package_type or "clothing")
        route_dict.setdefault("delivery_partner", "delhivery")
        route_dict.setdefault("vehicle_type", "van")
        route_dict.setdefault("delivery_mode", "express")
        route_dict.setdefault("region", "west")
        route_dict.setdefault("distance_km", 150)
        route_dict.setdefault("package_weight_kg", 25)
        route_dict.setdefault("delivery_rating", 4)

        delay_prob = ews.predict_delay_probability(route_dict)
        risk = ews.calculate_risk_score(package_type, delay_prob, route_dict)

        merged_factors = list(risk.get("risk_factors", [])) + [f for f in risk_factors if f and f not in risk.get("risk_factors", [])]
        buffer_map = {"CRITICAL": 2, "HIGH": 1, "MEDIUM": 1, "LOW": 0}
        min_buffer_days = buffer_map.get(risk["risk_level"], 0)

        out = {
            "risk_level": risk["risk_level"],
            "risk_score": risk["risk_score"],
            "risk_factors": merged_factors,
            "warnings": merged_factors.copy(),
            "delay_probability": delay_prob,
            # Deterministic safety floor; Risk Agent may recommend higher buffer.
            "min_buffer_days": min_buffer_days,
            "alert_required": risk.get("alert_required", False),
        }
        if risk.get("alert_required"):
            out["warnings"].append("Alert required: high disruption risk")

        return json.dumps(_json_safe(out))
    except Exception as e:
        return json.dumps(
            {
                "risk_level": "MEDIUM",
                "risk_score": 2.0,
                "risk_factors": [str(e)],
                "delay_probability": 0.2,
                "min_buffer_days": 1,
                "error": str(e),
            }
        )

