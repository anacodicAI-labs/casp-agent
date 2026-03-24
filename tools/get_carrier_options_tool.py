from __future__ import annotations
import json

from tools.stream_events import emit_subtool_event
from tools._common import _get_orchestrator, _json_safe, tool


@tool
def get_carrier_options_tool(features_json: str, risk_assessment_json: str) -> str:
    """
    Compute carrier options based on features + fused risk.
    Returns JSON list of carrier options.
    """

    emit_subtool_event("sourcing.carriers", "start")
    orch = _get_orchestrator()
    try:
        from utils.extraction_tools import extract_features_from_dict

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

        return json.dumps(_json_safe(out))
    except Exception as e:
        return json.dumps({"error": str(e), "options": []})

    finally:
        emit_subtool_event("sourcing.carriers", "done")
