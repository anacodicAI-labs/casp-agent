"""
Run Mumbai–Delhi case studies (insulin and fashion) via Python backend (no LLM)
and save JSON for the paper. Saves casestudy_output.json (insulin) and
casestudy_fashion_output.json (clothing, 20 kg). Usage: from code/ run:
  python3 run_casestudy_output.py           # both
  python3 run_casestudy_output.py insulin  # insulin only
  python3 run_casestudy_output.py fashion  # fashion only
"""

import os
import sys
import json

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

# Avoid heavy print from modules
import warnings
warnings.filterwarnings("ignore")

def _serialize(obj):
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(x) for x in obj]
    return str(obj)

def run_case(orch, origin, destination, package_type, package_weight_kg, query_label, out_path):
    features = {
        "origin": origin,
        "destination": destination,
        "package_type": package_type,
        "package_weight_kg": package_weight_kg,
    }
    risk = orch.risk_service.assess(
        origin=origin,
        destination=destination,
        package_type=package_type,
        use_weather_cascade=False,
    )
    route_info = __import__("utils.sourcing_tools", fromlist=["lookup_route"]).lookup_route(origin, destination)
    distance_km = route_info["distance_km"] if route_info else 1400
    features["distance_km"] = distance_km
    options = orch.sourcing_service.get_carrier_options(features, risk)
    route_options = [opt["route"] for opt in options]
    result = orch.optimize_delivery(
        package_type=package_type,
        route_options=route_options,
        origin=origin,
        destination=destination,
        priority="carbon",
    )
    breakdown = result.get("breakdown", {})
    early = result.get("early_warning", {})
    out = {
        "query": query_label,
        "risk_assessment": {
            "risk_level": risk.get("risk_level"),
            "delay_probability": round(risk.get("delay_probability", 0), 4),
            "recommended_buffer_days": risk.get("recommended_buffer_days"),
            "risk_factors": risk.get("risk_factors", []),
        },
        "distance_km": distance_km,
        "carrier_options_count": len(options),
        "best_route": {
            "delivery_partner": breakdown.get("route", {}).get("delivery_partner"),
            "vehicle_type": breakdown.get("route", {}).get("vehicle_type"),
        },
        "optimization": {
            "cost": round(breakdown.get("cost", 0), 2),
            "predicted_on_time_pct": round(breakdown.get("predicted_on_time", 0), 2),
            "total_carbon_gco2": round(breakdown.get("total_carbon_gco2", 0), 0),
        },
        "early_warning": {
            "risk_level": early.get("risk_level"),
            "delay_probability": round(early.get("delay_probability", 0), 4),
            "risk_score": round(early.get("risk_score", 0), 2),
        },
    }
    with open(os.path.join(script_dir, out_path), "w") as f:
        json.dump(_serialize(out), f, indent=2)
    print(json.dumps(_serialize(out), indent=2))
    return out

def main():
    from supply_chain_orchestrator import SupplyChainOrchestrator
    orch = SupplyChainOrchestrator()
    origin, destination = "Mumbai", "Delhi"
    which = (sys.argv[1] if len(sys.argv) > 1 else "both").lower()

    if which in ("both", "insulin"):
        run_case(
            orch, origin, destination,
            package_type="pharmacy",
            package_weight_kg=5.0,
            query_label="Ship insulin from Mumbai to Delhi",
            out_path="casestudy_output.json",
        )
        if which == "both":
            print("\n--- Fashion case ---\n")
    if which in ("both", "fashion"):
        run_case(
            orch, origin, destination,
            package_type="clothing",
            package_weight_kg=20.0,
            query_label="Ship clothing (20 kg) from Mumbai to Delhi",
            out_path="casestudy_fashion_output.json",
        )

if __name__ == "__main__":
    main()
