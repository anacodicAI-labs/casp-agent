"""
Carbon tools: wrap Module 03, 04, 05, 06 for the Carbon Agent.
"""

from typing import Dict, List, Any
import sys
import os

_code_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _code_root not in sys.path:
    sys.path.insert(0, _code_root)

from config.grid_carbon import get_grid_carbon_intensity


def calculate_ai_carbon(
    carbon_intel,
    model_type: str = "gemini-flash",
    country: str = "india",
    num_inferences: int = 1,
) -> Dict[str, Any]:
    result = carbon_intel.calculate_inference_carbon(
        model_type=model_type,
        country=country,
        num_inferences=num_inferences,
    )
    return {
        "ai_carbon_gco2": result["carbon_gco2"],
        "ai_carbon_kg": result["carbon_kg"],
        "model": model_type,
        "country": country,
        "grid_intensity_gco2_kwh": result["grid_intensity_gco2_kwh"],
    }


def get_grid_intensity(country: str = "india") -> float:
    return get_grid_carbon_intensity(country)


def analyze_tradeoffs(frontiers, carrier_options: List[Dict], package_type: str) -> Dict[str, Any]:
    for opt in carrier_options:
        carbon = opt.get("total_carbon_gco2") or opt.get("carbon_gco2") or (opt.get("carbon_kg", 0) * 1000)
        service = opt.get("predicted_on_time_pct") or opt.get("estimated_on_time_pct") or opt.get("service_level", 0)
        cost = opt.get("predicted_cost") or opt.get("cost_inr") or opt.get("cost", 0)
        casp = (service / carbon) if carbon > 0 else 0
        frontiers.add_result(package_type, carbon, service, cost, casp)

    pareto = frontiers.calculate_pareto_frontier()
    casp_ranking = frontiers.calculate_casp_ranking()

    return {
        "pareto_frontier": pareto,
        "casp_ranking": casp_ranking,
        "recommendations": frontiers.get_recommendations() if hasattr(frontiers, "get_recommendations") else {},
    }


def get_governance_advice(governance, package_type: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    return governance.generate_policy_recommendations(package_type, analysis_results)

