"""
Carbon Service: Python backend for carbon and governance (Modules 03–06).
Used by carbon_analysis_tool. No LLM; tool only.
"""

from typing import Dict, List, Any
from utils.carbon_tools import (
    calculate_ai_carbon,
    get_grid_intensity,
    analyze_tradeoffs,
    get_governance_advice,
)


class CarbonService:
    """Python service for carbon and governance using Modules 03, 04, 05, 06."""

    def __init__(self, carbon_intel, frontiers, governance, country: str = "india"):
        """
        Args:
            carbon_intel: CarbonCostOfIntelligence instance (Module 03)
            frontiers: TradeOffFrontiers instance (Module 05)
            governance: GovernanceLevers instance (Module 06)
            country: Default country for grid carbon (Module 04 via config)
        """
        self.carbon_intel = carbon_intel
        self.frontiers = frontiers
        self.governance = governance
        self.country = country

    @staticmethod
    def _compute_casp_score(service_performance_pct: float, total_carbon_gco2: float) -> float:
        """
        CASP = Service Performance / Total Carbon.
        Service performance is used on the 0..100 percentage scale to stay
        consistent with the paper's reported CASP values.
        """
        if total_carbon_gco2 is None or float(total_carbon_gco2) <= 0:
            return 0.0
        service_norm = max(0.0, float(service_performance_pct or 0.0))
        return service_norm / float(total_carbon_gco2)

    @staticmethod
    def _casp_tier_from_score(casp_score: float) -> str:
        """
        Tiering aligned to paper-style interpretation:
        - CRITICAL: low CASP (harder to achieve service per carbon)
        - HIGH: mid CASP
        - STANDARD: high CASP
        """
        s = float(casp_score or 0.0)
        if s < 8e-4:
            return "CRITICAL"
        if s < 1.5e-3:
            return "HIGH"
        return "STANDARD"

    def analyze(
        self,
        carrier_options: List[Dict],
        package_type: str,
        optimization_result: Dict[str, Any] = None,
        country: str = None,
    ) -> Dict[str, Any]:
        """
        Analyze carbon trade-offs and get governance advice.
        carrier_options: list from Sourcing Service (with carbon, cost, on_time).
        optimization_result: optional full result from sourcing (for governance input).
        """
        country = country or self.country

        ai_carbon_result = calculate_ai_carbon(
            self.carbon_intel,
            model_type="claude-3-sonnet",
            country=country,
            num_inferences=3,
        )
        ai_carbon_gco2 = ai_carbon_result["ai_carbon_gco2"]
        grid_intensity = get_grid_intensity(country)
        tradeoff_result = analyze_tradeoffs(
            self.frontiers,
            carrier_options,
            package_type,
        )

        viable = [o for o in carrier_options if o.get("meets_sla", True)]
        options_to_rank = viable if viable else carrier_options
        if options_to_rank:
            greenest = min(
                options_to_rank,
                key=lambda x: x.get("total_carbon_gco2")
                or x.get("predicted_carbon_gco2")
                or (x.get("carbon_kg", 0) * 1000),
            )
            transport_carbon = (
                greenest.get("total_carbon_gco2")
                or greenest.get("predicted_carbon_gco2")
                or (greenest.get("carbon_kg", 0) * 1000)
            )
            greenest_carrier = (
                greenest.get("carrier")
                or greenest.get("route", {}).get("delivery_partner", "")
            )
        else:
            transport_carbon = 0
            greenest_carrier = ""

        total_carbon = transport_carbon + ai_carbon_gco2
        service_performance_pct = 0.0
        if isinstance(optimization_result, dict):
            bd = optimization_result.get("breakdown") if isinstance(optimization_result.get("breakdown"), dict) else {}
            service_performance_pct = float(bd.get("predicted_on_time", 0.0) or 0.0)
        casp_score = self._compute_casp_score(service_performance_pct, total_carbon)
        casp_tier = self._casp_tier_from_score(casp_score)

        # Carbon ROI: transport carbon reduction per unit AI-carbon spent.
        carbon_roi = None
        if isinstance(optimization_result, dict):
            pc = optimization_result.get("plan_comparison") if isinstance(optimization_result.get("plan_comparison"), dict) else {}
            b0 = pc.get("baseline_plan") if isinstance(pc.get("baseline_plan"), dict) else {}
            b0_breakdown = b0.get("breakdown") if isinstance(b0.get("breakdown"), dict) else {}
            baseline_carbon = float(b0_breakdown.get("total_carbon_gco2", 0.0) or 0.0)
            final_carbon = float(
                (optimization_result.get("breakdown") or {}).get("total_carbon_gco2", 0.0)
                if isinstance(optimization_result.get("breakdown"), dict)
                else 0.0
            )
            if ai_carbon_gco2 and ai_carbon_gco2 > 0:
                carbon_roi = (baseline_carbon - final_carbon) / ai_carbon_gco2

        gov_input = optimization_result or {
            "best_route": carrier_options[0].get("route", {}) if carrier_options else {},
            "breakdown": {
                "total_carbon_gco2": total_carbon,
                "predicted_on_time": carrier_options[0].get("predicted_on_time_pct", 0)
                if carrier_options
                else 0,
                "cost": carrier_options[0].get("predicted_cost", 0) if carrier_options else 0,
            },
            "package_type": package_type,
        }
        governance_recs = get_governance_advice(
            self.governance, package_type, gov_input
        )

        return {
            "greenest_viable": greenest_carrier,
            "transport_carbon_gco2": transport_carbon,
            "ai_carbon_gco2": ai_carbon_gco2,
            "total_carbon_gco2": total_carbon,
            "service_performance_pct": service_performance_pct,
            "casp_score": casp_score,
            "casp_tier": casp_tier,
            "carbon_roi": carbon_roi,
            "grid_intensity_gco2_kwh": grid_intensity,
            "tradeoff_analysis": tradeoff_result,
            "governance": governance_recs,
        }
