"""
Sourcing Service: Python backend for carrier options and optimization (Modules 01, 02, carriers, routes).
Used by get_carrier_options_tool, run_optimization_tool. Strands LLM Sourcing Agent is in orchestration/sourcing_agent_strands.py.
"""

from typing import Dict, List, Any, Optional
from utils.sourcing_tools import (
    lookup_route,
    get_package_rules,
    get_carrier_quotes_for_features,
    predict_performance,
    get_carrier_cluster_name,
    filter_options_by_sla,
    build_route_options_from_quotes,
)
from config.agent_mapping import get_stakes_level
from services.stakes_optimizer import StakesOptimizer


class SourcingService:
    """
    Python service that finds carrier options and runs optimization using Module 01, 02,
    carriers, routes, agent_mapping, and StakesOptimizer (single optimizer for all stakes levels).
    """

    def __init__(self, cost_predictor, on_time_predictor, analytics, vendor_segmentation=None):
        """
        Args:
            cost_predictor: From create_predictors (Module 01)
            on_time_predictor: From create_predictors (Module 01)
            analytics: PredictiveAnalytics instance (Module 01)
            vendor_segmentation: VendorSegmentation instance (Module 02), optional
        """
        self.cost_predictor = cost_predictor
        self.on_time_predictor = on_time_predictor
        self.analytics = analytics
        self.vendor_segmentation = vendor_segmentation
        self.stakes_optimizer = StakesOptimizer()

    def get_carrier_options(
        self,
        features: Dict[str, Any],
        risk_assessment: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Get carrier quotes and enrich with ML predictions (Module 01) and cluster (Module 02).
        Returns list of options with cost, on_time_pct, carbon, meets_sla, cluster.
        """
        quotes = get_carrier_quotes_for_features(features, risk_assessment)
        rules = get_package_rules(features.get("package_type") or "clothing")
        threshold_pct = rules["on_time_threshold_pct"]

        distance_km = features.get("distance_km")
        if distance_km is None and features.get("origin") and features.get("destination"):
            route_info = lookup_route(
                str(features["origin"]), str(features["destination"])
            )
            distance_km = route_info["distance_km"] if route_info else 150
        distance_km = distance_km or 150

        route_options = build_route_options_from_quotes(quotes, features, distance_km)
        options = []
        for i, route in enumerate(route_options):
            pred = predict_performance(self.analytics, route)
            on_time = pred["predicted_on_time_pct"]
            meets_sla = on_time >= threshold_pct
            cluster = None
            if self.vendor_segmentation:
                try:
                    cluster = get_carrier_cluster_name(
                        self.vendor_segmentation,
                        route.get("delivery_partner", "delhivery"),
                    )
                except Exception:
                    pass
            options.append({
                "route": route,
                "carrier": quotes[i].get("carrier_name", route.get("delivery_partner", "")),
                "carrier_code": quotes[i].get("carrier_code", ""),
                "predicted_cost": pred["predicted_cost"],
                "predicted_on_time_pct": on_time,
                "predicted_carbon_gco2": pred["predicted_carbon_gco2"],
                "total_carbon_gco2": pred["predicted_carbon_gco2"],
                "meets_sla": meets_sla,
                "cluster": cluster,
                "cost_inr": quotes[i].get("cost_inr", pred["predicted_cost"]),
                "carbon_kg": quotes[i].get("carbon_kg", pred["predicted_carbon_gco2"] / 1000),
            })

        return options

    def run_optimization(
        self,
        package_type: str,
        route_options: List[Dict],
        cost_predictor,
        on_time_predictor,
        origin: str = None,
        destination: str = None,
        priority: str = "carbon",
    ) -> Dict[str, Any]:
        """
        Run full optimization using StakesOptimizer (single optimizer for all stakes levels).
        route_options: list of route dicts (from build_route_options_from_quotes or manual).
        """
        return self.stakes_optimizer.optimize_route(
            package_type,
            route_options,
            cost_predictor,
            on_time_predictor,
            origin,
            destination,
            priority,
        )
