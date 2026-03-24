"""
Discrete route optimizer: evaluate all route options and pick the best by stakes-level rules.
Used by StakesOptimizer (services/stakes_optimizer.py).
"""

from typing import Any, Callable, Dict, List

from config.agent_mapping import get_agent_config, get_stakes_level
from config.grid_carbon import calculate_optimization_ai_carbon
from config.vehicle_emissions import calculate_transport_carbon


class RouteOptimizer:
    """
    Constraint-aware route optimizer. Evaluates each route with cost and on-time predictors,
    applies on-time threshold by package type, and selects the best route by priority
    (cost for critical/high_value, carbon for standard).
    """

    def __init__(self, package_type: str):
        self.package_type = (package_type or "clothing").lower()
        self.config = get_agent_config(self.package_type)
        self.stakes = get_stakes_level(self.package_type)
        self.threshold_fraction = self.config["on_time_threshold"]
        self.threshold_pct = self.threshold_fraction * 100
        self.cold_chain_multiplier = self.config.get("cold_chain_multiplier", 1.0)
        self.priority = self.config.get("priority", "reliability")

    def optimize(
        self,
        route_options: List[Dict],
        cost_predictor: Callable[[Dict], float],
        on_time_predictor: Callable[[Dict], float],
    ) -> Dict[str, Any]:
        if not route_options:
            empty_route = {}
            return {
                "best_route": empty_route,
                "breakdown": self._make_breakdown_row(empty_route, 0.0, 0.0, 0.0, 0.0, False, "No routes provided."),
                "all_results": [],
                "best_objective": 0.0,
            }

        total_ai_gco2 = calculate_optimization_ai_carbon(
            num_routes=len(route_options),
            model="gradient-boosting",
            country="india",
            predictions_per_route=3,
        )
        ai_per_route = total_ai_gco2 / len(route_options) if route_options else 0.0

        all_results = []
        short_haul_only = {"bike", "ev bike", "scooter"}
        for route in route_options:
            cost = float(cost_predictor(route))
            on_time_pct = float(on_time_predictor(route))
            distance_km = float(route.get("distance_km") or 150)
            vehicle_type = (route.get("vehicle_type") or "van").lower()
            if distance_km > 200 and vehicle_type in short_haul_only:
                # Hard feasibility gate: two-wheelers are not valid for long-haul.
                continue
            transport_gco2 = calculate_transport_carbon(distance_km, vehicle_type, self.cold_chain_multiplier)
            total_gco2 = transport_gco2 + ai_per_route
            meets = on_time_pct >= self.threshold_pct
            objective = cost if self.stakes != "standard" else total_gco2
            warning = "" if meets else f"On-time {on_time_pct:.1f}% below threshold {self.threshold_pct:.0f}%."
            row = self._make_breakdown_row(route, cost, on_time_pct, transport_gco2, total_gco2, meets, warning)
            row["objective_value"] = objective
            all_results.append(row)

        if not all_results:
            empty_route = {}
            return {
                "best_route": empty_route,
                "breakdown": self._make_breakdown_row(
                    empty_route, 0.0, 0.0, 0.0, 0.0, False, "No routes passed feasibility filters."
                ),
                "all_results": [],
                "best_objective": 0.0,
            }

        valid = [r for r in all_results if r.get("meets_constraint", False)]
        if valid:
            if self.stakes == "standard":
                best_row = min(valid, key=lambda x: x["total_carbon_gco2"])
            else:
                best_row = min(valid, key=lambda x: x["cost"])
        else:
            best_row = max(all_results, key=lambda x: x.get("predicted_on_time", 0))

        return {
            "best_route": best_row["route"],
            "breakdown": best_row,
            "all_results": all_results,
            "best_objective": best_row.get("objective_value", 0.0),
        }

    def _make_breakdown_row(
        self,
        route: Dict,
        cost: float,
        on_time_pct: float,
        transport_gco2: float,
        total_gco2: float,
        meets_constraint: bool,
        warning: str,
    ) -> Dict[str, Any]:
        ai_gco2 = max(0, total_gco2 - transport_gco2)
        return {
            "route": route,
            "cost": cost,
            "predicted_on_time": on_time_pct,
            "transport_carbon_gco2": transport_gco2,
            "ai_carbon_gco2": ai_gco2,
            "total_carbon_gco2": total_gco2,
            "meets_constraint": meets_constraint,
            "warning": warning,
        }

