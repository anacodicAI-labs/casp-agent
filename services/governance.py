"""Services: Governance levers and simulated policy impact."""

from copy import deepcopy
from typing import Any, Dict, List, Optional

from config.agent_mapping import get_agent_config


class GovernanceLevers:
    """Generate governance policy recommendations and compute impact via simulation."""

    def __init__(self):
        self.policies = []

    def generate_policy_recommendations(
        self,
        package_type: str,
        analysis_results: Dict,
        simulation_context: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        config = get_agent_config(package_type)
        stakes_level = config["agent"]

        policies = {
            "package_type": package_type,
            "stakes_level": stakes_level,
            "on_time_threshold": config["on_time_threshold"] * 100,
            "recommendations": [],
            "governance_impact": {},
        }

        if stakes_level == "critical":
            policies["recommendations"].append(
                {
                    "category": "Vehicle Selection",
                    "recommendation": "Use EV Van (minimize carbon within safety constraints)",
                    "rationale": "High-stakes packages require reliable transport with carbon optimization",
                }
            )
        elif stakes_level == "standard":
            policies["recommendations"].append(
                {
                    "category": "Vehicle Selection",
                    "recommendation": "Use Bike/Scooter (maximize carbon savings)",
                    "rationale": "Low-stakes packages can use low-carbon vehicles without penalty",
                }
            )
        else:
            policies["recommendations"].append(
                {
                    "category": "Vehicle Selection",
                    "recommendation": "Use EV Van or Van (balanced approach)",
                    "rationale": "Medium-stakes packages balance cost and carbon",
                }
            )

        if stakes_level == "critical":
            policies["recommendations"].append(
                {
                    "category": "Inventory Buffer",
                    "recommendation": "+20% safety stock buffer",
                    "rationale": "High-stakes packages require safety stock to meet ≥99% on-time",
                }
            )
        elif stakes_level == "standard":
            policies["recommendations"].append(
                {
                    "category": "Inventory Buffer",
                    "recommendation": "-10% lean inventory (reduce waste)",
                    "rationale": "Low-stakes packages can operate with lean inventory",
                }
            )
        else:
            policies["recommendations"].append(
                {
                    "category": "Inventory Buffer",
                    "recommendation": "+10% standard buffer",
                    "rationale": "Medium-stakes packages need moderate buffer",
                }
            )

        if stakes_level == "critical":
            policies["recommendations"].append(
                {
                    "category": "Carrier Selection",
                    "recommendation": "Premium partners only (≥98% reliability)",
                    "rationale": "High-stakes packages require maximum reliability",
                }
            )
        elif stakes_level == "standard":
            policies["recommendations"].append(
                {
                    "category": "Carrier Selection",
                    "recommendation": "Budget carriers OK (≥85% reliability)",
                    "rationale": "Low-stakes packages can use cost-effective carriers",
                }
            )
        else:
            policies["recommendations"].append(
                {
                    "category": "Carrier Selection",
                    "recommendation": "Standard carriers (≥95% reliability)",
                    "rationale": "Medium-stakes packages need reliable but cost-effective options",
                }
            )

        if stakes_level == "critical":
            policies["recommendations"].append(
                {
                    "category": "AI Compute Policy",
                    "recommendation": "Use Claude-3-Sonnet with controlled inference budget",
                    "rationale": "Align compute policy with the deployed LLM stack while minimizing AI carbon",
                }
            )
        elif stakes_level == "standard":
            policies["recommendations"].append(
                {
                    "category": "AI Compute Policy",
                    "recommendation": "Batch queries (reduce AI calls)",
                    "rationale": "Low-stakes packages can batch optimizations to reduce AI carbon",
                }
            )
        else:
            policies["recommendations"].append(
                {
                    "category": "AI Compute Policy",
                    "recommendation": "Use Claude-3-Sonnet with moderate query frequency",
                    "rationale": "Balance AI optimization benefits with carbon cost",
                }
            )

        if simulation_context:
            policies["governance_impact"] = self.compute_governance_impact(
                package_type=package_type,
                analysis_results=analysis_results,
                simulation_context=simulation_context,
            )

        return policies

    def compute_governance_impact(
        self,
        package_type: str,
        analysis_results: Dict[str, Any],
        simulation_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Re-run optimizer under each governance lever and compute deltas vs baseline.

        Required simulation_context keys:
        - route_options: list[dict] route candidates
        - cost_predictor: callable(route)->float
        - on_time_predictor: callable(route)->float
        Optional:
        - origin, destination, priority
        """
        from services.stakes_optimizer import StakesOptimizer

        route_options: List[Dict[str, Any]] = simulation_context.get("route_options") or []
        cost_predictor = simulation_context.get("cost_predictor")
        on_time_predictor = simulation_context.get("on_time_predictor")
        if not route_options or cost_predictor is None or on_time_predictor is None:
            return {"status": "skipped", "reason": "Missing route_options/predictors for simulation"}

        optimizer = StakesOptimizer()
        origin = simulation_context.get("origin")
        destination = simulation_context.get("destination")
        priority = simulation_context.get("priority", "carbon")

        baseline = optimizer.optimize_route(
            package_type=package_type,
            route_options=deepcopy(route_options),
            cost_predictor=cost_predictor,
            on_time_predictor=on_time_predictor,
            origin=origin,
            destination=destination,
            priority=priority,
        )
        baseline_metrics = self._extract_metrics(baseline)

        scenarios: List[Dict[str, Any]] = []
        scenarios.append(
            self._run_vehicle_scenario(
                optimizer, package_type, route_options, cost_predictor, on_time_predictor, origin, destination, priority, baseline_metrics
            )
        )
        scenarios.append(
            self._run_buffer_scenario(
                optimizer, package_type, route_options, cost_predictor, on_time_predictor, origin, destination, priority, baseline_metrics, uplift_pct=2.0
            )
        )
        scenarios.append(
            self._run_carrier_scenario(
                optimizer, package_type, route_options, cost_predictor, on_time_predictor, origin, destination, priority, baseline_metrics
            )
        )
        scenarios.append(self._run_ai_policy_scenario(baseline, baseline_metrics))

        return {
            "status": "computed",
            "baseline": baseline_metrics,
            "scenarios": [s for s in scenarios if s],
        }

    @staticmethod
    def _extract_metrics(result: Dict[str, Any]) -> Dict[str, float]:
        b = result.get("breakdown", {})
        return {
            "cost_inr": float(b.get("cost", 0.0) or 0.0),
            "on_time_pct": float(b.get("predicted_on_time", 0.0) or 0.0),
            "transport_carbon_gco2": float(b.get("transport_carbon_gco2", 0.0) or 0.0),
            "ai_carbon_gco2": float(b.get("ai_carbon_gco2", 0.0) or 0.0),
            "total_carbon_gco2": float(b.get("total_carbon_gco2", 0.0) or 0.0),
        }

    @staticmethod
    def _pct_delta(new: float, base: float) -> float:
        if base == 0:
            return 0.0
        return ((new - base) / base) * 100.0

    def _format_scenario(self, lever_name: str, description: str, metrics: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, Any]:
        return {
            "lever": lever_name,
            "description": description,
            "metrics": metrics,
            "delta_pct": {
                "cost_inr": self._pct_delta(metrics["cost_inr"], baseline["cost_inr"]),
                "on_time_pct": self._pct_delta(metrics["on_time_pct"], baseline["on_time_pct"]),
                "transport_carbon_gco2": self._pct_delta(metrics["transport_carbon_gco2"], baseline["transport_carbon_gco2"]),
                "ai_carbon_gco2": self._pct_delta(metrics["ai_carbon_gco2"], baseline["ai_carbon_gco2"]),
                "total_carbon_gco2": self._pct_delta(metrics["total_carbon_gco2"], baseline["total_carbon_gco2"]),
            },
        }

    def _run_vehicle_scenario(
        self,
        optimizer,
        package_type: str,
        route_options: List[Dict[str, Any]],
        cost_predictor,
        on_time_predictor,
        origin,
        destination,
        priority: str,
        baseline: Dict[str, float],
    ) -> Dict[str, Any]:
        distances = [float(r.get("distance_km") or 0) for r in route_options if r.get("distance_km")]
        representative_distance = distances[0] if distances else 0.0
        if representative_distance < 80:
            ev_vehicle = "ev bike"
            description = f"Force EV Bike (distance {representative_distance:.0f} km < 80 km)"
        elif representative_distance <= 400:
            ev_vehicle = "ev van"
            description = f"Force EV Van (distance {representative_distance:.0f} km, 80–400 km)"
        else:
            # No long-haul EV truck in current fleet assumptions: keep incumbent vehicle types.
            ev_vehicle = None
            description = (
                f"No EV option viable for {representative_distance:.0f} km — "
                "delta_pct reflects 0% EV improvement"
            )

        forced = []
        for route in deepcopy(route_options):
            if ev_vehicle:
                route["vehicle_type"] = ev_vehicle
            forced.append(route)
        res = optimizer.optimize_route(package_type, forced, cost_predictor, on_time_predictor, origin, destination, priority)
        return self._format_scenario("vehicle_selection", description, self._extract_metrics(res), baseline)

    def _run_buffer_scenario(
        self,
        optimizer,
        package_type: str,
        route_options: List[Dict[str, Any]],
        cost_predictor,
        on_time_predictor,
        origin,
        destination,
        priority: str,
        baseline: Dict[str, float],
        uplift_pct: float,
    ) -> Dict[str, Any]:
        def boosted_on_time(route: Dict[str, Any]) -> float:
            return min(100.0, float(on_time_predictor(route)) + uplift_pct)

        res = optimizer.optimize_route(package_type, deepcopy(route_options), cost_predictor, boosted_on_time, origin, destination, priority)
        return self._format_scenario("inventory_buffer", f"Apply safety-stock policy as +{uplift_pct:.1f}pp service uplift", self._extract_metrics(res), baseline)

    def _run_carrier_scenario(
        self,
        optimizer,
        package_type: str,
        route_options: List[Dict[str, Any]],
        cost_predictor,
        on_time_predictor,
        origin,
        destination,
        priority: str,
        baseline: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        premium = [r for r in deepcopy(route_options) if float(on_time_predictor(r)) >= 98.0]
        if not premium:
            return {
                "lever": "carrier_selection",
                "description": "Premium partners only (>=98% reliability)",
                "status": "skipped",
                "reason": "No candidate route satisfied premium reliability cutoff",
            }
        res = optimizer.optimize_route(package_type, premium, cost_predictor, on_time_predictor, origin, destination, priority)
        return self._format_scenario("carrier_selection", "Premium partners only (>=98% reliability)", self._extract_metrics(res), baseline)

    def _run_ai_policy_scenario(self, baseline_result: Dict[str, Any], baseline: Dict[str, float]) -> Dict[str, Any]:
        breakdown = baseline_result.get("breakdown", {})
        ai_base = float(breakdown.get("ai_carbon_gco2", 0.0) or 0.0)
        ai_with_flash = ai_base * 0.85
        metrics = dict(baseline)
        metrics["ai_carbon_gco2"] = ai_with_flash
        metrics["total_carbon_gco2"] = metrics["transport_carbon_gco2"] + ai_with_flash
        return self._format_scenario("ai_compute_policy", "Use lower-energy inference profile (assumed 15% lower AI carbon)", metrics, baseline)

