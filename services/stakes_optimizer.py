"""
Stakes Optimizer: single Python optimizer for all stakes levels (critical, high_value, standard).
Used by SourcingService.run_optimization and SupplyChainOrchestrator.optimize_delivery.
Not an LLM agent; threshold and carbon-priority logic only.
"""

from typing import List, Dict, Optional
from services.route_optimizer import RouteOptimizer
from config.agent_mapping import get_agent_config, get_stakes_level


# Stakes level from get_stakes_level(package_type) -> display threshold (fraction) and agent_type label
THRESHOLDS = {
    "critical": 0.99,    # pharmacy, groceries
    "high_value": 0.95,  # automobile parts, furniture, documents, fragile items, electronics
    "standard": 0.85,    # clothing, cosmetics
}
AGENT_TYPE_LABEL = {
    "critical": "high_stakes",
    "high_value": "medium_stakes",
    "standard": "low_stakes",
}


class StakesOptimizer:
    """Single optimizer for all stakes levels; branches on get_stakes_level(package_type)."""

    def __init__(self):
        self.thresholds = THRESHOLDS
        self.agent_type_label = AGENT_TYPE_LABEL

    def optimize_route(
        self,
        package_type: str,
        route_options: List[Dict],
        cost_predictor,
        on_time_predictor,
        origin: str = None,
        destination: str = None,
        priority: str = "carbon",
    ) -> Dict:
        """
        Optimize route for any package type. Stakes level is derived from package_type.
        For standard (low-stakes) only: priority='carbon' re-sorts valid routes by lowest carbon and sets carbon_savings_pct.
        """
        stakes = get_stakes_level(package_type)
        config = get_agent_config(package_type)
        optimizer = RouteOptimizer(package_type)
        result = optimizer.optimize(route_options, cost_predictor, on_time_predictor)

        # Low-stakes only: carbon-priority re-sort among valid routes
        if stakes == "standard" and priority == "carbon":
            valid_routes = [r for r in result.get("all_results", []) if r.get("meets_constraint", False)]
            if valid_routes:
                best_carbon_route = min(valid_routes, key=lambda x: x["total_carbon_gco2"])
                result["best_route"] = best_carbon_route["route"]
                result["breakdown"] = best_carbon_route
                result["best_objective"] = best_carbon_route["objective_value"]

        agent_type = AGENT_TYPE_LABEL.get(stakes, "low_stakes")
        result["agent_type"] = agent_type
        result["on_time_threshold"] = THRESHOLDS.get(stakes, 0.85) * 100
        result["priority"] = priority if stakes == "standard" else config["priority"]
        result["cold_chain_required"] = config["cold_chain_multiplier"] > 1.0 if stakes != "standard" else False

        if result["breakdown"]["total_carbon_gco2"] > 0:
            casp = result["breakdown"].get("predicted_on_time", 0) / result["breakdown"]["total_carbon_gco2"]
        else:
            casp = 0
        result["casp_score"] = casp

        # Low-stakes only: carbon savings vs fastest route
        if stakes == "standard" and result.get("all_results"):
            fastest_route = min(result["all_results"], key=lambda x: x.get("predicted_on_time", 0))
            if fastest_route["total_carbon_gco2"] > 0:
                result["carbon_savings_pct"] = (
                    (fastest_route["total_carbon_gco2"] - result["breakdown"]["total_carbon_gco2"])
                    / fastest_route["total_carbon_gco2"]
                    * 100
                )
        return result

    def format_output(self, result: Dict) -> str:
        """Format optimization results; header and optional lines depend on result['agent_type']."""
        route = result["best_route"]
        breakdown = result["breakdown"]
        agent_type = result.get("agent_type", "low_stakes")
        on_time = breakdown.get("predicted_on_time", 0)
        check = "✅" if breakdown.get("meets_constraint") else "❌"

        if agent_type == "high_stakes":
            output = f"""
🏥 HIGH-STAKES LOGISTICS OPTIMIZATION
Package Type: {result.get('package_type', '')}
Agent: {agent_type} | Priority: {result.get('priority', '')}
Constraint: ≥{result.get('on_time_threshold', 99)}% on-time delivery

🎯 RECOMMENDED ROUTE: {route.get('route_id', 'N/A')}
• Distance: {route.get('distance_km', 0):.0f} km
• Vehicle: {route.get('vehicle_type', 'N/A')}
• Predicted On-Time: {on_time:.1f}% {check}
• Predicted Cost: ₹{breakdown.get('cost', 0):.2f}

📊 CARBON BREAKDOWN:
• Transport Carbon: {breakdown.get('transport_carbon_gco2', 0):.0f} gCO2
• AI Compute Carbon: {breakdown.get('ai_carbon_gco2', 0):.0f} gCO2
• Total Carbon: {breakdown.get('total_carbon_gco2', 0):.0f} gCO2

📈 METRICS:
• CASP Score: {result.get('casp_score', 0):.6f}
• Objective Value: {result.get('best_objective', 0):.2f}
"""
        elif agent_type == "medium_stakes":
            output = f"""
📦 MEDIUM-STAKES LOGISTICS OPTIMIZATION
Package Type: {result.get('package_type', '')}
Agent: {agent_type} | Priority: {result.get('priority', '')}
Constraint: ≥{result.get('on_time_threshold', 95)}% on-time delivery

🎯 RECOMMENDED ROUTE: {route.get('route_id', 'N/A')}
• Distance: {route.get('distance_km', 0):.0f} km
• Vehicle: {route.get('vehicle_type', 'N/A')}
• Predicted On-Time: {on_time:.1f}% {check}
• Predicted Cost: ₹{breakdown.get('cost', 0):.2f}

📊 CARBON BREAKDOWN:
• Transport Carbon: {breakdown.get('transport_carbon_gco2', 0):.0f} gCO2
• AI Compute Carbon: {breakdown.get('ai_carbon_gco2', 0):.0f} gCO2
• Total Carbon: {breakdown.get('total_carbon_gco2', 0):.0f} gCO2

📈 METRICS:
• CASP Score: {result.get('casp_score', 0):.6f}
• Objective Value: {result.get('best_objective', 0):.2f}
"""
        else:
            output = f"""
👗 LOW-STAKES LOGISTICS OPTIMIZATION
Package Type: {result.get('package_type', '')}
Agent: {agent_type} | Priority: {result.get('priority', '')}
Constraint: ≥{result.get('on_time_threshold', 85)}% on-time delivery (flexible)

🎯 RECOMMENDED ROUTE: {route.get('route_id', 'N/A')}
• Distance: {route.get('distance_km', 0):.0f} km
• Vehicle: {route.get('vehicle_type', 'N/A')}
• Predicted On-Time: {on_time:.1f}% {check}
• Predicted Cost: ₹{breakdown.get('cost', 0):.2f}

📊 CARBON BREAKDOWN:
• Transport Carbon: {breakdown.get('transport_carbon_gco2', 0):.0f} gCO2
• AI Compute Carbon: {breakdown.get('ai_carbon_gco2', 0):.0f} gCO2
• Total Carbon: {breakdown.get('total_carbon_gco2', 0):.0f} gCO2

📈 METRICS:
• CASP Score: {result.get('casp_score', 0):.6f}
• Objective Value: {result.get('best_objective', 0):.2f}
"""
        if not breakdown.get("meets_constraint"):
            output += f"\n⚠️ WARNING: {breakdown.get('warning', 'Constraint not met!')}"
        if result.get("cold_chain_required"):
            output += "\n❄️ Cold chain required (2.5× carbon multiplier)"
        if agent_type == "low_stakes":
            if "carbon_savings_pct" in result:
                output += f"\n💚 CARBON SAVINGS: {result['carbon_savings_pct']:.1f}% vs fastest route!"
            output += "\n🌟 LOW-STAKES ADVANTAGE: Flexible timing allows aggressive carbon optimization!"
        return output
