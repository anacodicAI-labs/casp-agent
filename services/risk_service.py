"""
Risk Service: Python backend for risk assessment using Module 07 (Early-Warning System).
Strands LLM Risk Agent is in orchestration/risk_agent_strands.py.
"""

from typing import Dict, Any, Optional
from utils.risk_tools import assess_risk, assess_risk_with_cascade


class RiskService:
    """Python service that assesses supply chain risk using Module 07 with optional weather cascade."""

    def __init__(self, ews):
        """
        Args:
            ews: EarlyWarningSystem instance (Module 07)
        """
        self.ews = ews

    def assess(
        self,
        origin: str,
        destination: str,
        weather_condition: Optional[str] = None,
        package_type: str = "clothing",
        route_dict: Optional[Dict] = None,
        distance_km: Optional[float] = None,
        package_weight_kg: Optional[float] = None,
        delivery_partner: Optional[str] = None,
        use_weather_cascade: bool = True,
    ) -> Dict[str, Any]:
        """
        Assess risk for a shipment.
        When use_weather_cascade is True, resolves weather via API → News → Web search → local.
        Returns: risk_level, risk_factors, warnings, recommended_buffer_days, delay_probability.
        """
        if not use_weather_cascade and weather_condition is not None:
            return assess_risk(
                self.ews,
                origin=origin,
                destination=destination,
                weather_condition=weather_condition,
                package_type=package_type,
                route_dict=route_dict,
                distance_km=distance_km,
                package_weight_kg=package_weight_kg,
                delivery_partner=delivery_partner,
            )
        return assess_risk_with_cascade(
            self.ews,
            origin=origin,
            destination=destination,
            package_type=package_type,
            route_dict=route_dict,
            distance_km=distance_km,
            package_weight_kg=package_weight_kg,
            delivery_partner=delivery_partner,
            use_weather_cascade=use_weather_cascade,
        )
