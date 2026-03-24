"""
Analytics: Grid Carbon Scenarios (country comparisons).
"""

import numpy as np
import pandas as pd

from config.grid_carbon import (
    GRID_CARBON_INTENSITY,
    calculate_ai_compute_carbon,
    calculate_model_carbon,
    get_country_from_region,
)


class GridCarbonScenarios:
    """Analyze grid carbon scenarios by country."""

    def __init__(self):
        self.countries = list(GRID_CARBON_INTENSITY.keys())

    def compare_countries(self, energy_joules: float = 500) -> pd.DataFrame:
        results = []
        for country in self.countries:
            grid_intensity = GRID_CARBON_INTENSITY[country]
            ai_carbon = calculate_ai_compute_carbon(energy_joules, country)
            results.append(
                {
                    "country": country,
                    "grid_intensity_gco2_kwh": grid_intensity,
                    "ai_compute_carbon_gco2": ai_carbon,
                    "energy_joules": energy_joules,
                }
            )
        df = pd.DataFrame(results)
        return df.sort_values("ai_compute_carbon_gco2")

    def analyze_route_by_country(self, transport_carbon: float, ai_tokens: int = 500, model: str = "gemini-flash") -> pd.DataFrame:
        results = []
        for country in self.countries:
            ai_carbon = calculate_model_carbon(ai_tokens, model, country)
            total_carbon = transport_carbon + ai_carbon
            results.append(
                {
                    "country": country,
                    "transport_carbon_gco2": transport_carbon,
                    "ai_compute_carbon_gco2": ai_carbon,
                    "total_carbon_gco2": total_carbon,
                    "ai_carbon_pct": (ai_carbon / total_carbon * 100) if total_carbon > 0 else 0,
                }
            )
        df = pd.DataFrame(results)
        return df.sort_values("total_carbon_gco2")

    def get_optimal_country(self, transport_carbon: float, ai_tokens: int = 500, model: str = "gemini-flash") -> dict:
        df = self.analyze_route_by_country(transport_carbon, ai_tokens, model)
        optimal = df.iloc[0]
        india = df[df["country"] == "India"].iloc[0] if "India" in set(df["country"]) else None
        return {
            "optimal_country": optimal["country"],
            "total_carbon_gco2": optimal["total_carbon_gco2"],
            "ai_carbon_gco2": optimal["ai_compute_carbon_gco2"],
            "transport_carbon_gco2": optimal["transport_carbon_gco2"],
            "savings_vs_india": (india["total_carbon_gco2"] - optimal["total_carbon_gco2"]) if india is not None else None,
        }

    def scenario_analysis(self, distance_km: float, vehicle_type: str = "van", ai_tokens: int = 500, model: str = "gemini-flash") -> pd.DataFrame:
        from config.vehicle_emissions import calculate_transport_carbon

        transport_carbon = calculate_transport_carbon(distance_km, vehicle_type)
        return self.analyze_route_by_country(transport_carbon, ai_tokens, model)

