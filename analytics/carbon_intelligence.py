"""
Analytics: Carbon Cost of Intelligence (AI compute carbon).
"""

from dataclasses import dataclass
from typing import Dict

from config.grid_carbon import GRID_CARBON_INTENSITY


@dataclass
class AIModelConfig:
    name: str
    energy_per_inference_wh: float
    description: str
    source: str


class CarbonCostOfIntelligence:
    MODEL_ENERGY_WH = {
        "claude-3-haiku": 0.0006,
        "claude-3-sonnet": 0.0015,
        "claude-3-opus": 0.0027,
        "gpt-4": 0.0029,
        "gpt-4-turbo": 0.0023,
        "gpt-3.5-turbo": 0.0004,
        "gemini-pro": 0.0018,
        "gemini-flash": 0.0006,
        "gradient-boosting": 0.00001,
        "random-forest": 0.00001,
        "xgboost": 0.00002,
        "neural-net-small": 0.0001,
        "default": 0.001,
    }

    TOKENS_PER_QUERY = {
        "route_optimization": 1500,
        "demand_forecasting": 800,
        "vendor_selection": 600,
        "risk_assessment": 1000,
        "simple_query": 200,
    }

    def __init__(self, default_country: str = "usa"):
        self.default_country = default_country.lower()
        self.total_inferences = 0
        self.total_carbon_gco2 = 0.0
        self.inference_log = []

    def get_grid_intensity(self, country: str) -> float:
        return GRID_CARBON_INTENSITY.get(country.lower(), 400)

    def calculate_inference_carbon(
        self,
        model_type: str,
        country: str | None = None,
        num_inferences: int = 1,
        task_type: str = "default",
    ) -> Dict:
        if country is None:
            country = self.default_country
        energy_wh = self.MODEL_ENERGY_WH.get(model_type.lower(), self.MODEL_ENERGY_WH["default"])
        grid_intensity = self.get_grid_intensity(country)
        energy_kwh = (energy_wh * num_inferences) / 1000
        carbon_gco2 = energy_kwh * grid_intensity
        self.total_inferences += num_inferences
        self.total_carbon_gco2 += carbon_gco2
        result = {
            "model": model_type,
            "country": country,
            "num_inferences": num_inferences,
            "energy_per_inference_wh": energy_wh,
            "total_energy_wh": energy_wh * num_inferences,
            "total_energy_kwh": energy_kwh,
            "grid_intensity_gco2_kwh": grid_intensity,
            "carbon_gco2": carbon_gco2,
            "carbon_kg": carbon_gco2 / 1000,
        }
        self.inference_log.append(result)
        return result

    def calculate_optimization_carbon(
        self,
        num_routes: int,
        model_type: str = "gradient-boosting",
        country: str | None = None,
        include_llm_reasoning: bool = False,
    ) -> Dict:
        if country is None:
            country = self.default_country
        ml_inferences = num_routes * 3
        ml_carbon = self.calculate_inference_carbon(
            model_type=model_type,
            country=country,
            num_inferences=ml_inferences,
            task_type="route_optimization",
        )
        result = {
            "task": "route_optimization",
            "num_routes_evaluated": num_routes,
            "ml_model": model_type,
            "ml_inferences": ml_inferences,
            "ml_carbon_gco2": ml_carbon["carbon_gco2"],
            "country": country,
            "grid_intensity": ml_carbon["grid_intensity_gco2_kwh"],
        }
        if include_llm_reasoning:
            llm_model = "claude-3-sonnet"
            llm_carbon = self.calculate_inference_carbon(
                model_type=llm_model,
                country=country,
                num_inferences=3,
                task_type="route_optimization",
            )
            result["llm_model"] = llm_model
            result["llm_carbon_gco2"] = llm_carbon["carbon_gco2"]
            result["llm_inferences"] = 3
            result["total_ai_carbon_gco2"] = ml_carbon["carbon_gco2"] + llm_carbon["carbon_gco2"]
        else:
            result["total_ai_carbon_gco2"] = ml_carbon["carbon_gco2"]
            result["llm_model"] = None
            result["llm_carbon_gco2"] = 0.0
            result["llm_inferences"] = 0
        return result

