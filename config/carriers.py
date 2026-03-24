"""
Realistic Carrier Database
Real Indian logistics carriers with actual performance characteristics

This replaces the vague "delivery_partner" column with actionable carrier data.

Data is now loaded from data/reference/carriers.json instead of being hardcoded.

References:
- Carrier performance metrics compiled from industry reports and carrier public information (2024)
- Indian logistics market analysis reports
- Carrier websites and service documentation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import json
import os
import random

@dataclass
class CarrierProfile:
    """Profile for a logistics carrier."""
    name: str
    code: str
    type: str  # 'express', 'standard', 'economy'
    
    # Performance metrics (based on industry data)
    avg_on_time_pct: float
    on_time_variance: float  # How much it varies
    
    # Pricing (₹ per kg-km, approximate)
    base_rate_per_kg_km: float
    express_multiplier: float
    cold_chain_multiplier: float
    
    # Capabilities
    has_cold_chain: bool
    has_ev_fleet: bool
    max_weight_kg: float
    coverage_regions: List[str]
    
    # Carbon profile
    carbon_gco2_per_km: float  # Average across fleet
    
    # Reliability by route type
    metro_to_metro_bonus: float  # % bonus for metro routes
    rural_penalty: float  # % penalty for rural


def _load_carriers() -> Dict[str, CarrierProfile]:
    """Load carriers from JSON file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate from config/ to data/reference/
    json_path = os.path.join(current_dir, '..', 'data', 'reference', 'carriers.json')
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    carriers = {}
    for key, carrier_data in data['carriers'].items():
        # Filter out metadata fields (those starting with _)
        carrier_dict = {k: v for k, v in carrier_data.items() if not k.startswith('_')}
        carriers[key] = CarrierProfile(**carrier_dict)
    
    return carriers


# Load carriers from JSON file
CARRIERS = _load_carriers()


def get_carrier(carrier_code: str) -> Optional[CarrierProfile]:
    """Get carrier profile by code."""
    return CARRIERS.get(carrier_code.lower())


def get_carriers_for_route(
    package_type: str,
    weight_kg: float,
    origin_region: str,
    requires_cold_chain: bool = False,
    min_on_time_pct: float = 85.0
) -> List[CarrierProfile]:
    """
    Get carriers that can handle this shipment.
    """
    eligible = []
    for code, carrier in CARRIERS.items():
        if weight_kg > carrier.max_weight_kg:
            continue
        if requires_cold_chain and not carrier.has_cold_chain:
            continue
        if origin_region.lower() not in carrier.coverage_regions:
            continue
        worst_case_on_time = carrier.avg_on_time_pct - carrier.on_time_variance
        if worst_case_on_time < min_on_time_pct - 5:
            continue
        eligible.append(carrier)
    return eligible


def calculate_carrier_quote(
    carrier: CarrierProfile,
    distance_km: float,
    weight_kg: float,
    is_express: bool = False,
    requires_cold_chain: bool = False,
    is_metro_to_metro: bool = False,
    is_rural: bool = False,
    weather_condition: str = 'clear'
) -> Dict:
    """Calculate shipping quote for a carrier."""
    base_cost = distance_km * weight_kg * carrier.base_rate_per_kg_km
    if is_express:
        base_cost *= carrier.express_multiplier
    if requires_cold_chain:
        if not carrier.has_cold_chain:
            return None
        base_cost *= carrier.cold_chain_multiplier
    base_cost = max(base_cost, 50)
    on_time = carrier.avg_on_time_pct
    if is_metro_to_metro:
        on_time += carrier.metro_to_metro_bonus
    if is_rural:
        on_time -= carrier.rural_penalty
    weather_penalties = {'clear': 0, 'cloudy': 1, 'rainy': 5, 'foggy': 8, 'storm': 15}
    on_time -= weather_penalties.get(weather_condition.lower(), 0)
    on_time = max(50, min(99.9, on_time))
    on_time_with_variance = on_time + random.uniform(-carrier.on_time_variance/2, carrier.on_time_variance/2)
    on_time_with_variance = max(50, min(99.9, on_time_with_variance))
    carbon_gco2 = distance_km * carrier.carbon_gco2_per_km / 1000
    if requires_cold_chain:
        carbon_gco2 *= 1.5
    return {
        'carrier_code': carrier.code,
        'carrier_name': carrier.name,
        'cost_inr': round(base_cost, 2),
        'estimated_on_time_pct': round(on_time_with_variance, 1),
        'carbon_kg': round(carbon_gco2, 2),
        'has_cold_chain': carrier.has_cold_chain,
        'has_ev_fleet': carrier.has_ev_fleet,
        'is_express': is_express
    }


def get_all_quotes(
    distance_km: float,
    weight_kg: float,
    package_type: str,
    origin_region: str,
    is_express: bool = False,
    requires_cold_chain: bool = False,
    is_metro_to_metro: bool = False,
    weather_condition: str = 'clear',
    min_on_time_pct: float = 85.0
) -> List[Dict]:
    """Get quotes from all eligible carriers."""
    eligible_carriers = get_carriers_for_route(
        package_type=package_type,
        weight_kg=weight_kg,
        origin_region=origin_region,
        requires_cold_chain=requires_cold_chain,
        min_on_time_pct=min_on_time_pct
    )
    quotes = []
    for carrier in eligible_carriers:
        quote = calculate_carrier_quote(
            carrier=carrier,
            distance_km=distance_km,
            weight_kg=weight_kg,
            is_express=is_express,
            requires_cold_chain=requires_cold_chain,
            is_metro_to_metro=is_metro_to_metro,
            weather_condition=weather_condition
        )
        if quote:
            quotes.append(quote)
    quotes.sort(key=lambda x: x['cost_inr'])
    return quotes
