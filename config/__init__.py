"""
Configuration Module
"""

from config.agent_mapping import AGENT_MAPPING, get_agent_config, get_stakes_level
from config.vehicle_emissions import VEHICLE_EMISSIONS, get_vehicle_emission, calculate_transport_carbon
from config.grid_carbon import (
    GRID_CARBON_INTENSITY,
    get_grid_carbon_intensity,
    calculate_ai_compute_carbon,
    calculate_model_carbon,
    get_country_from_region
)
from config.carriers import (
    CARRIERS,
    CarrierProfile,
    get_carrier,
    get_carriers_for_route,
    get_all_quotes,
    calculate_carrier_quote,
)
from config.routes import (
    CITIES,
    City,
    ROUTE_DISTANCES,
    get_city,
    get_distance,
    get_route_info,
    get_popular_routes,
    is_metro_to_metro,
)

__all__ = [
    'AGENT_MAPPING',
    'get_agent_config',
    'get_stakes_level',
    'VEHICLE_EMISSIONS',
    'get_vehicle_emission',
    'calculate_transport_carbon',
    'GRID_CARBON_INTENSITY',
    'get_grid_carbon_intensity',
    'calculate_ai_compute_carbon',
    'calculate_model_carbon',
    'get_country_from_region',
    'CARRIERS',
    'CarrierProfile',
    'get_carrier',
    'get_carriers_for_route',
    'get_all_quotes',
    'calculate_carrier_quote',
    'CITIES',
    'City',
    'ROUTE_DISTANCES',
    'get_city',
    'get_distance',
    'get_route_info',
    'get_popular_routes',
    'is_metro_to_metro',
]
