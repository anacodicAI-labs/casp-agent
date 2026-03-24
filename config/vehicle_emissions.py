"""
Vehicle Emission Factors
Carbon emissions per km by vehicle type (gCO2/km).

Data is now loaded from data/reference/vehicle_emissions.csv instead of being hardcoded.

References (with links):
- CPCB India Two-Wheeler Emission Inventory 2023: https://cpcb.nic.in/
- BEE India EV Report 2023 (India grid-adjusted EV factors): https://beeindia.gov.in/
- ICCT India Freight Transport 2023: https://theicct.org/
- UK Government GHG conversion factors 2024 (EV van proxy): https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024
- Bektaş & Laporte (2011) Pollution-Routing Problem: https://doi.org/10.1016/j.trb.2011.02.004
- Demir et al. (2014) Green road freight review: https://doi.org/10.1016/j.ejor.2014.04.030
"""

import csv
import os


def _load_vehicle_emissions() -> dict:
    """Load vehicle emissions from CSV file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate from config/ to data/reference/
    csv_path = os.path.join(current_dir, '..', 'data', 'reference', 'vehicle_emissions.csv')
    
    vehicle_emissions = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vehicle_type = row['vehicle_type']
            gco2_per_km = float(row['gco2_per_km'])
            vehicle_emissions[vehicle_type] = gco2_per_km
    
    return vehicle_emissions


# Load vehicle emissions from CSV file
VEHICLE_EMISSIONS = _load_vehicle_emissions()

def get_vehicle_emission(vehicle_type: str) -> float:
    """Get carbon emission factor for vehicle type (gCO2/km)."""
    return VEHICLE_EMISSIONS.get(vehicle_type.lower(), 800)

def calculate_transport_carbon(
    distance_km: float,
    vehicle_type: str,
    cold_chain_multiplier: float = 1.0
) -> float:
    """
    Calculate transport carbon emissions.
    
    Args:
        distance_km: Distance in kilometers
        vehicle_type: Type of vehicle
        cold_chain_multiplier: Multiplier for cold chain (1.0 = no cold chain, 2.5 = cold chain)
    
    Returns:
        Carbon emissions in gCO2
    """
    base_emission = get_vehicle_emission(vehicle_type)
    return distance_km * base_emission * cold_chain_multiplier
