"""
Grid Carbon Intensity by Country
National energy-transition conditions affecting AI compute carbon

Updated to be consistent with Module 03: Carbon Cost of Intelligence

Data is now loaded from data/reference/grid_carbon.json instead of being hardcoded.

References (with links):
- EPA eGRID 2023: https://www.epa.gov/egrid
- Electricity Maps: https://www.electricitymaps.com/
- Kaur et al. (2026) "The Carbon Cost of Intelligence": https://doi.org/10.3390/en19030642
- Patterson et al. (2021) "Carbon emissions and large neural network training": https://arxiv.org/abs/2104.10350
"""

import json
import os


def _load_grid_carbon_data():
    """Load grid carbon intensity data from JSON file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate from config/ to data/reference/
    json_path = os.path.join(current_dir, '..', 'data', 'reference', 'grid_carbon.json')
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract grid intensities into a simple dict
    grid_intensities = {}
    for country, country_data in data['grid_intensities'].items():
        grid_intensities[country] = country_data['intensity_gco2_kwh']
    
    return grid_intensities, data['region_to_country']


# Load grid carbon data from JSON file
_GRID_INTENSITIES, _REGION_MAPPING = _load_grid_carbon_data()
GRID_CARBON_INTENSITY = _GRID_INTENSITIES
REGION_TO_COUNTRY = _REGION_MAPPING

def get_grid_carbon_intensity(country: str = 'india') -> float:
    """
    Get grid carbon intensity for a country (gCO2/kWh).
    
    Args:
        country: Country name (case-insensitive)
    
    Returns:
        Carbon intensity in gCO2/kWh
    """
    return GRID_CARBON_INTENSITY.get(country.lower(), GRID_CARBON_INTENSITY['default'])

def get_country_from_region(region: str) -> str:
    """Map region to country (defaults to India for Indian regions)."""
    return REGION_TO_COUNTRY.get(region.lower(), 'india')


def _load_ai_model_energy():
    """Load AI model energy consumption data from JSON file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate from config/ to data/reference/
    json_path = os.path.join(current_dir, '..', 'data', 'reference', 'ai_model_energy.json')
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract model energy into a simple dict
    model_energy = {}
    for model, model_data in data['model_energy_wh'].items():
        model_energy[model] = model_data['energy_wh']
    
    return model_energy


# AI Model Energy Consumption (Wh per inference/request)
# Data loaded from data/reference/ai_model_energy.json
# Sources: Patterson et al. (2021) https://arxiv.org/abs/2104.10350,
#          Kaur et al. (2026) https://doi.org/10.3390/en19030642, internal benchmarks
MODEL_ENERGY_WH = _load_ai_model_energy()


def calculate_model_carbon(
    tokens: int = 500,
    model: str = 'gemini-flash',
    country: str = 'india'
) -> float:
    """
    Calculate carbon for AI model inference.
    
    Updated formula: Carbon = Energy_Wh / 1000 × Grid_Intensity
    
    Args:
        tokens: Number of tokens processed (used for scaling LLM requests)
        model: Model name
        country: Country for grid intensity
    
    Returns:
        Carbon emissions in gCO2
    """
    # Get base energy per inference
    energy_wh = MODEL_ENERGY_WH.get(model.lower(), MODEL_ENERGY_WH['default'])
    
    # Scale by tokens if it's an LLM (rough scaling)
    if model.lower() in ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo', 'gpt-3.5', 
                          'claude-3-opus', 'claude-3-sonnet', 'claude-3',
                          'gemini-pro', 'gemini-flash', 'gemini-2.5-flash']:
        # Energy scales roughly with sqrt(tokens) for transformers
        scaling_factor = (tokens / 500) ** 0.5  # Normalized to 500 tokens
        energy_wh *= scaling_factor
    
    # Convert to kWh and multiply by grid intensity
    energy_kwh = energy_wh / 1000
    grid_intensity = get_grid_carbon_intensity(country)
    
    return energy_kwh * grid_intensity


def calculate_ai_compute_carbon(
    energy_joules: float = None,
    energy_wh: float = None,
    country: str = 'india'
) -> float:
    """
    Calculate AI compute carbon based on grid intensity.
    
    Args:
        energy_joules: Energy consumption in Joules (legacy)
        energy_wh: Energy consumption in Watt-hours (preferred)
        country: Country name
    
    Returns:
        Carbon emissions in gCO2
    """
    grid_intensity = get_grid_carbon_intensity(country)
    
    if energy_wh is not None:
        energy_kwh = energy_wh / 1000
    elif energy_joules is not None:
        energy_kwh = energy_joules / 3_600_000  # Convert J to kWh
    else:
        raise ValueError("Either energy_joules or energy_wh must be provided")
    
    return energy_kwh * grid_intensity


def calculate_optimization_ai_carbon(
    num_routes: int = 3,
    model: str = 'gradient-boosting',
    country: str = 'india',
    predictions_per_route: int = 3  # cost, carbon, on-time
) -> float:
    """
    Calculate total AI carbon for a route optimization task.
    
    This is the function that should be called by the optimizer
    instead of hardcoding 50 gCO2.
    
    Args:
        num_routes: Number of routes being evaluated
        model: ML model used for predictions
        country: Country where compute runs
        predictions_per_route: Number of predictions per route (default: 3)
    
    Returns:
        Total AI carbon in gCO2
    """
    total_inferences = num_routes * predictions_per_route
    energy_wh = MODEL_ENERGY_WH.get(model.lower(), MODEL_ENERGY_WH['default'])
    total_energy_wh = energy_wh * total_inferences
    
    return calculate_ai_compute_carbon(energy_wh=total_energy_wh, country=country)


if __name__ == "__main__":
    # Test the module
    print("Grid Carbon Intensity Tests")
    print("=" * 50)
    
    # Test grid intensities
    countries = ['india', 'usa', 'france', 'germany', 'norway']
    print("\nGrid Intensities:")
    for country in countries:
        print(f"  {country:10s}: {get_grid_carbon_intensity(country)} gCO2/kWh")
    
    # Test model carbon
    print("\nAI Model Carbon (500 tokens, India):")
    models = ['gpt-4', 'gpt-3.5-turbo', 'gemini-flash', 'gradient-boosting']
    for model in models:
        carbon = calculate_model_carbon(500, model, 'india')
        print(f"  {model:20s}: {carbon:.6f} gCO2")
    
    # Test optimization carbon
    print("\nRoute Optimization Carbon (3 routes):")
    for country in ['india', 'usa', 'france']:
        carbon = calculate_optimization_ai_carbon(3, 'gradient-boosting', country)
        print(f"  {country:10s}: {carbon:.8f} gCO2")
    
    # Key finding
    print("\n" + "=" * 50)
    print("KEY FINDING: AI carbon is NEGLIGIBLE")
    print("=" * 50)
    transport = 150000  # Typical route carbon
    ai = calculate_optimization_ai_carbon(3, 'gradient-boosting', 'india')
    print(f"Transport carbon: {transport:,} gCO2")
    print(f"AI carbon:        {ai:.8f} gCO2")
    print(f"Ratio:            AI is {transport/ai:,.0f}× smaller than transport")
