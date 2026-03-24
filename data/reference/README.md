# Data Files Directory

This directory contains all reference/config data files separated from code, following industry best practices.

## Files

### `carriers.json`
- **Description**: Indian logistics carrier profiles with performance metrics, pricing, capabilities, and carbon data
- **Format**: JSON
- **Count**: 10 carriers
- **Fields**: name, code, type, avg_on_time_pct, base_rate_per_kg_km, carbon_gco2_per_km, etc.
- **Sources**: Industry reports and carrier public information (2024), carrier websites and service documentation, Indian logistics market analysis reports
- **Note**: Carrier performance metrics compiled from public carrier information, industry reports, and market analysis. Carbon factors derived from fleet composition data where available.

### `cities.json`
- **Description**: Major Indian cities with logistics metadata (region, metro status, coordinates)
- **Format**: JSON
- **Count**: 22 cities
- **Fields**: name, code, region, is_metro, state, latitude, longitude
- **Sources**: Google Maps Geocoding API, Indian Census data and administrative boundaries, logistics industry geographic classifications
- **Note**: City coordinates from Google Maps Geocoding API. Regional classifications and metro status based on Indian administrative boundaries and logistics industry standards.

### `routes.csv`
- **Description**: Route distances between Indian cities
- **Format**: CSV
- **Count**: 33 routes
- **Fields**: origin, destination, distance_km, source
- **Sources**: Google Maps Distance Matrix API
- **Citation**: See `googlemaps2024` in bibliography

### `grid_carbon.json`
- **Description**: Grid carbon intensity by country (gCO2/kWh)
- **Format**: JSON
- **Count**: 12 countries + default
- **Fields**: intensity_gco2_kwh, _source, _note
- **Sources**: 
  - EPA eGRID 2023~\cite{epa2025egrid}
  - Electricity Maps 2024~\cite{electricitymaps2025}
  - Kaur et al. (2026)~\cite{kaur2025cci}
  - Patterson et al. (2021)~\cite{patterson2021carbon}
- **Citations**: See bibliography entries `epa2025egrid`, `electricitymaps2025`, `kaur2025cci`, `patterson2021carbon`

### `ai_model_energy.json`
- **Description**: AI model energy consumption (Wh per inference/request)
- **Format**: JSON
- **Count**: 15 models
- **Fields**: energy_wh, _note, _source
- **Sources**: 
  - Patterson et al. (2021)~\cite{patterson2021carbon}
  - Kaur et al. (2026)~\cite{kaur2025cci}
  - Internal benchmarks
- **Citations**: See bibliography entries `patterson2021carbon`, `kaur2025cci`

### `vehicle_emissions.csv`
- **Description**: Vehicle emission factors (gCO2/km)
- **Format**: CSV
- **Count**: 6 vehicle types
- **Fields**: vehicle_type, gco2_per_km, source
- **Sources**: 
  - UK Government GHG conversion factors 2024~\cite{ukgov2024ghg}
  - DEFRA road transport methodology~\cite{defra2011road}
  - Bektaş \& Laporte (2011)~\cite{bektas2011role}
  - Demir et al. (2014)~\cite{demir2014review}
- **Citations**: See bibliography entries `ukgov2024ghg`, `defra2011road`, `bektas2011role`, `demir2014review`

## Benefits

1. **Reproducibility**: Reviewers can inspect data files without reading Python code
2. **Source Tracing**: Every data point includes source citations
3. **Versioning**: Data changes show up clearly in git diffs
4. **Reusability**: Other tools (R, Excel, another paper) can read these files
5. **Separation of Concerns**: Code logic and data don't mix
6. **Paper Credibility**: Can cite data sources explicitly

## Loading

All data files are automatically loaded by their respective Python modules:
- `code/config/carriers.py` → loads `carriers.json`
- `code/config/routes.py` → loads `cities.json` and `routes.csv`
- `code/config/grid_carbon.py` → loads `grid_carbon.json` and `ai_model_energy.json`
- `code/config/vehicle_emissions.py` → loads `vehicle_emissions.csv`

Function signatures remain identical - no changes needed in code that imports these modules.
