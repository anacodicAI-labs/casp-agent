"""
Indian City Routes Database
Major logistics corridors with distances and characteristics

Data is now loaded from data/reference/cities.json and data/reference/routes.csv instead of being hardcoded.

References:
- Google Maps Distance Matrix API~\cite{googlemaps2024} for route distances
- Google Maps Geocoding API for city coordinates
- Indian Census data and administrative boundaries for regional classifications
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import csv
import os

@dataclass
class City:
    """City profile for logistics."""
    name: str
    code: str
    region: str  # north, south, east, west, central
    is_metro: bool
    state: str
    latitude: float
    longitude: float


def _load_cities() -> Dict[str, City]:
    """Load cities from JSON file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate from config/ to data/reference/
    json_path = os.path.join(current_dir, '..', 'data', 'reference', 'cities.json')
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cities = {}
    for key, city_data in data['cities'].items():
        cities[key] = City(**city_data)
    
    return cities


def _load_route_distances() -> Dict[Tuple[str, str], float]:
    """Load route distances from CSV file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate from config/ to data/reference/
    csv_path = os.path.join(current_dir, '..', 'data', 'reference', 'routes.csv')
    
    route_distances = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            origin = row['origin'].lower()
            destination = row['destination'].lower()
            distance = float(row['distance_km'])
            route_distances[(origin, destination)] = distance
    
    return route_distances


# Load cities and routes from data files
CITIES = _load_cities()
ROUTE_DISTANCES = _load_route_distances()


def get_city(city_code: str) -> Optional[City]:
    """Get city by code or name."""
    city_code = city_code.lower()
    if city_code in CITIES:
        return CITIES[city_code]
    for code, city in CITIES.items():
        if city.name.lower() == city_code:
            return city
    return None


def get_distance(origin: str, destination: str) -> Optional[float]:
    """Get distance between two cities in km."""
    origin = origin.lower()
    destination = destination.lower()
    if (origin, destination) in ROUTE_DISTANCES:
        return ROUTE_DISTANCES[(origin, destination)]
    if (destination, origin) in ROUTE_DISTANCES:
        return ROUTE_DISTANCES[(destination, origin)]
    origin_city = get_city(origin)
    dest_city = get_city(destination)
    if origin_city and dest_city:
        return _haversine_distance(
            origin_city.latitude, origin_city.longitude,
            dest_city.latitude, dest_city.longitude
        ) * 1.3
    return None


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance in km."""
    import math
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def is_metro_to_metro(origin: str, destination: str) -> bool:
    """Check if route is between two metro cities."""
    origin_city = get_city(origin)
    dest_city = get_city(destination)
    if origin_city and dest_city:
        return origin_city.is_metro and dest_city.is_metro
    return False


def get_route_info(origin: str, destination: str) -> Optional[Dict]:
    """Get complete route information."""
    origin_city = get_city(origin)
    dest_city = get_city(destination)
    distance = get_distance(origin, destination)
    if not origin_city or not dest_city or not distance:
        return None
    return {
        'origin': {'name': origin_city.name, 'code': origin_city.code, 'region': origin_city.region, 'is_metro': origin_city.is_metro, 'state': origin_city.state},
        'destination': {'name': dest_city.name, 'code': dest_city.code, 'region': dest_city.region, 'is_metro': dest_city.is_metro, 'state': dest_city.state},
        'distance_km': distance,
        'is_metro_to_metro': origin_city.is_metro and dest_city.is_metro,
        'same_region': origin_city.region == dest_city.region,
        'same_state': origin_city.state == dest_city.state
    }


def get_popular_routes() -> List[Dict]:
    """Get list of popular logistics routes."""
    popular = [
        ('delhi', 'mumbai'), ('delhi', 'bangalore'), ('mumbai', 'bangalore'),
        ('mumbai', 'chennai'), ('delhi', 'kolkata'), ('bangalore', 'chennai'),
        ('delhi', 'hyderabad'), ('mumbai', 'hyderabad'), ('delhi', 'ahmedabad'),
        ('mumbai', 'pune'),
    ]
    routes = []
    for origin, dest in popular:
        route_info = get_route_info(origin, dest)
        if route_info:
            routes.append(route_info)
    return routes
