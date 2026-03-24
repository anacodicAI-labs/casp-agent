"""
External API wrappers for Risk/Sourcing cascades.
OpenWeatherMap, NewsAPI, OpenRouteService.
"""

from data.apis.weather_api import get_weather
from data.apis.news_api import search_news
from data.apis.distance_api import get_distance_km

__all__ = ["get_weather", "search_news", "get_distance_km"]
