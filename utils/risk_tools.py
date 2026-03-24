"""
Risk tools: wrap Module 07 (Early-Warning System) for the Risk Agent.
Cascade: OpenWeatherMap API → NewsAPI → Web search → Local CSV + Module 07 (fallback).
"""

from typing import Dict, List, Optional, Any


def _recommend_buffer_days(
    delay_prob: float,
    risk_level: str,
    weather_condition: str,
    package_type: str,
    route_dict: Dict[str, Any],
) -> int:
    """
    Context-aware deterministic fallback for buffer recommendation.
    Uses delay probability first, then upgrades for route/weather/package context,
    while enforcing a risk-level minimum floor.
    """
    level = str(risk_level or "LOW").upper()
    floor_map = {"CRITICAL": 2, "HIGH": 1, "MEDIUM": 1, "LOW": 0}
    min_buffer = floor_map.get(level, 0)

    # Continuous probability bands.
    if delay_prob > 0.30:
        rec = 3
    elif delay_prob > 0.15:
        rec = 2
    elif delay_prob > 0.05:
        rec = 1
    else:
        rec = 0

    # Context upgrades.
    w = str(weather_condition or "clear").lower()
    if w in {"stormy"}:
        rec = max(rec, 2)
    elif w in {"rainy", "foggy", "hot", "cold"}:
        rec = max(rec, 1)

    try:
        dist = float(route_dict.get("distance_km", 0) or 0)
    except (TypeError, ValueError):
        dist = 0
    if dist > 500:
        rec = max(rec, 1)

    if str(package_type or "").lower() in {"pharmacy", "groceries"} and rec > 0:
        rec += 1

    return max(min_buffer, rec)


def get_weather_for_risk(origin: str, destination: str) -> str:
    try:
        from config.api_config import API_KEYS, FALLBACK_ENABLED, WEB_SEARCH_ENABLED
    except ImportError:
        return "clear"
    weather = None
    key = (API_KEYS or {}).get("openweathermap", "")
    if key:
        try:
            from data.apis.weather_api import get_weather
            weather = get_weather(origin, key) or get_weather(destination, key)
        except Exception:
            pass
    if not weather and key and (API_KEYS or {}).get("newsapi"):
        try:
            from data.apis.news_api import search_news
            search_news("supply chain disruption India", (API_KEYS or {}).get("newsapi", ""))
        except Exception:
            pass
    if not weather and WEB_SEARCH_ENABLED:
        try:
            from utils.web_search import web_search, parse_weather_from_text
            raw = web_search(f"{origin} weather today")
            weather = parse_weather_from_text(raw) if raw else None
            if not weather:
                raw = web_search(f"{destination} weather today")
                weather = parse_weather_from_text(raw) if raw else None
        except Exception:
            pass
    if not weather or not FALLBACK_ENABLED:
        weather = "clear"
    return (weather or "clear").lower()


def get_disruption_news(region: str = "India") -> Optional[List[Dict[str, Any]]]:
    try:
        from config.api_config import API_KEYS, WEB_SEARCH_ENABLED
        key = (API_KEYS or {}).get("newsapi", "")
        if key:
            from data.apis.news_api import search_news
            return search_news(f"supply chain disruption {region}", key)
        if WEB_SEARCH_ENABLED:
            from utils.web_search import web_search
            raw = web_search(f"logistics news {region} today")
            return [{"title": raw}] if raw else None
    except Exception:
        pass
    return None


def assess_risk(
    ews,
    origin: str,
    destination: str,
    weather_condition: str,
    package_type: str,
    route_dict: Optional[Dict] = None,
    distance_km: Optional[float] = None,
    package_weight_kg: Optional[float] = None,
    delivery_partner: Optional[str] = None,
) -> Dict[str, Any]:
    if route_dict is None:
        route_dict = {
            "delivery_partner": delivery_partner or "delhivery",
            "package_type": package_type,
            "vehicle_type": "van",
            "delivery_mode": "express",
            "region": "west",
            "weather_condition": weather_condition or "clear",
            "distance_km": distance_km or 150,
            "package_weight_kg": package_weight_kg or 25,
            "delivery_rating": 4,
        }

    delay_prob = ews.predict_delay_probability(route_dict)
    risk = ews.calculate_risk_score(package_type, delay_prob, route_dict)
    early_warning_indicators = ews.compute_early_warning_indicators(package_type, route_dict=route_dict)

    recommended_buffer_days = _recommend_buffer_days(
        delay_prob=delay_prob,
        risk_level=risk.get("risk_level", "LOW"),
        weather_condition=weather_condition,
        package_type=package_type,
        route_dict=route_dict,
    )

    warnings = list(risk.get("risk_factors", []))
    if risk.get("alert_required"):
        warnings.append("Alert required: high disruption risk")

    return {
        "risk_level": risk["risk_level"],
        "risk_score": risk["risk_score"],
        "risk_factors": risk.get("risk_factors", []),
        "warnings": warnings,
        "delay_probability": delay_prob,
        "recommended_buffer_days": recommended_buffer_days,
        "alert_required": risk.get("alert_required", False),
        "early_warning_indicators": early_warning_indicators,
    }


def assess_risk_with_cascade(
    ews,
    origin: str,
    destination: str,
    package_type: str,
    route_dict: Optional[Dict] = None,
    distance_km: Optional[float] = None,
    package_weight_kg: Optional[float] = None,
    delivery_partner: Optional[str] = None,
    use_weather_cascade: bool = True,
) -> Dict[str, Any]:
    weather_condition = get_weather_for_risk(origin, destination) if use_weather_cascade else "clear"
    return assess_risk(
        ews,
        origin=origin,
        destination=destination,
        weather_condition=weather_condition,
        package_type=package_type,
        route_dict=route_dict,
        distance_km=distance_km,
        package_weight_kg=package_weight_kg,
        delivery_partner=delivery_partner,
    )

