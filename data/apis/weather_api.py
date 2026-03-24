"""
OpenWeatherMap API wrapper for Risk Agent.
Step 1 in risk cascade; fallback to web search / local if fail.
Free tier: 1000 calls/day. Get key: openweathermap.org/api
"""

from typing import Optional
import urllib.parse
import urllib.request
import json

# Normalize API response to our weather_condition: clear, rainy, cold, hot, foggy, stormy
OWM_TO_CONDITION = {
    "clear": "clear",
    "clouds": "clear",
    "drizzle": "rainy",
    "rain": "rainy",
    "thunderstorm": "stormy",
    "snow": "cold",
    "mist": "foggy",
    "fog": "foggy",
    "haze": "foggy",
    "extreme": "stormy",
}


def get_weather(city: str, api_key: str) -> Optional[str]:
    """
    Get current weather condition for a city (OpenWeatherMap).
    Returns one of: clear, rainy, cold, hot, foggy, stormy; or None on failure.
    """
    if not api_key or not city:
        return None
    city_enc = urllib.parse.quote(city.strip())
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_enc}&appid={api_key}&units=metric"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        main = (data.get("weather") or [{}])[0].get("main", "").lower()
        temp = (data.get("main") or {}).get("temp")
        condition = OWM_TO_CONDITION.get(main, "clear")
        if temp is not None:
            if temp >= 35:
                condition = "hot"
            elif temp <= 10:
                condition = "cold"
        return condition
    except Exception:
        return None
