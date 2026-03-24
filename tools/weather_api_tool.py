from __future__ import annotations
import json

from tools.stream_events import emit_subtool_event
from tools._common import tool


@tool
def weather_api_tool(city: str) -> str:
    emit_subtool_event("risk.weather", "start")
    try:
        from config.api_config import API_KEYS

        key = (API_KEYS or {}).get("openweathermap", "")
        if not key:
            print(f"[WEATHER API] ❌ no_api_key for city={city!r}")
            return json.dumps({"weather_condition": None, "source": "api", "error": "no_api_key"})

        from data.apis.weather_api import get_weather

        condition = get_weather(city.strip(), key)
        print(f"[WEATHER API] ✅ city={city!r} → condition={condition!r}")
        return json.dumps({"weather_condition": condition, "source": "api", "city": city})
    except Exception as e:
        print(f"[WEATHER API] ❌ city={city!r} error: {e}")
        return json.dumps({"weather_condition": None, "source": "api", "error": str(e)})

    finally:
        emit_subtool_event("risk.weather", "done")
