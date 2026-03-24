from __future__ import annotations
import json

from tools.stream_events import emit_subtool_event
from tools._common import tool


@tool
def distance_api_tool(origin: str, destination: str) -> str:
    emit_subtool_event("sourcing.distance", "start")
    try:
        from config.api_config import API_KEYS

        key = (API_KEYS or {}).get("openrouteservice", "")
        if not key:
            print(f"[DISTANCE API] ❌ no_api_key for {origin!r} → {destination!r}")
            return json.dumps({"distance_km": None, "source": "api", "error": "no_api_key"})

        from data.apis.distance_api import get_distance_km

        d = get_distance_km(origin.strip(), destination.strip(), key)
        print(f"[DISTANCE API] ✅ {origin!r} → {destination!r} = {d} km")
        return json.dumps({"distance_km": d, "source": "openrouteservice"})
    except Exception as e:
        print(f"[DISTANCE API] ❌ {origin!r} → {destination!r} error: {e}")
        return json.dumps({"distance_km": None, "source": "api", "error": str(e)})

    finally:
        emit_subtool_event("sourcing.distance", "done")
