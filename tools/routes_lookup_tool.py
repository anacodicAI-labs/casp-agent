from __future__ import annotations
import json

from tools.stream_events import emit_subtool_event
from tools._common import _json_safe, tool


@tool
def routes_lookup_tool(origin: str, destination: str) -> str:
    emit_subtool_event("sourcing.distance", "start")
    try:
        from utils.sourcing_tools import lookup_route

        info = lookup_route(origin.strip(), destination.strip())
        if info is None:
            print(f"[ROUTES LOOKUP] ⚠️  no route found for {origin!r} → {destination!r}")
            return json.dumps({"distance_km": None, "region": "west", "is_metro_to_metro": False, "source": "none"})
        print(f"[ROUTES LOOKUP] ✅ {origin!r} → {destination!r} = {info.get('distance_km')} km (local table)")
        return json.dumps(_json_safe(info))
    except Exception as e:
        print(f"[ROUTES LOOKUP] ❌ {origin!r} → {destination!r} error: {e}")
        return json.dumps({"distance_km": None, "error": str(e)})

    finally:
        emit_subtool_event("sourcing.distance", "done")
