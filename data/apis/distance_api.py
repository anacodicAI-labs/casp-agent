"""
OpenRouteService API wrapper for Sourcing Agent.
Step 2 in distance cascade (after local routes.py); fallback to web search / haversine.
Free tier: 2000 requests/day. Get key: openrouteservice.org
"""

from typing import Optional
import urllib.parse
import urllib.request
import json


def get_distance_km(origin: str, destination: str, api_key: str) -> Optional[float]:
    """
    Get road distance in km between two places (OpenRouteService matrix).
    origin/destination: city names or "lat,lon" strings.
    Returns km or None on failure.
    """
    if not api_key or not origin or not destination:
        return None
    # ORS matrix needs coordinates; we don't geocode here - caller can pass coords
    # Try geocode first via ORS geocode endpoint
    try:
        # Geocode origin
        geo_origin = _geocode(origin, api_key)
        geo_dest = _geocode(destination, api_key)
        if not geo_origin or not geo_dest:
            return None
        lon1, lat1 = geo_origin
        lon2, lat2 = geo_dest
        # Distance matrix (driving)
        url = "https://api.openrouteservice.org/v2/matrix/driving-car"
        payload = {
            "locations": [[lon1, lat1], [lon2, lat2]],
            "metrics": ["distance"],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Authorization": api_key, "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        dist_m = (data.get("distances") or [[None]])[0][1]
        if dist_m is not None:
            return round(float(dist_m) / 1000.0, 2)
        return None
    except Exception:
        return None


def _geocode(place: str, api_key: str) -> Optional[tuple]:
    """Return (lon, lat) for place name."""
    try:
        url = "https://api.openrouteservice.org/geocode/search?" + urllib.parse.urlencode({"api_key": api_key, "text": place})
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        features = data.get("features") or []
        if not features:
            return None
        coords = features[0].get("geometry", {}).get("coordinates")
        if coords and len(coords) >= 2:
            return (float(coords[0]), float(coords[1]))
        return None
    except Exception:
        return None
