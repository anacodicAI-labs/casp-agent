"""
API keys and feature flags for Risk/Sourcing cascades.
Use env vars or .env for keys in production; placeholder keys here.
"""

import os
from typing import Dict

# API keys (set via env: OPENWEATHERMAP_API_KEY, NEWSAPI_API_KEY, OPENROUTESERVICE_API_KEY)
API_KEYS: Dict[str, str] = {
    "openweathermap": os.environ.get("OPENWEATHERMAP_API_KEY", ""),
    "newsapi": os.environ.get("NEWSAPI_API_KEY", ""),
    "openrouteservice": os.environ.get("OPENROUTESERVICE_API_KEY", ""),
}

FALLBACK_ENABLED = os.environ.get("FALLBACK_ENABLED", "true").strip().lower() in ("1", "true", "yes")
WEB_SEARCH_ENABLED = os.environ.get("WEB_SEARCH_ENABLED", "true").strip().lower() in ("1", "true", "yes")
