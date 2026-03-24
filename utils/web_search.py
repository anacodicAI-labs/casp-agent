"""
Web search wrapper for Risk/Sourcing fallback.
Uses Tavily as the single web search provider.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from typing import List, Optional

logger = logging.getLogger(__name__)


def _tavily_search(query: str, timeout: float = 15.0) -> Optional[str]:
    """
    Tavily Search API (recommended primary provider).
    Docs: https://docs.tavily.com/
    """
    key = (os.environ.get("TAVILY_API_KEY") or "").strip()
    if not key:
        return None

    body = json.dumps(
        {
            "api_key": key,
            "query": query,
            "search_depth": "basic",
            "max_results": 5,
            "include_answer": True,
            "include_raw_content": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "CASP-Agent/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        logger.debug("Tavily search failed for %r: %s", query[:80], e)
        return None

    parts: List[str] = []
    answer = data.get("answer")
    if isinstance(answer, str) and answer.strip():
        parts.append(answer.strip())
    for item in (data.get("results") or [])[:5]:
        if isinstance(item, dict):
            title = (item.get("title") or "").strip()
            content = (item.get("content") or "").strip()
            if title or content:
                parts.append(f"{title}: {content}".strip(": "))
    out = "\n".join(parts).strip()
    return out[:8000] if out else None


def web_search(query: str, enabled: bool = True) -> Optional[str]:
    """
    Run a web search and return a short snippet (for parsing weather/distance/pricing).

    Uses Tavily only (requires ``TAVILY_API_KEY``).
    """
    if not enabled or not query or not query.strip():
        return None

    env_off = os.environ.get("WEB_SEARCH_ENABLED", "true").strip().lower() in (
        "0",
        "false",
        "no",
    )
    if env_off:
        return None

    q = query.strip()

    tavily_out = _tavily_search(q)
    if tavily_out:
        return tavily_out[:8000]

    return None


def parse_distance_from_text(text: Optional[str]) -> Optional[float]:
    """Extract distance in km from search snippet (e.g. 'Mumbai to Delhi 1400 km')."""
    if not text:
        return None
    m = re.search(r"\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*km\b", text.replace(",", ""), re.IGNORECASE)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except (ValueError, TypeError):
            pass
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:kilometers?|kms?)\b", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            pass
    return None


def parse_cost_from_text(text: Optional[str]) -> Optional[float]:
    """Extract cost in INR from search snippet (e.g. '₹2100', 'Rs 1500', 'INR 2000')."""
    if not text:
        return None
    for pattern in [
        r"₹\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
        r"Rs?\.?\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
        r"INR\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
        r"(?:cost|rate|price)[:\s]*(\d+(?:,\d{3})*(?:\.\d+)?)",
    ]:
        m = re.search(pattern, text.replace(",", ""), re.IGNORECASE)
        if m:
            try:
                v = float(m.group(1).replace(",", ""))
                if 0 < v < 1e7:
                    return v
            except (ValueError, TypeError):
                pass
    return None


def parse_weather_from_text(text: Optional[str]) -> Optional[str]:
    """Map snippet to our weather_condition: clear, rainy, cold, hot, foggy, stormy."""
    if not text:
        return None
    t = text.lower()
    if any(x in t for x in ["storm", "thunder", "cyclone"]):
        return "stormy"
    if any(x in t for x in ["rain", "drizzle", "shower"]):
        return "rainy"
    if any(x in t for x in ["fog", "mist", "haze"]):
        return "foggy"
    if any(x in t for x in ["hot", "heat", "40", "45"]):
        return "hot"
    if any(x in t for x in ["cold", "snow", "freez"]):
        return "cold"
    if any(x in t for x in ["clear", "sunny", "partly"]):
        return "clear"
    return None
