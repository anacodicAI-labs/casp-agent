"""
NewsAPI wrapper for Risk Agent.
Step 2 in risk cascade: "supply chain disruption India".
Free tier: 100 requests/day. Get key: newsapi.org
"""

from typing import Optional, List, Dict, Any
import urllib.parse
import urllib.request
import json


def search_news(query: str, api_key: str, country: Optional[str] = "in", max_results: int = 5) -> Optional[List[Dict[str, Any]]]:
    """
    Search news (NewsAPI). Returns list of articles or None on failure.
    Used for disruption signals in risk assessment.
    """
    if not api_key or not query:
        return None
    q = urllib.parse.quote(query.strip())
    url = f"https://newsapi.org/v2/top-headlines?q={q}&country={country or 'in'}&pageSize={max_results}&apiKey={api_key}"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        articles = data.get("articles") or []
        return [{"title": a.get("title"), "description": a.get("description"), "url": a.get("url")} for a in articles if a.get("title")]
    except Exception:
        return None
