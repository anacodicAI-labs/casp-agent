from __future__ import annotations
import json

from tools.stream_events import emit_subtool_event
from tools._common import tool


@tool
def news_api_tool(query: str) -> str:
    emit_subtool_event("risk.news", "start")
    try:
        from config.api_config import API_KEYS

        key = (API_KEYS or {}).get("newsapi", "")
        if not key:
            print(f"[NEWS API] ❌ no_api_key for query={query!r}")
            return json.dumps({"articles": [], "source": "api", "error": "no_api_key"})

        from data.apis.news_api import search_news

        articles = search_news(query.strip(), key)
        out = [{"title": a.get("title"), "description": a.get("description"), "url": a.get("url")} for a in (articles or [])]
        print(f"[NEWS API] ✅ query={query!r} → {len(out)} articles")
        for a in out[:3]:
            print(f"  • {a.get('title', '')[:80]}")
        return json.dumps({"articles": out, "source": "api"})
    except Exception as e:
        print(f"[NEWS API] ❌ query={query!r} error: {e}")
        return json.dumps({"articles": [], "source": "api", "error": str(e)})

    finally:
        emit_subtool_event("risk.news", "done")
