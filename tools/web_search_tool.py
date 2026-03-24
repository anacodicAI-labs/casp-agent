from __future__ import annotations

import json

from tools._common import tool
from tools.stream_events import emit_subtool_event, get_stream_scope


@tool
def web_search_tool(queries: str) -> str:
    scope = get_stream_scope()
    event_id = "risk.web" if scope == "risk" else "sourcing.web"
    emit_subtool_event(event_id, "start")
    try:
        from utils.web_search import web_search as do_web_search

        if queries.strip().startswith("["):
            qlist = json.loads(queries)
        else:
            qlist = [queries.strip()]

        results = []
        for q in qlist:
            if not q:
                continue
            print(f"[WEB SEARCH] 🔍 query={q!r}")
            raw = do_web_search(q)
            snippet_preview = (raw or "")[:200]
            print(f"[WEB SEARCH] → snippet: {snippet_preview!r}" if raw else f"[WEB SEARCH] → (empty result)")
            results.append({"query": q, "snippet": raw or ""})
        return json.dumps({"results": results})
    except Exception as e:
        print(f"[WEB SEARCH] ❌ error: {e}")
        return json.dumps({"results": [], "error": str(e)})
    finally:
        emit_subtool_event(event_id, "done")

