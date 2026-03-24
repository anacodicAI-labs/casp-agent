from __future__ import annotations

import json

from tools._common import _json_safe, tool


@tool
def extract_features_tool(query: str) -> str:
    from utils.extraction_tools import extract_from_query_and_merge_defaults

    features, defaults_used = extract_from_query_and_merge_defaults(query)
    return json.dumps(_json_safe({"features": features, "defaults_used": defaults_used}))

