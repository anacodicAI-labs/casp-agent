"""
Internal helpers shared by LLM-callable tool wrappers.

This module MUST NOT define any `@tool` itself; it is plumbing only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Ensure project root is cwd for imports like `config/`, `data/`, etc.
_CODE_DIR = Path(__file__).resolve().parent.parent
if os.getcwd() != str(_CODE_DIR):
    os.chdir(_CODE_DIR)


def _get_orchestrator():
    from agents._instance import get_orchestrator

    return get_orchestrator()


def _json_safe(obj: Any) -> Any:
    """Convert common numpy/pandas values into JSON-serializable values."""

    import numpy as np

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


try:
    from strands import tool  # type: ignore
except ImportError:  # pragma: no cover

    def tool(fn):  # type: ignore
        return fn

