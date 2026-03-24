from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Callable, Optional

_emitter_var: ContextVar[Optional[Callable[[dict[str, Any]], None]]] = ContextVar(
    "tool_stream_emitter", default=None
)
_scope_var: ContextVar[str] = ContextVar("tool_stream_scope", default="")


def set_stream_emitter(emitter: Optional[Callable[[dict[str, Any]], None]]) -> None:
    _emitter_var.set(emitter)


def set_stream_scope(scope: str) -> None:
    _scope_var.set(scope or "")


def get_stream_scope() -> str:
    return _scope_var.get() or ""


def emit_subtool_event(event_id: str, state: str, meta: Optional[dict[str, Any]] = None) -> None:
    emitter = _emitter_var.get()
    if emitter is None:
        return
    payload: dict[str, Any] = {"subtool_event": {"id": event_id, "state": state}}
    if isinstance(meta, dict) and meta:
        payload["subtool_event"]["meta"] = meta
    try:
        emitter(payload)
    except Exception:
        # Streaming must never fail tool execution.
        return
