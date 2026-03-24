import json
import os
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse

from api.models import ChatRequest, ExtractRequest, ExtractResponse, OptimizeRequest

router = APIRouter()

_CODE_DIR = Path(__file__).resolve().parent.parent
_DATA_PATH = str(_CODE_DIR / "data" / "datasets" / "Delivery_Logistics.csv")


def _get_orchestrator():
    # Lazy singleton orchestrator (heavy init on first /optimize)
    from agents._instance import get_orchestrator

    return get_orchestrator(data_path=_DATA_PATH)


def _json_safe(obj: Any) -> Any:
    import math
    import numpy as np

    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        f = float(obj)
        return None if math.isnan(f) else f
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    try:
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            try:
                clean = obj.where(pd.notnull(obj), None)
            except Exception:
                clean = obj.fillna(value=None)
            return _json_safe(clean.to_dict(orient="records"))
        if isinstance(obj, pd.Series):
            try:
                clean = obj.where(pd.notnull(obj), None)
            except Exception:
                clean = obj.fillna(value=None)
            return _json_safe(clean.to_dict())
    except ImportError:
        pass
    return obj


@router.get("/")
async def root():
    """Serve the Option 2 UI (single page)."""
    html_path = _CODE_DIR / "frontend" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=open(html_path, encoding="utf-8").read())
    return PlainTextResponse(
        "Option 2 API. Use POST /api/extract and POST /api/optimize. Mount frontend at / if index.html exists."
    )


@router.post("/api/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest) -> ExtractResponse:
    try:
        from utils.extraction_tools import extract_from_query_and_merge_defaults

        features, defaults_used = extract_from_query_and_merge_defaults(request.query.strip())
        return ExtractResponse(features=features, defaults_used=defaults_used)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/api/optimize")
async def optimize(request: OptimizeRequest) -> Dict[str, Any]:
    try:
        from agents.expert_consultation import run_expert_consultation

        result = run_expert_consultation(request.features)
        try:
            return _json_safe(result)
        except Exception as serr:
            raise HTTPException(status_code=500, detail=f"Response serialization failed: {serr}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/consult-and-optimize")
async def consult_and_optimize(request: OptimizeRequest) -> Dict[str, Any]:
    """Primary expert-simulation endpoint for Control Tower runs."""
    return await optimize(request)


@router.get("/api/health")
async def health():
    return {"status": "ok"}


def _chat_available() -> bool:
    try:
        from strands import Agent  # noqa: F401

        return True
    except ImportError:
        return False


@router.post("/api/chat")
async def chat(request: ChatRequest) -> Dict[str, Any]:
    try:
        from agents.expert_consultation import run_chat_gather, run_chat_optimize

        if request.phase == "gather":
            if not (request.message or "").strip():
                raise HTTPException(status_code=422, detail="message is required for gather phase")
            return _json_safe(run_chat_gather(request.message.strip()))
        if request.phase == "optimize":
            return _json_safe(run_chat_optimize(request.context))
        raise HTTPException(status_code=422, detail="Unsupported chat phase")
    except Exception as e:
        error_msg = str(e)
        if "NoCredentialsError" in error_msg or "Unable to locate credentials" in error_msg:
            error_msg = (
                "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
                "environment variables or configure ~/.aws/credentials. See DEPLOYMENT.md for details."
            )
        elif "AccessDenied" in error_msg or "UnauthorizedOperation" in error_msg:
            error_msg = (
                "AWS access denied. Please ensure your AWS credentials have bedrock:InvokeModel permission. "
                "See DEPLOYMENT.md for IAM policy requirements."
            )
        elif "region" in error_msg.lower() or "Region" in error_msg:
            error_msg = (
                f"AWS region issue: {error_msg}. "
                "Please set AWS_REGION environment variable (e.g., us-east-1). "
                "See DEPLOYMENT.md for supported regions."
            )
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/api/chat/stream-gather")
async def chat_stream_gather(message: str):
    """
    SSE endpoint for gather-phase progressive updates.
    """
    from agents.expert_consultation import run_chat_gather_stream

    def event_gen():
        for evt in run_chat_gather_stream(message):
            # Must sanitize numpy/pandas in tool_result payloads or json.dumps crashes and the stream dies.
            yield f"data: {json.dumps(_json_safe(evt))}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.get("/api/chat/stream-full")
async def chat_stream_full(message: str):
    """
    SSE endpoint for full pipeline updates (gather + optimize + synthesis).
    """
    from agents.expert_consultation import run_chat_full_stream

    def event_gen():
        for evt in run_chat_full_stream(message):
            yield f"data: {json.dumps(_json_safe(evt))}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.post("/api/chat/stream-optimize")
async def chat_stream_optimize(request: Request):
    """
    SSE endpoint for optimize-only reruns from edited context.
    """
    from agents.expert_consultation import run_chat_optimize_stream

    body = await request.json()
    context = body.get("context") if isinstance(body, dict) else {}

    def event_gen():
        for evt in run_chat_optimize_stream(context):
            yield f"data: {json.dumps(_json_safe(evt))}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")

