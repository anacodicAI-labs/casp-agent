from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field


class ExtractRequest(BaseModel):
    query: str = Field(..., description="Natural language shipment description")


class ExtractResponse(BaseModel):
    features: Dict[str, Any] = Field(..., description="Full feature dict (extracted + defaults)")
    defaults_used: List[str] = Field(..., description="Keys that were filled from data-derived defaults")


class OptimizeRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Feature dict (from extract or after user edit)")


class ChatRequest(BaseModel):
    message: Optional[str] = Field(default="", description="Natural language shipment request")
    phase: Literal["gather", "optimize"] = Field(default="gather", description="Chat workflow phase")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Editable gathered context for optimize phase")

