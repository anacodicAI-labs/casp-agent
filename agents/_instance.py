"""
Singleton accessor for the heavy Python (non-LLM) orchestrator.

This keeps initialization (model training, data loading) lazy and shared between
API requests and tool calls.
"""

from __future__ import annotations

from typing import Optional

_orch_singleton = None


def get_orchestrator(data_path: Optional[str] = None):
    global _orch_singleton
    if _orch_singleton is None:
        from services.orchestrator import SupplyChainOrchestrator

        _orch_singleton = SupplyChainOrchestrator(data_path=data_path or "data/datasets/Delivery_Logistics.csv")
    return _orch_singleton

