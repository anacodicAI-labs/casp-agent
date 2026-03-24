"""
Python backend services for the supply chain pipeline.

These are NOT LLM agents. They are deterministic Python logic used by:
- run_optimization_tool, carbon_analysis_tool, get_carrier_options_tool (orchestration/agent_tools.py)
- SupplyChainOrchestrator

LLM agents (Strands) live in orchestration/: orchestrator_agent, risk_agent_strands, sourcing_agent_strands.
"""

from services.stakes_optimizer import StakesOptimizer
from services.risk_service import RiskService
from services.sourcing_service import SourcingService
from services.carbon_service import CarbonService

__all__ = [
    "StakesOptimizer",
    "RiskService",
    "SourcingService",
    "CarbonService",
]
