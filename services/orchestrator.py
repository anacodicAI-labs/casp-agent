"""Control-tower supervisor orchestration for expert-simulation logistics flow."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

from analytics.carbon_intelligence import CarbonCostOfIntelligence
from analytics.trade_off_frontiers import TradeOffFrontiers
from ml.early_warning import EarlyWarningSystem
from ml.predictive_analytics import create_predictors
from ml.vendor_segmentation import VendorSegmentation
from services.carbon_service import CarbonService
from services.governance import GovernanceLevers
from services.risk_service import RiskService
from services.sourcing_service import SourcingService
from services.stakes_optimizer import StakesOptimizer
from utils.extraction_tools import extract_features_from_dict


class SupplyChainOrchestrator:
    @staticmethod
    def _priority_for_package(package_type: str) -> str:
        p = (package_type or "").strip().lower()
        return "carbon" if p in {"clothing", "cosmetics"} else "cost"

    """Control Tower Supervisor for the expert-desk simulation pipeline."""

    def __init__(self, data_path: str = "data/datasets/Delivery_Logistics.csv"):
        self.stakes_optimizer = StakesOptimizer()

        print("🔧 Initializing ML predictors...")
        self.cost_predictor, self.on_time_predictor, self.analytics = create_predictors(data_path)

        print("🔧 Initializing early-warning system...")
        self.ews = EarlyWarningSystem(data_path)
        self.ews.load_data()
        self.ews.train_delay_predictor()

        self.governance = GovernanceLevers()
        self.frontiers = TradeOffFrontiers()

        print("🔧 Initializing vendor segmentation...")
        self.vendor_segmentation = VendorSegmentation(data_path)
        self.vendor_segmentation.load_data()
        self.vendor_segmentation.prepare_vendor_features()
        self.vendor_segmentation.cluster_vendors(n_clusters=4)

        self.carbon_intel = CarbonCostOfIntelligence(default_country="india")

        self.risk_service = RiskService(self.ews)
        self.sourcing_service = SourcingService(
            self.cost_predictor,
            self.on_time_predictor,
            self.analytics,
            self.vendor_segmentation,
        )
        self.carbon_service = CarbonService(
            self.carbon_intel,
            self.frontiers,
            self.governance,
            country="india",
        )
        print("✅ Orchestrator initialized!")

    def route_to_agent(self, package_type: str):
        return self.stakes_optimizer

    @staticmethod
    def _expert_packet(
        desk: str,
        status: str,
        finding: str,
        constraints_proposed: Optional[Dict[str, Any]] = None,
        evidence: Optional[List[str]] = None,
        confidence: float = 0.0,
        insufficient_evidence: bool = False,
        fallback_used: bool = False,
        notes: str = "",
    ) -> Dict[str, Any]:
        return {
            "desk": desk,
            "status": status,
            "finding": finding,
            "constraints_proposed": constraints_proposed or {},
            "evidence": evidence or [],
            "confidence": round(float(confidence), 2),
            "insufficient_evidence": bool(insufficient_evidence),
            "fallback_used": bool(fallback_used),
            "notes": notes,
        }

    def _run_sourcing_benchmark(
        self,
        features: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        package_type: str,
        origin: str,
        destination: str,
        distance_km: Optional[float],
        package_weight_kg: Optional[float],
    ) -> Dict[str, Any]:
        benchmark_cost_inr = None
        benchmark_source = None
        benchmark_query = None
        benchmark_reasoning = None
        llm_recommendation = None
        llm_status = "unavailable"
        llm_error = None
        llm_obj: Dict[str, Any] = {}
        llm_carrier_options: List[Dict[str, Any]] = []

        llm_enabled = os.environ.get("LLM_BENCHMARK_ENABLED", "true").strip().lower() in ("1", "true", "yes")
        if llm_enabled:
            try:
                from agents.sourcing import run_sourcing_agent

                dist = distance_km if distance_km is not None else 150
                weight = package_weight_kg if package_weight_kg is not None else 25
                sourcing_queries = [
                    f"delivery cost {package_type} {dist}km {weight}kg India",
                    f"cold chain {package_type} delivery cost {origin} to {destination}",
                ]
                llm_out = run_sourcing_agent(
                    features_json=json.dumps(features),
                    risk_assessment_json=json.dumps(risk_assessment),
                    sourcing_queries_json=json.dumps(sourcing_queries),
                )
                llm_obj = json.loads(llm_out) if llm_out else {}
                llm_status = "available"
                if isinstance(llm_obj, dict):
                    raw_opts = llm_obj.get("carrier_options")
                    if isinstance(raw_opts, list):
                        # Accept only options that contain a route dict so planning can optimize.
                        llm_carrier_options = [o for o in raw_opts if isinstance(o, dict) and isinstance(o.get("route"), dict)]
                    llm_recommendation = llm_obj.get("recommendation")
                    benchmark_reasoning = llm_obj.get("benchmark_reasoning")
                    llm_b = llm_obj.get("industry_benchmark")
                    if llm_b is not None:
                        try:
                            benchmark_cost_inr = float(llm_b)
                            benchmark_source = "market_rate_llm"
                        except (TypeError, ValueError):
                            pass
            except Exception as exc:  # pragma: no cover
                llm_status = "unavailable"
                llm_error = str(exc)

        if benchmark_cost_inr is None:
            try:
                from utils.web_search import parse_cost_from_text, web_search

                dist = distance_km if distance_km is not None else 150
                weight = package_weight_kg if package_weight_kg is not None else 25
                benchmark_query = f"delivery cost {package_type} {dist}km {weight}kg India"
                snippet = web_search(benchmark_query)
                if snippet:
                    parsed = parse_cost_from_text(snippet)
                    if parsed is not None and parsed > 0:
                        benchmark_cost_inr = float(parsed)
                        benchmark_source = "market_rate_web"
            except Exception:
                pass

        if benchmark_cost_inr is None:
            try:
                import pandas as pd

                csv_path = "data/datasets/Delivery_Logistics.csv"
                df = pd.read_csv(csv_path)
                if "delivery_cost" in df.columns:
                    subset = df
                    if "package_type" in df.columns:
                        subset = df[df["package_type"].astype(str).str.lower().str.strip() == str(package_type).lower().strip()]
                    series = subset["delivery_cost"] if len(subset) else df["delivery_cost"]
                    series = series.dropna()
                    if len(series):
                        mean_val = float(series.mean())
                        if mean_val > 0:
                            benchmark_cost_inr = mean_val
                            benchmark_source = "historical_dataset"
            except Exception:
                pass

        return {
            "benchmark_cost_inr": benchmark_cost_inr,
            "benchmark_source": benchmark_source,
            "benchmark_query": benchmark_query,
            "benchmark_reasoning": benchmark_reasoning,
            "llm_recommendation": llm_recommendation,
            "llm_status": llm_status,
            "llm_error": llm_error,
            "llm_obj": llm_obj,
            "llm_carrier_options": llm_carrier_options,
        }

    def _build_risk_packet(
        self,
        features: Dict[str, Any],
        package_type: str,
        origin: str,
        destination: str,
        weather: str,
        distance_km: Optional[float],
        package_weight_kg: Optional[float],
    ) -> Dict[str, Any]:
        risk_assessment = self.risk_service.assess(
            origin=origin,
            destination=destination,
            weather_condition=weather,
            package_type=package_type,
            distance_km=distance_km,
            package_weight_kg=package_weight_kg,
        )

        llm_status = "unavailable"
        llm_error = None
        buffer_rationale = None
        llm_risk = None
        llm_enabled = os.environ.get("LLM_BENCHMARK_ENABLED", "true").strip().lower() in ("1", "true", "yes")
        if llm_enabled:
            try:
                from agents.risk import run_risk_agent

                risk_queries = [
                    f"{origin} weather today",
                    f"{destination} weather today",
                    f"{origin} logistics disruption",
                    f"{origin} {destination} lane disruption",
                ]
                llm_txt = run_risk_agent(json.dumps(features), json.dumps(risk_queries))
                llm_risk = json.loads(llm_txt) if llm_txt else {}
                llm_status = "available"
                if isinstance(llm_risk, dict):
                    llm_buffer = llm_risk.get("recommended_buffer_days")
                    if llm_buffer is not None:
                        try:
                            risk_assessment["recommended_buffer_days"] = max(
                                int(risk_assessment.get("recommended_buffer_days", 0)),
                                int(llm_buffer),
                            )
                        except (TypeError, ValueError):
                            pass
                    buffer_rationale = llm_risk.get("buffer_rationale")
            except Exception as exc:
                llm_status = "unavailable"
                llm_error = str(exc)

        risk_assessment["buffer_rationale"] = buffer_rationale or risk_assessment.get("buffer_rationale")
        packet = self._expert_packet(
            desk="Operations Risk Desk",
            status="ok",
            finding=f"Risk level {risk_assessment.get('risk_level', 'LOW')} with "
            f"recommended buffer {risk_assessment.get('recommended_buffer_days', 0)} day(s).",
            constraints_proposed={
                "buffer_days": int(risk_assessment.get("recommended_buffer_days", 0) or 0),
            },
            evidence=[
                "weather_api(origin+destination)",
                "news_api(disruptions)",
                "web_search(regional alerts)",
            ],
            confidence=0.82 if llm_status == "available" else 0.68,
            insufficient_evidence=False,
            fallback_used=llm_status != "available",
            notes=buffer_rationale or (f"LLM unavailable: {llm_error}" if llm_error else "Deterministic risk path used."),
        )
        return {
            "risk_assessment": risk_assessment,
            "packet": packet,
            "llm_status": llm_status,
            "llm_error": llm_error,
        }

    @staticmethod
    def _filter_routes_by_constraints(route_options: List[Dict[str, Any]], constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        filtered = deepcopy(route_options)
        allowed_vehicles = constraints.get("allowed_vehicle_types") or []
        if allowed_vehicles:
            allowed_norm = {str(v).strip().lower() for v in allowed_vehicles}
            filtered = [r for r in filtered if str(r.get("vehicle_type", "")).strip().lower() in allowed_norm]
        preferred_carriers = constraints.get("preferred_carrier_codes") or []
        if preferred_carriers:
            allowed_codes = {str(c).strip().lower() for c in preferred_carriers}
            by_code = [r for r in filtered if str(r.get("route_id", "")).strip().lower() in allowed_codes]
            if by_code:
                filtered = by_code
        return filtered

    def _optimize_with_constraints(
        self,
        package_type: str,
        route_options: List[Dict[str, Any]],
        origin: str,
        destination: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        constraints = constraints or {}
        filtered_routes = self._filter_routes_by_constraints(route_options, constraints)
        if not filtered_routes:
            filtered_routes = route_options

        min_reliability_pct = constraints.get("min_reliability_pct")
        buffer_days = float(constraints.get("buffer_days", 0) or 0)
        uplift_pct = min(5.0, max(0.0, buffer_days * 1.0))

        def constrained_on_time(route: Dict[str, Any]) -> float:
            base = float(self.on_time_predictor(route))
            adjusted = min(100.0, base + uplift_pct)
            if min_reliability_pct is not None and adjusted < float(min_reliability_pct):
                return adjusted
            return adjusted

        priority = self._priority_for_package(package_type)
        res = self.stakes_optimizer.optimize_route(
            package_type=package_type,
            route_options=filtered_routes,
            cost_predictor=self.cost_predictor,
            on_time_predictor=constrained_on_time if (uplift_pct > 0 or min_reliability_pct is not None) else self.on_time_predictor,
            origin=origin,
            destination=destination,
            priority=priority,
        )
        return res

    @staticmethod
    def _metric_delta(new: float, base: float) -> float:
        if base == 0:
            return 0.0
        return ((new - base) / base) * 100.0

    def run_integrated_pipeline(self, features: Dict[str, Any]) -> Dict[str, Any]:
        features = extract_features_from_dict(features)
        package_type = (features.get("package_type") or "clothing").lower()
        origin = str(features.get("origin") or "mumbai").lower()
        destination = str(features.get("destination") or "delhi").lower()
        weather = (features.get("weather_condition") or "clear").lower()
        distance_km = features.get("distance_km")
        package_weight_kg = features.get("package_weight_kg")

        risk_run = self._build_risk_packet(features, package_type, origin, destination, weather, distance_km, package_weight_kg)
        risk_assessment = risk_run["risk_assessment"]

        benchmark = self._run_sourcing_benchmark(
            features=features,
            risk_assessment=risk_assessment,
            package_type=package_type,
            origin=origin,
            destination=destination,
            distance_km=distance_km,
            package_weight_kg=package_weight_kg,
        )

        # Primary source: LLM Sourcing Agent packet (expert desk).
        # Fallback: deterministic sourcing service if LLM returned no valid options.
        llm_carrier_options = benchmark.get("llm_carrier_options") or []
        used_llm_sourcing_options = len(llm_carrier_options) > 0
        carrier_options = llm_carrier_options or self.sourcing_service.get_carrier_options(features, risk_assessment)
        if not carrier_options:
            return {
                "recommendation": None,
                "error": "No carrier options found for this shipment",
                "execution_mode": "deterministic_fallback",
                "llm_status": "unavailable",
                "carrier_options": [],
            }

        route_options = [o["route"] for o in carrier_options]

        constraints = {
            "buffer_days": int(risk_assessment.get("recommended_buffer_days", 0) or 0),
        }
        if package_type in {"pharmacy", "groceries"}:
            constraints["min_reliability_pct"] = 99.0

        if benchmark.get("llm_obj", {}).get("constraints_proposed"):
            constraints.update(benchmark["llm_obj"]["constraints_proposed"])

        baseline_result = self._optimize_with_constraints(
            package_type=package_type,
            route_options=route_options,
            origin=origin,
            destination=destination,
            constraints={},
        )
        final_result = self._optimize_with_constraints(
            package_type=package_type,
            route_options=route_options,
            origin=origin,
            destination=destination,
            constraints=constraints,
        )

        best_route = final_result.get("best_route", route_options[0] if route_options else {})
        delay_prob = self.ews.predict_delay_probability(best_route)
        risk = self.ews.calculate_risk_score(package_type, delay_prob, best_route)
        final_result["early_warning"] = risk
        final_result["governance"] = self.governance.generate_policy_recommendations(
            package_type,
            final_result,
            simulation_context={
                "route_options": route_options,
                "cost_predictor": self.cost_predictor,
                "on_time_predictor": self.on_time_predictor,
                "origin": origin,
                "destination": destination,
                "priority": self._priority_for_package(package_type),
            },
        )

        breakdown = final_result.get("breakdown", {})
        baseline_breakdown = baseline_result.get("breakdown", {})

        self.frontiers.add_result(
            package_type,
            breakdown.get("total_carbon_gco2", 0),
            breakdown.get("predicted_on_time", 0),
            breakdown.get("cost", 0),
            final_result.get("casp_score", 0),
        )

        carbon_result = self.carbon_service.analyze(
            carrier_options,
            package_type,
            optimization_result=final_result,
            country="india",
        )

        cost = float(breakdown.get("cost", 0) or 0)
        benchmark_cost_inr = benchmark.get("benchmark_cost_inr")
        benchmark_source = benchmark.get("benchmark_source")
        benchmark_query = benchmark.get("benchmark_query")
        benchmark_reasoning = benchmark.get("benchmark_reasoning")
        llm_recommendation = benchmark.get("llm_recommendation")

        efficiency = None
        efficiency_percentage = None
        if benchmark_cost_inr is not None and cost and benchmark_cost_inr > 0:
            pct = ((benchmark_cost_inr - cost) / benchmark_cost_inr) * 100
            label = "market rate" if benchmark_source in {"market_rate_web", "market_rate_llm"} else "historical average"
            efficiency_percentage = f"{pct:.1f}% {'below' if pct > 0 else 'above'} {label}"
            efficiency = f"Below {label}" if cost < benchmark_cost_inr else f"Above {label}"

        planning_packet = self._expert_packet(
            desk="Pricing & Planning Desk",
            status="ok",
            finding="Computed baseline and constrained rerun using expert-proposed constraints.",
            constraints_proposed=constraints,
            evidence=["stakes_optimizer", "route_optimizer", "cost_predictor", "on_time_predictor"],
            confidence=0.9,
            insufficient_evidence=False,
            fallback_used=False,
            notes="Rerun applied after expert consultation.",
        )
        sourcing_packet = self._expert_packet(
            desk="Carrier Procurement Desk",
            status="ok" if benchmark_source is not None else "insufficient_evidence",
            finding=llm_recommendation
            or (
                "Carrier shortlist generated by LLM sourcing desk."
                if used_llm_sourcing_options
                else "Carrier shortlist generated from deterministic sourcing service."
            ),
            constraints_proposed={"min_reliability_pct": constraints.get("min_reliability_pct")} if constraints.get("min_reliability_pct") else {},
            evidence=["carrier_options", "market benchmark scan", "dataset benchmark fallback"],
            confidence=0.78 if benchmark_source in {"market_rate_web", "market_rate_llm"} else 0.6,
            insufficient_evidence=benchmark_source is None,
            fallback_used=(not used_llm_sourcing_options) or benchmark_source == "historical_dataset" or benchmark_source is None,
            notes=(
                ("LLM sourcing options unavailable; used deterministic carrier options. " if not used_llm_sourcing_options else "")
                + ("NO TRUSTED MARKET BENCHMARK" if benchmark_source is None else f"Benchmark source: {benchmark_source}.")
            ),
        )
        sustainability_packet = self._expert_packet(
            desk="Sustainability & Policy Desk",
            status="ok",
            finding=f"Greenest viable carrier: {carbon_result.get('greenest_viable', '')}",
            constraints_proposed={},
            evidence=["carbon_service", "governance_scenarios", "grid_intensity"],
            confidence=0.88,
            insufficient_evidence=False,
            fallback_used=False,
            notes="Includes transport and AI compute carbon.",
        )

        expert_packets = {
            "risk_expert_packet": risk_run["packet"],
            "sourcing_expert_packet": sourcing_packet,
            "planning_expert_packet": planning_packet,
            "sustainability_expert_packet": sustainability_packet,
        }
        fallback_notes: List[str] = []
        if risk_run["packet"]["fallback_used"]:
            fallback_notes.append("Risk desk used deterministic fallback due to LLM unavailability.")
        if benchmark.get("llm_status") != "available":
            fallback_notes.append("Sourcing LLM unavailable; benchmark used deterministic web/dataset fallback.")
        if not used_llm_sourcing_options:
            fallback_notes.append("Sourcing desk used deterministic carrier options because no valid LLM carrier list was returned.")
        if sourcing_packet["insufficient_evidence"]:
            fallback_notes.append("NO TRUSTED MARKET BENCHMARK")

        deltas = {
            "cost_pct": round(self._metric_delta(float(breakdown.get("cost", 0) or 0), float(baseline_breakdown.get("cost", 0) or 0)), 3),
            "on_time_pct": round(self._metric_delta(float(breakdown.get("predicted_on_time", 0) or 0), float(baseline_breakdown.get("predicted_on_time", 0) or 0)), 3),
            "carbon_pct": round(self._metric_delta(float(breakdown.get("total_carbon_gco2", 0) or 0), float(baseline_breakdown.get("total_carbon_gco2", 0) or 0)), 3),
        }

        supervisor_summary = (
            "Control Tower Supervisor synthesized expert desks and selected final constrained plan."
        )
        execution_mode = "agentic_llm" if benchmark.get("llm_status") == "available" or risk_run.get("llm_status") == "available" else "deterministic_fallback"
        llm_status = "available" if execution_mode == "agentic_llm" else "unavailable"
        llm_error = risk_run.get("llm_error") or benchmark.get("llm_error")

        out = {
            "control_tower_supervisor": "active",
            "execution_mode": execution_mode,
            "llm_status": llm_status,
            "llm_error": llm_error,
            "supervisor_summary": supervisor_summary,
            "expert_packets": expert_packets,
            "plan_comparison": {
                "baseline_plan": baseline_result,
                "final_plan": final_result,
                "deltas": deltas,
            },
            "fallback_notes": fallback_notes,
            "recommendation": llm_recommendation
            or best_route.get("delivery_partner")
            or (carrier_options[0].get("carrier") if carrier_options else None),
            "cost": cost,
            "on_time_probability": breakdown.get("predicted_on_time", 0),
            "transport_carbon": breakdown.get("total_carbon_gco2", 0),
            "ai_carbon": carbon_result.get("ai_carbon_gco2", 0),
            "total_carbon": carbon_result.get("total_carbon_gco2", 0),
            "casp_score": carbon_result.get("casp_score", final_result.get("casp_score")),
            "casp_tier": carbon_result.get("casp_tier"),
            "carbon_roi": carbon_result.get("carbon_roi"),
            "risk_level": risk_assessment.get("risk_level", "LOW"),
            "risk_factors": risk_assessment.get("risk_factors", []),
            "warnings": risk_assessment.get("warnings", []),
            "recommended_buffer_days": risk_assessment.get("recommended_buffer_days", 0),
            "buffer_rationale": risk_assessment.get("buffer_rationale"),
            "early_warning_indicators": risk_assessment.get("early_warning_indicators", {}),
            "governance": carbon_result.get("governance", final_result.get("governance", {})),
            "tradeoff": carbon_result.get("tradeoff_analysis", {}),
            "greenest_viable": carbon_result.get("greenest_viable", ""),
            "carrier_options": carrier_options,
            "optimization_result": final_result,
            "carbon_analysis": carbon_result,
        }
        if benchmark_cost_inr is not None:
            out["benchmark_cost_inr"] = round(float(benchmark_cost_inr), 2)
            out["benchmark_source"] = benchmark_source
            if benchmark_query and benchmark_source == "market_rate_web":
                out["benchmark_query"] = benchmark_query
        if benchmark_reasoning:
            out["benchmark_reasoning"] = benchmark_reasoning
        if efficiency is not None:
            out["efficiency"] = efficiency
        if efficiency_percentage is not None:
            out["efficiency_percentage"] = efficiency_percentage
        return out

    def optimize_from_gathered_context(
        self,
        features: Dict[str, Any],
        risk_assessment: Optional[Dict[str, Any]] = None,
        carrier_options: Optional[List[Dict[str, Any]]] = None,
        benchmark: Optional[float] = None,
        recommendation: Optional[str] = None,
        benchmark_reasoning: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Phase-2 optimize-only path: uses gathered context directly and avoids re-running
        Risk/Sourcing gather agents.
        """
        features = extract_features_from_dict(features)
        package_type = (features.get("package_type") or "clothing").lower()
        origin = str(features.get("origin") or "mumbai").lower()
        destination = str(features.get("destination") or "delhi").lower()
        risk_assessment = risk_assessment or {}
        carrier_options = carrier_options or []

        if not carrier_options:
            carrier_options = self.sourcing_service.get_carrier_options(features, risk_assessment)
        if not carrier_options:
            return {
                "recommendation": None,
                "error": "No carrier options found for this shipment",
                "execution_mode": "deterministic_fallback",
                "llm_status": "unavailable",
                "carrier_options": [],
            }

        route_options = [o.get("route", {}) for o in carrier_options if isinstance(o, dict) and isinstance(o.get("route"), dict)]
        if not route_options:
            return {"error": "No valid route options in gathered carrier list", "carrier_options": carrier_options}

        constraints = {"buffer_days": int(risk_assessment.get("recommended_buffer_days", 0) or 0)}
        if package_type in {"pharmacy", "groceries"}:
            constraints["min_reliability_pct"] = 99.0

        baseline_result = self._optimize_with_constraints(
            package_type=package_type,
            route_options=route_options,
            origin=origin,
            destination=destination,
            constraints={},
        )
        final_result = self._optimize_with_constraints(
            package_type=package_type,
            route_options=route_options,
            origin=origin,
            destination=destination,
            constraints=constraints,
        )

        best_route = final_result.get("best_route", route_options[0] if route_options else {})
        delay_prob = self.ews.predict_delay_probability(best_route)
        risk = self.ews.calculate_risk_score(package_type, delay_prob, best_route)
        final_result["early_warning"] = risk
        final_result["governance"] = self.governance.generate_policy_recommendations(
            package_type,
            final_result,
            simulation_context={
                "route_options": route_options,
                "cost_predictor": self.cost_predictor,
                "on_time_predictor": self.on_time_predictor,
                "origin": origin,
                "destination": destination,
                "priority": self._priority_for_package(package_type),
            },
        )

        breakdown = final_result.get("breakdown", {})
        baseline_breakdown = baseline_result.get("breakdown", {})
        self.frontiers.add_result(
            package_type,
            breakdown.get("total_carbon_gco2", 0),
            breakdown.get("predicted_on_time", 0),
            breakdown.get("cost", 0),
            final_result.get("casp_score", 0),
        )
        carbon_result = self.carbon_service.analyze(
            carrier_options,
            package_type,
            optimization_result=final_result,
            country="india",
        )

        cost = float(breakdown.get("cost", 0) or 0)
        benchmark_cost_inr = float(benchmark) if benchmark is not None else None
        efficiency = None
        efficiency_percentage = None
        if benchmark_cost_inr is not None and cost and benchmark_cost_inr > 0:
            pct = ((benchmark_cost_inr - cost) / benchmark_cost_inr) * 100
            efficiency_percentage = f"{pct:.1f}% {'below' if pct > 0 else 'above'} market rate"
            efficiency = "Below market rate" if cost < benchmark_cost_inr else "Above market rate"

        deltas = {
            "cost_pct": round(self._metric_delta(float(breakdown.get("cost", 0) or 0), float(baseline_breakdown.get("cost", 0) or 0)), 3),
            "on_time_pct": round(self._metric_delta(float(breakdown.get("predicted_on_time", 0) or 0), float(baseline_breakdown.get("predicted_on_time", 0) or 0)), 3),
            "carbon_pct": round(self._metric_delta(float(breakdown.get("total_carbon_gco2", 0) or 0), float(baseline_breakdown.get("total_carbon_gco2", 0) or 0)), 3),
        }

        out = {
            "control_tower_supervisor": "active",
            "execution_mode": "agentic_llm",
            "llm_status": "available",
            "supervisor_summary": "Optimization ran on gathered context without re-running gather agents.",
            "expert_packets": {
                "planning_expert_packet": self._expert_packet(
                    desk="Pricing & Planning Desk",
                    status="ok",
                    finding="Optimize-only phase used gathered context.",
                    constraints_proposed=constraints,
                    evidence=["gathered_risk", "gathered_carrier_options", "stakes_optimizer"],
                    confidence=0.9,
                    insufficient_evidence=False,
                    fallback_used=False,
                    notes="Phase 2 executed without re-running Risk/Sourcing gather steps.",
                )
            },
            "plan_comparison": {
                "baseline_plan": baseline_result,
                "final_plan": final_result,
                "deltas": deltas,
            },
            "fallback_notes": [],
            "recommendation": recommendation
            or best_route.get("delivery_partner")
            or (carrier_options[0].get("carrier") if carrier_options else None),
            "cost": cost,
            "on_time_probability": breakdown.get("predicted_on_time", 0),
            "transport_carbon": breakdown.get("total_carbon_gco2", 0),
            "ai_carbon": carbon_result.get("ai_carbon_gco2", 0),
            "total_carbon": carbon_result.get("total_carbon_gco2", 0),
            "casp_score": carbon_result.get("casp_score", final_result.get("casp_score")),
            "casp_tier": carbon_result.get("casp_tier"),
            "carbon_roi": carbon_result.get("carbon_roi"),
            "risk_level": risk_assessment.get("risk_level", risk.get("risk_level", "LOW")),
            "risk_factors": risk_assessment.get("risk_factors", []),
            "warnings": risk_assessment.get("warnings", []),
            "recommended_buffer_days": risk_assessment.get("recommended_buffer_days", 0),
            "buffer_rationale": risk_assessment.get("buffer_rationale"),
            "early_warning_indicators": risk_assessment.get("early_warning_indicators", {}),
            "governance": carbon_result.get("governance", final_result.get("governance", {})),
            "tradeoff": carbon_result.get("tradeoff_analysis", {}),
            "greenest_viable": carbon_result.get("greenest_viable", ""),
            "carrier_options": carrier_options,
            "optimization_result": final_result,
            "carbon_analysis": carbon_result,
            "benchmark_reasoning": benchmark_reasoning,
        }
        if benchmark_cost_inr is not None:
            out["benchmark_cost_inr"] = round(float(benchmark_cost_inr), 2)
            out["benchmark_source"] = "market_rate_llm"
        if efficiency is not None:
            out["efficiency"] = efficiency
        if efficiency_percentage is not None:
            out["efficiency_percentage"] = efficiency_percentage
        return out

