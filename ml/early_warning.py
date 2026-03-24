"""
ML: Early Warning System (delay prediction + early-warning indicators).
"""

import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.agent_mapping import get_agent_config

warnings.filterwarnings("ignore")

_DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/datasets/Delivery_Logistics.csv")


class EarlyWarningSystem:
    """Early-warning system for supply chain disruptions."""

    def __init__(self, data_path: str = _DEFAULT_DATA):
        self.data_path = data_path
        self.df = None
        self.delay_model = None
        self.preprocessor = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        self.df["is_delayed"] = (self.df["delayed"] == "yes").astype(int)
        print(f"✓ Loaded {len(self.df)} records")
        print(f"  Delay rate: {self.df['is_delayed'].mean()*100:.1f}%")

    def train_delay_predictor(self):
        drop_cols = ["delivery_id", "delivery_time_hours", "expected_time_hours", "delayed"]
        X = self.df.drop(columns=drop_cols + ["is_delayed", "delivery_status"])
        y = self.df["is_delayed"].values

        categorical_features = [
            "delivery_partner",
            "package_type",
            "vehicle_type",
            "delivery_mode",
            "region",
            "weather_condition",
        ]
        numerical_features = ["distance_km", "package_weight_kg", "delivery_rating"]

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ]
        )
        X_processed = self.preprocessor.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )

        self.delay_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.delay_model.fit(X_train, y_train)

        y_pred = self.delay_model.predict(X_test)
        print("\n📊 Delay Prediction Model Performance:")
        print(classification_report(y_test, y_pred, target_names=["On-Time", "Delayed"]))
        return self.delay_model

    def predict_delay_probability(self, route_dict: dict) -> float:
        if self.delay_model is None:
            self.load_data()
            self.train_delay_predictor()

        route_df = pd.DataFrame([route_dict])
        required_cols = [
            "delivery_partner",
            "package_type",
            "vehicle_type",
            "delivery_mode",
            "region",
            "weather_condition",
            "distance_km",
            "package_weight_kg",
            "delivery_rating",
        ]
        for col in required_cols:
            if col not in route_df.columns:
                if col in ["distance_km", "package_weight_kg", "delivery_rating"]:
                    route_df[col] = self.df[col].mean()
                else:
                    route_df[col] = self.df[col].mode()[0]

        X_processed = self.preprocessor.transform(route_df)
        return float(self.delay_model.predict_proba(X_processed)[0][1])

    def calculate_risk_score(self, package_type: str, delay_probability: float, route_dict: dict | None = None) -> Dict:
        config = get_agent_config(package_type)
        impact_multiplier = config["impact_multiplier"]

        risk_score = delay_probability * impact_multiplier
        risk_factors = []

        if route_dict:
            if route_dict.get("weather_condition") == "stormy":
                risk_score *= 1.5
                risk_factors.append("Stormy weather (+50% risk)")
            distance = route_dict.get("distance_km", 0)
            weight = route_dict.get("package_weight_kg", 0)
            if distance > 250 and weight > 40:
                risk_score *= 1.3
                risk_factors.append("Long distance + heavy weight (+30% risk)")
            partner = route_dict.get("delivery_partner", "")
            if partner:
                partner_delay_rate = self.df[self.df["delivery_partner"] == partner]["is_delayed"].mean()
                if partner_delay_rate > 0.15:
                    risk_score *= 1.2
                    risk_factors.append(f"High-risk partner: {partner} (+20% risk)")

        if risk_score > 5:
            risk_level = "CRITICAL"
        elif risk_score > 3:
            risk_level = "HIGH"
        elif risk_score > 1.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "delay_probability": delay_probability,
            "impact_multiplier": impact_multiplier,
            "package_type": package_type,
            "risk_factors": risk_factors,
            "alert_required": risk_score > 5,
        }

    def compute_early_warning_indicators(
        self,
        package_type: str,
        route_dict: Optional[Dict] = None,
        portfolio_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        if self.df is None:
            self.load_data()
        df = self.df
        pkg = (package_type or "").lower().strip()
        portfolio = portfolio_df if portfolio_df is not None else df[df["package_type"].astype(str).str.lower() == pkg]
        if portfolio.empty:
            portfolio = df
        n = len(portfolio)

        supplier_threshold_pct = 40.0
        partner_counts = portfolio["delivery_partner"].value_counts()
        top_share = float(partner_counts.iloc[0] / n) if n else 0.0
        top_partner = partner_counts.index[0] if len(partner_counts) else ""
        supplier_exceeds = bool(top_share * 100 > supplier_threshold_pct)
        supplier_amp = 0.0
        if supplier_exceeds:
            excess = (top_share * 100 - supplier_threshold_pct) / (100 - supplier_threshold_pct)
            supplier_amp = min(1.0, max(0.0, excess))

        supplier_concentration_index = {
            "name": "Supplier Concentration Index",
            "value_pct": round(float(top_share * 100), 2),
            "threshold_pct": supplier_threshold_pct,
            "exceeds": supplier_exceeds,
            "dominant_partner": str(top_partner),
            "interpretation": f">{supplier_threshold_pct}% single-supplier dependence"
            if supplier_exceeds
            else f"Within threshold (max {top_share*100:.1f}%)",
            "amplification_risk": round(supplier_amp, 3),
        }

        region_threshold_pct = 60.0
        region_counts = portfolio["region"].value_counts()
        top_region_share = float(region_counts.iloc[0] / n) if n else 0.0
        top_region = region_counts.index[0] if len(region_counts) else ""
        geo_exceeds = bool(top_region_share * 100 > region_threshold_pct)
        geo_amp = 0.0
        if geo_exceeds:
            excess = (top_region_share * 100 - region_threshold_pct) / (100 - region_threshold_pct)
            geo_amp = min(1.0, max(0.0, excess))

        weather_delay_correlation = None
        if "region" in portfolio.columns and "is_delayed" in portfolio.columns and n > 0:
            overall_delay = portfolio["is_delayed"].mean()
            by_region = portfolio.groupby("region")["is_delayed"].agg(["mean", "count"])
            by_region = by_region[by_region["count"] >= 10]
            if not by_region.empty and overall_delay > 0:
                max_region_delay = by_region["mean"].max()
                weather_delay_correlation = round(float(max_region_delay - overall_delay), 4)

        geographic_clustering = {
            "name": "Geographic Clustering",
            "value_pct": round(float(top_region_share * 100), 2),
            "threshold_pct": region_threshold_pct,
            "exceeds": geo_exceeds,
            "dominant_region": str(top_region),
            "interpretation": f">{region_threshold_pct}% regional concentration"
            if geo_exceeds
            else f"Within threshold (max {top_region_share*100:.1f}%)",
            "amplification_risk": round(geo_amp, 3),
            "weather_delay_correlation": weather_delay_correlation,
        }

        config = get_agent_config(package_type)
        cold_mult = config.get("cold_chain_multiplier", 1.0)
        tier = (config.get("agent") or config.get("tier_name") or "").lower()
        is_cold_chain = cold_mult > 1.0
        is_fragile = is_cold_chain and tier in ("critical", "high_value")
        cold_amp = 0.5 if is_fragile else (0.2 if is_cold_chain else 0.0)
        reason = (
            "Critical/cold-chain (e.g. pharmacy): limited temperature buffer"
            if is_fragile
            else ("Cold chain present" if is_cold_chain else "Ambient only")
        )

        cold_chain_fragility = {
            "name": "Cold-Chain Fragility",
            "is_fragile": is_fragile,
            "cold_chain_multiplier": cold_mult,
            "tier": config.get("tier_name", tier),
            "reason": reason,
            "interpretation": "Higher spoilage risk during disruptions" if is_fragile else "Lower spoilage risk",
            "amplification_risk": round(cold_amp, 3),
        }

        overall_amp = (supplier_amp + geo_amp + cold_amp) / 3.0
        if overall_amp >= 0.6:
            amplification_label = "HIGH"
        elif overall_amp >= 0.3:
            amplification_label = "MEDIUM"
        else:
            amplification_label = "LOW"

        return {
            "supplier_concentration_index": supplier_concentration_index,
            "geographic_clustering": geographic_clustering,
            "cold_chain_fragility": cold_chain_fragility,
            "quantified_amplification_risk": {"score": round(overall_amp, 3), "label": amplification_label},
        }

