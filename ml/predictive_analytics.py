"""
ML: Predictive Analytics (Cost/On-time)
Transport carbon is computed deterministically via the emissions formula.
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.agent_mapping import get_agent_config
from config.vehicle_emissions import calculate_transport_carbon

warnings.filterwarnings("ignore")

_DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/datasets/Delivery_Logistics.csv")


class PredictiveAnalytics:
    """Predictive analytics for supply chain forecasting."""

    def __init__(self, data_path: str = _DEFAULT_DATA):
        self.data_path = data_path
        self.df = None
        self.cost_model = None
        self.on_time_model = None
        self.preprocessor = None
        self.feature_names = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        # Binary on-time label derived from delayed flag.
        self.df["on_time_label"] = (self.df["delayed"] == "no").astype(int)
        self.df["on_time_pct"] = (self.df["delayed"] == "no").astype(int) * 100
        # Deterministic transport carbon: distance × EF × cold-chain multiplier.
        self.df["carbon_gco2"] = self.df.apply(
            lambda row: calculate_transport_carbon(
                distance_km=float(row["distance_km"]) if not pd.isna(row["distance_km"]) else 150.0,
                vehicle_type=str(row["vehicle_type"]) if not pd.isna(row["vehicle_type"]) else "van",
                cold_chain_multiplier=float(
                    get_agent_config(str(row["package_type"]) if not pd.isna(row["package_type"]) else "clothing").get(
                        "cold_chain_multiplier", 1.0
                    )
                ),
            ),
            axis=1,
        )
        print(f"✓ Loaded {len(self.df)} records")

    def prepare_features(self):
        """
        Prepare features for ML models.
        Uses only features known BEFORE delivery occurs.
        """
        drop_cols = ["delivery_id", "delivery_time_hours", "expected_time_hours"]
        leaky_features = ["delayed", "delivery_status"]
        X = self.df.drop(columns=drop_cols + leaky_features + ["delivery_cost", "carbon_gco2", "on_time_pct"])

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
        self.feature_names = X.columns.tolist()
        return X_processed

    def train_models(self):
        X_processed = self.prepare_features()
        y_cost = self.df["delivery_cost"].values
        y_on_time = self.df["on_time_label"].values

        X_train, X_test, y_cost_train, y_cost_test = train_test_split(X_processed, y_cost, test_size=0.2, random_state=42)
        _, _, y_on_time_train, y_on_time_test = train_test_split(X_processed, y_on_time, test_size=0.2, random_state=42)

        print("\n📊 Training Cost Prediction Model...")
        self.cost_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.cost_model.fit(X_train, y_cost_train)
        cost_pred = self.cost_model.predict(X_test)
        print(f"  R² Score: {r2_score(y_cost_test, cost_pred):.4f}")
        print(f"  MAE: ₹{mean_absolute_error(y_cost_test, cost_pred):.2f}")

        print("\n📊 Training On-Time Prediction Model (Classifier)...")
        self.on_time_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.on_time_model.fit(X_train, y_on_time_train)
        on_time_pred_labels = self.on_time_model.predict(X_test)
        on_time_pred_pct = self.on_time_model.predict_proba(X_test)[:, 1] * 100
        print(f"  Accuracy: {accuracy_score(y_on_time_test, on_time_pred_labels):.4f}")
        print(f"  F1 Score: {f1_score(y_on_time_test, on_time_pred_labels, zero_division=0):.4f}")
        print(f"  MAE (vs 0/100 target): {mean_absolute_error(y_on_time_test * 100, on_time_pred_pct):.2f}%")

        print("\n✅ All models trained successfully!")

    def predict(self, route_dict: dict) -> dict:
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
        cost_pred = self.cost_model.predict(X_processed)[0]

        # Deterministic transport carbon: distance × EF × cold-chain multiplier.
        vehicle_type = str(route_df.iloc[0].get("vehicle_type", "van") or "van").lower()
        distance_km_raw = route_df.iloc[0].get("distance_km", 150.0)
        distance_km = float(distance_km_raw) if distance_km_raw is not None and not pd.isna(distance_km_raw) else 150.0
        pkg = str(route_df.iloc[0].get("package_type", "clothing") or "clothing").lower()
        cold_chain_multiplier = float(get_agent_config(pkg).get("cold_chain_multiplier", 1.0))
        carbon_pred = calculate_transport_carbon(distance_km, vehicle_type, cold_chain_multiplier)

        # Predict probability of on-time class (1), then convert to percentage.
        on_time_pred = float(self.on_time_model.predict_proba(X_processed)[0][1] * 100)
        return {
            "predicted_cost": float(cost_pred),
            "predicted_carbon_gco2": float(carbon_pred),
            "predicted_on_time_pct": on_time_pred,
        }

    def identify_forecast_failures(self):
        X_processed = self.prepare_features()
        cost_pred = self.cost_model.predict(X_processed)
        on_time_pred = self.on_time_model.predict_proba(X_processed)[:, 1] * 100
        cost_error = np.abs(self.df["delivery_cost"].values - cost_pred)
        on_time_error = np.abs(self.df["on_time_pct"].values - on_time_pred)
        self.df["cost_error"] = cost_error
        self.df["on_time_error"] = on_time_error
        failure_analysis = {}
        for feature in ["weather_condition", "delivery_partner", "region", "vehicle_type"]:
            if feature in self.df.columns:
                failures = (
                    self.df.groupby(feature)
                    .agg({"cost_error": "mean", "on_time_error": "mean", "delayed": lambda x: (x == "yes").sum()})
                    .sort_values("on_time_error", ascending=False)
                )
                failure_analysis[feature] = failures
        return failure_analysis


def create_predictors(data_path: str = _DEFAULT_DATA):
    analytics = PredictiveAnalytics(data_path)
    analytics.load_data()
    analytics.train_models()

    def cost_predictor(route_dict: dict) -> float:
        return analytics.predict(route_dict)["predicted_cost"]

    def on_time_predictor(route_dict: dict) -> float:
        return analytics.predict(route_dict)["predicted_on_time_pct"]

    return cost_predictor, on_time_predictor, analytics

