"""
ML: Vendor Segmentation (K-Means clustering on delivery partners).
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/datasets/Delivery_Logistics.csv")


class VendorSegmentation:
    """K-Means clustering for vendor segmentation."""

    def __init__(self, data_path: str = _DEFAULT_DATA):
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()
        self.kmeans = None
        self.cluster_labels = None
        self.vendor_features = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df)} records")

    def prepare_vendor_features(self):
        vendor_stats = (
            self.df.groupby("delivery_partner")
            .agg(
                {
                    "delivery_cost": "mean",
                    "distance_km": "mean",
                    "delayed": lambda x: (x == "yes").mean(),
                    "delivery_rating": "mean",
                    "delivery_id": "count",
                }
            )
            .rename(
                columns={
                    "delivery_cost": "avg_cost",
                    "distance_km": "avg_distance",
                    "delayed": "delay_rate",
                    "delivery_rating": "avg_rating",
                    "delivery_id": "volume",
                }
            )
        )
        vendor_stats["on_time_pct"] = (1 - vendor_stats["delay_rate"]) * 100
        vendor_stats["cost_per_km"] = vendor_stats["avg_cost"] / vendor_stats["avg_distance"]
        vendor_stats["reliability_score"] = vendor_stats["on_time_pct"] * 0.6 + vendor_stats["avg_rating"] * 20 * 0.4
        self.vendor_features = vendor_stats

        print(f"\n✓ Calculated features for {len(vendor_stats)} vendors")
        print("\nVendor Statistics:")
        print(vendor_stats)
        return vendor_stats

    def cluster_vendors(self, n_clusters: int = 4):
        if self.vendor_features is None:
            self.prepare_vendor_features()
        features = ["avg_cost", "delay_rate", "avg_rating", "on_time_pct", "reliability_score"]
        X = self.vendor_features[features].values
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = self.kmeans.fit_predict(X_scaled)
        self.vendor_features["cluster"] = self.cluster_labels
        print(f"\n✓ Clustered vendors into {n_clusters} groups")
        return self.cluster_labels

    def interpret_clusters(self) -> dict:
        if self.vendor_features is None or "cluster" not in self.vendor_features.columns:
            self.cluster_vendors()
        cluster_analysis = {}
        for cluster_id in sorted(self.vendor_features["cluster"].unique()):
            cluster_vendors = self.vendor_features[self.vendor_features["cluster"] == cluster_id]
            avg_cost = cluster_vendors["avg_cost"].mean()
            avg_on_time = cluster_vendors["on_time_pct"].mean()
            avg_rating = cluster_vendors["avg_rating"].mean()
            avg_reliability = cluster_vendors["reliability_score"].mean()
            if avg_on_time > 90 and avg_rating > 4.0:
                cluster_type = "Premium/Reliable"
            elif avg_cost < self.vendor_features["avg_cost"].median():
                cluster_type = "Budget/Cheap"
            elif avg_on_time < 85:
                cluster_type = "Unreliable"
            else:
                cluster_type = "Balanced"
            cluster_analysis[cluster_id] = {
                "type": cluster_type,
                "vendors": cluster_vendors.index.tolist(),
                "avg_cost": avg_cost,
                "avg_on_time": avg_on_time,
                "avg_rating": avg_rating,
                "avg_reliability": avg_reliability,
                "count": len(cluster_vendors),
            }
        return cluster_analysis

    def get_vendor_recommendation(self, package_type: str, priority: str = "reliability") -> str:
        if self.vendor_features is None or "cluster" not in self.vendor_features.columns:
            self.cluster_vendors()
            self.interpret_clusters()
        package_vendors = self.df[self.df["package_type"] == package_type].groupby("delivery_partner").agg(
            {"delayed": lambda x: (x == "yes").mean(), "delivery_cost": "mean", "delivery_rating": "mean"}
        )
        if priority == "reliability":
            package_vendors["on_time_pct"] = (1 - package_vendors["delayed"]) * 100
            best_vendor = package_vendors["on_time_pct"].idxmax()
        elif priority == "cost":
            best_vendor = package_vendors["delivery_cost"].idxmin()
        else:
            best_vendor = package_vendors["delivery_rating"].idxmax()
        return best_vendor

