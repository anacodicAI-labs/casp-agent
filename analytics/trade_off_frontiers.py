"""
Analytics: Trade-off frontiers (Pareto + CASP ranking).
"""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TradeOffFrontiers:
    """Analyze and visualize trade-off frontiers."""

    def __init__(self):
        self.results = []

    def add_result(self, package_type: str, carbon_gco2: float, service_level: float, cost: float, casp_score: float = None):
        if casp_score is None:
            casp_score = service_level / carbon_gco2 if carbon_gco2 > 0 else 0
        self.results.append(
            {
                "package_type": package_type,
                "carbon_gco2": carbon_gco2,
                "service_level": service_level,
                "cost": cost,
                "casp_score": casp_score,
            }
        )

    def calculate_pareto_frontier(self, dimension: str = "carbon") -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        dim_col = "carbon_gco2" if dimension == "carbon" else dimension
        df = pd.DataFrame(self.results)
        df = df.sort_values(["service_level", dim_col], ascending=[False, True])
        pareto_points = []
        best_dimension = float("inf")
        for _, row in df.iterrows():
            if row[dim_col] < best_dimension:
                pareto_points.append(row)
                best_dimension = row[dim_col]
        return pd.DataFrame(pareto_points)

    def plot_pareto_frontier_2d(self, x_axis: str = "carbon_gco2", y_axis: str = "service_level", save_path: str = None):
        if not self.results:
            print("No results to plot!")
            return
        df = pd.DataFrame(self.results)
        pareto = self.calculate_pareto_frontier()
        plt.figure(figsize=(10, 6))
        plt.scatter(df[x_axis], df[y_axis], alpha=0.5, label="All Routes", color="gray")
        if not pareto.empty:
            pareto_sorted = pareto.sort_values(x_axis)
            plt.plot(pareto_sorted[x_axis], pareto_sorted[y_axis], "r-", linewidth=2, label="Pareto Frontier", marker="o")
        for ptype in df["package_type"].unique():
            subset = df[df["package_type"] == ptype]
            plt.scatter(subset[x_axis], subset[y_axis], label=f"{ptype}", alpha=0.7, s=100)
        plt.xlabel(x_axis.replace("_", " ").title())
        plt.ylabel(y_axis.replace("_", " ").title())
        plt.title("Pareto Frontier: Trade-off Analysis")
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Saved figure to {save_path}")
        plt.show()

    def calculate_casp_ranking(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        df = pd.DataFrame(self.results)
        return df.sort_values("casp_score", ascending=False)

    def get_recommendations(self) -> Dict:
        if not self.results:
            return {}
        df = pd.DataFrame(self.results)
        best_casp = df.loc[df["casp_score"].idxmax()]
        lowest_carbon = df.loc[df["carbon_gco2"].idxmin()]
        highest_service = df.loc[df["service_level"].idxmax()]
        lowest_cost = df.loc[df["cost"].idxmin()]
        return {
            "best_casp": best_casp.to_dict(),
            "lowest_carbon": lowest_carbon.to_dict(),
            "highest_service": highest_service.to_dict(),
            "lowest_cost": lowest_cost.to_dict(),
        }

