"""
Data-derived defaults for shipment features.
Computes per package_type: mode for categorical columns, mean for numeric (from Delivery_Logistics.csv).
Aligns with the Jupyter notebook logic for transparent defaults.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd

CATEGORICAL_FEATURES = [
    "delivery_partner",
    "vehicle_type",
    "delivery_mode",
    "region",
    "weather_condition",
]
NUMERICAL_FEATURES = ["distance_km", "package_weight_kg", "delivery_rating"]

FALLBACK_DEFAULTS: Dict[str, Any] = {
    "delivery_partner": "delhivery",
    "vehicle_type": "van",
    "delivery_mode": "two day",
    "region": "west",
    "weather_condition": "clear",
    "distance_km": 150.0,
    "package_weight_kg": 25.0,
    "delivery_rating": 4.0,
}

_defaults_cache: Optional[Dict[str, Dict[str, Any]]] = None


def _resolve_csv_path() -> Optional[Path]:
    base = Path(__file__).resolve().parent.parent
    p = base / "data" / "datasets" / "Delivery_Logistics.csv"
    return p if p.exists() else None


def compute_defaults_by_package(csv_path: Optional[os.PathLike] = None) -> Dict[str, Dict[str, Any]]:
    path = Path(csv_path) if csv_path else _resolve_csv_path()
    if path is None or not path.exists():
        return {"clothing": {**FALLBACK_DEFAULTS, "package_type": "clothing"}}

    df = pd.read_csv(path)
    required = ["package_type"] + CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"clothing": {**FALLBACK_DEFAULTS, "package_type": "clothing"}}

    result: Dict[str, Dict[str, Any]] = {}
    for package_type in df["package_type"].unique():
        subset = df[df["package_type"] == package_type]
        defaults: Dict[str, Any] = {"package_type": str(package_type).strip().lower()}
        for col in CATEGORICAL_FEATURES:
            mode_vals = subset[col].mode()
            defaults[col] = str(mode_vals.iloc[0]).strip().lower() if len(mode_vals) else FALLBACK_DEFAULTS.get(col, "")
        for col in NUMERICAL_FEATURES:
            mean_val = subset[col].mean()
            defaults[col] = round(float(mean_val), 2) if col != "delivery_rating" else max(1, min(5, round(float(mean_val))))
        result[defaults["package_type"]] = defaults
    return result


def get_defaults_cache(force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
    global _defaults_cache
    if _defaults_cache is None or force_reload:
        _defaults_cache = compute_defaults_by_package()
    return _defaults_cache


def get_defaults_for_package(package_type: str, csv_path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    cache = get_defaults_cache() if csv_path is None else compute_defaults_by_package(csv_path)
    key = str(package_type).strip().lower()
    if key in cache:
        return dict(cache[key])
    fallback = {**FALLBACK_DEFAULTS, "package_type": key}
    return fallback


def get_all_package_types() -> List[str]:
    cache = get_defaults_cache()
    return list(cache.keys())

