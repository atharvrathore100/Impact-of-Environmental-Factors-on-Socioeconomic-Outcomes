"""
Modeling experiments for predicting inequality from environmental factors.
Runs a linear model and a random forest and writes metrics and feature importance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


FEATURES: List[str] = [
    "pm25_mean",
    "ndvi_mean",
    "vegetation_fraction",
    "temp_celsius",
    "precip_mm_month",
    "hazard_score",
    "population_millions",
    "density_per_km2",
    "economic_openness",
    "gdp_ppp",
    "environmental_stress_index",
    "socioeconomic_vulnerability",
]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
    }


def run_models(data_path: Path | str) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["gini"])
    X = df[[f for f in FEATURES if f in df.columns]].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = df["gini"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    lin = LinearRegression()
    lin.fit(X_train, y_train)
    lin_pred = lin.predict(X_test)

    forest = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    forest.fit(X_train, y_train)
    rf_pred = forest.predict(X_test)

    return {
        "linear_regression": evaluate(y_test, lin_pred),
        "random_forest": evaluate(y_test, rf_pred),
        "feature_importance": dict(
            zip(X.columns.tolist(), forest.feature_importances_.round(4))
        ),
    }


def _load_country_shapes() -> gpd.GeoDataFrame:
    url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    try:
        world = gpd.read_file(url)
    except Exception:
        world = gpd.read_file(
            "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        )
    iso_col = None
    for candidate in ["iso_a3", "ISO_A3", "SOV_A3"]:
        if candidate in world.columns:
            iso_col = candidate
            break
    if iso_col is None:
        raise ValueError("No ISO A3 column found in country shapes.")
    world["iso_a3"] = world[iso_col].astype(str).str.upper()
    return world[world["iso_a3"] != "-99"]


def run_delta_models(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Model year-over-year change in Gini (ΔGini)."""
    df_delta = df.dropna(subset=["delta_gini"]).copy()
    if df_delta.empty:
        return {}

    X = df_delta[[f for f in FEATURES if f in df_delta.columns]].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = df_delta["delta_gini"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    lin = LinearRegression().fit(X_train, y_train)
    forest = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1
    ).fit(X_train, y_train)

    return {
        "delta_linear_regression": evaluate(y_test, lin.predict(X_test)),
        "delta_random_forest": evaluate(y_test, forest.predict(X_test)),
    }


def cluster_profiles(df: pd.DataFrame, outdir: Path, k: int = 4) -> None:
    """Cluster countries on combined environmental/socioeconomic profiles."""
    feature_cols = [c for c in FEATURES + ["csvi"] if c in df.columns]
    if not feature_cols:
        return

    latest = (
        df.sort_values("year")
        .groupby("country_iso")
        .tail(1)
        .reset_index(drop=True)
    )
    if len(latest) < 2:
        return

    X = latest[feature_cols].copy()
    X = X.fillna(X.median(numeric_only=True))

    k_effective = min(k, len(latest))
    kmeans = KMeans(n_clusters=k_effective, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)
    latest["cluster"] = labels
    outdir.mkdir(parents=True, exist_ok=True)
    latest[["country_iso", "cluster"]].to_csv(outdir / "cluster_assignments.csv", index=False)


def compute_morans_i(df: pd.DataFrame) -> Dict[str, float]:
    """
    Lightweight global Moran's I on CSVI using centroid-based inverse-distance weights.
    """
    if "csvi" not in df.columns:
        return {}
    gdf = _load_country_shapes()[["iso_a3", "geometry"]]
    latest = (
        df.sort_values("year")
        .groupby("country_iso")
        .tail(1)
        .reset_index(drop=True)
    )
    merged = gdf.merge(latest, left_on="iso_a3", right_on="country_iso")
    merged = merged.dropna(subset=["csvi"])
    if merged.empty:
        return {}

    centroids = merged.geometry.centroid
    coords = np.vstack([centroids.x.values, centroids.y.values]).T
    x = merged["csvi"].values
    x_bar = x.mean()
    x_dev = x - x_bar

    # Inverse-distance weights with small epsilon to avoid divide-by-zero
    dist_matrix = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dist_matrix, np.nan)
    weights = 1 / (dist_matrix + 1e-6)
    weights = np.where(np.isfinite(weights), weights, 0)

    s0 = weights.sum()
    if s0 == 0:
        return {}

    num = 0.0
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                continue
            num += weights[i, j] * x_dev[i] * x_dev[j]
    den = (x_dev**2).sum()
    moran_i = (len(x) / s0) * (num / den) if den != 0 else np.nan
    return {"moran_i_csvi": float(moran_i), "n_countries": int(len(x))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inequality prediction models.")
    parser.add_argument("--data", required=True, help="Path to merged dataset CSV")
    parser.add_argument(
        "--outdir",
        default="reports",
        help="Directory to write metrics.json and feature_importance.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_csv(args.data)
    results = run_models(args.data)
    delta_results = run_delta_models(base_df)
    metrics = {
        "linear_regression": results["linear_regression"],
        "random_forest": results["random_forest"],
        **delta_results,
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (outdir / "feature_importance.json").write_text(
        json.dumps(results["feature_importance"], indent=2)
    )

    cluster_profiles(base_df, outdir)
    spatial = compute_morans_i(base_df)
    if spatial:
        (outdir / "spatial.json").write_text(json.dumps(spatial, indent=2))

    print(f"Saved metrics, feature importances, clusters, and spatial stats to {outdir}")


if __name__ == "__main__":
    main()
