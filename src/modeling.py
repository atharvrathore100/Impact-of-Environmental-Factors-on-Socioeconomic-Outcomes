"""
Modeling experiments for predicting inequality from environmental factors.
Runs a linear model and a random forest and writes metrics and feature importance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


FEATURES: List[str] = [
    "pm25_mean",
    "ndvi_mean",
    "temp_celsius",
    "precip_mm_month",
    "hazard_score",
    "population_millions",
    "density_per_km2",
    "economic_openness",
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

    results = run_models(args.data)
    metrics = {
        "linear_regression": results["linear_regression"],
        "random_forest": results["random_forest"],
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (outdir / "feature_importance.json").write_text(
        json.dumps(results["feature_importance"], indent=2)
    )
    print(f"Saved metrics and feature importances to {outdir}")


if __name__ == "__main__":
    main()
