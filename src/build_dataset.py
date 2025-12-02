"""
Command-line entrypoint to merge LGII and environmental datasets and output
an enriched, analysis-ready CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from data_processing import build_dataset
from data_extraction import (
    load_country_shapes,
    extract_temperature,
    extract_population,
    extract_gdp_timeseries,
    extract_landcover_veg_fraction,
    extract_deforestation_hazard,
    extract_pm25_placeholder,
    extract_precip_placeholder,
    load_lgii_excel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LGII and environmental data and compute indices."
    )
    parser.add_argument("--lgii", required=True, help="Path to LGII Excel file")
    parser.add_argument(
        "--kaggle_dir", required=True, help="Path to Kaggle data directory"
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV path for merged and enriched dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading LGII data...")
    lgii_df = load_lgii_excel(args.lgii)

    print("Loading country shapes...")
    shapes = load_country_shapes()

    print("Extracting environmental data...")
    env_static = pd.DataFrame(index=shapes.index)
    env_static["country_iso"] = shapes.index
    env_static["temp_celsius"] = extract_temperature(shapes, args.kaggle_dir)
    env_static["population"] = extract_population(shapes, args.kaggle_dir)
    env_static["vegetation_fraction"] = extract_landcover_veg_fraction(
        shapes, args.kaggle_dir
    )
    env_static["hazard_score"] = extract_deforestation_hazard(
        shapes, args.kaggle_dir
    )
    env_static["pm25_mean"] = extract_pm25_placeholder(shapes)
    env_static["precip_mm_month"] = extract_precip_placeholder(shapes)

    # Expand static variables across all LGII years
    lgii_years = lgii_df["year"].dropna().astype(int).unique().tolist()
    env_panel = pd.DataFrame(
        {
            "country_iso": np.repeat(env_static["country_iso"].values, len(lgii_years)),
            "year": np.tile(lgii_years, len(env_static)),
        }
    ).merge(env_static, on="country_iso", how="left")

    # Add GDP panel data where available
    gdp_panel = extract_gdp_timeseries(shapes, args.kaggle_dir, lgii_years)
    if not gdp_panel.empty:
        env_panel = env_panel.merge(gdp_panel, on=["country_iso", "year"], how="left")

    dataset = build_dataset(lgii_df, env_panel)
    dataset.to_csv(out_path, index=False)
    print(f"Saved merged dataset with {len(dataset)} rows to {out_path}")


if __name__ == "__main__":
    main()
