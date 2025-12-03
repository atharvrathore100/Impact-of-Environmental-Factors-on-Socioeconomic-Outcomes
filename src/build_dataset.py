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
    default_lgii = Path("data/Nasa Earth Data/spatialecon-lgii-measures-v1-xlsx.xlsx")
    default_kaggle = Path("data/Kaggle Dataset")
    parser.add_argument(
        "--lgii",
        default=str(default_lgii),
        help=f"Path to LGII Excel file (default: {default_lgii})",
    )
    parser.add_argument(
        "--kaggle_dir",
        default=str(default_kaggle),
        help=f"Path to Kaggle data directory (default: {default_kaggle})",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV path for merged and enriched dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lgii_path = Path(args.lgii)
    kaggle_dir = Path(args.kaggle_dir)
    if not lgii_path.exists():
        raise FileNotFoundError(
            f"LGII Excel file not found at {lgii_path}. "
            "Update --lgii to point to the NASA Earth Data download."
        )
    if not kaggle_dir.exists():
        raise FileNotFoundError(
            f"Kaggle data directory not found at {kaggle_dir}. "
            "Update --kaggle_dir to point to the Kaggle dataset folder."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading LGII data...")
    lgii_df = load_lgii_excel(lgii_path)

    print("Loading country shapes...")
    shapes = load_country_shapes()

    print("Extracting environmental data...")
    env_static = pd.DataFrame(index=shapes.index)
    env_static["country_iso"] = shapes.index
    env_static["temp_celsius"] = extract_temperature(shapes, kaggle_dir)
    env_static["population"] = extract_population(shapes, kaggle_dir)
    env_static["vegetation_fraction"] = extract_landcover_veg_fraction(
        shapes, kaggle_dir
    )
    env_static["hazard_score"] = extract_deforestation_hazard(
        shapes, kaggle_dir
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
    gdp_panel = extract_gdp_timeseries(shapes, kaggle_dir, lgii_years)
    if not gdp_panel.empty:
        env_panel = env_panel.merge(gdp_panel, on=["country_iso", "year"], how="left")

    dataset = build_dataset(lgii_df, env_panel)
    dataset.to_csv(out_path, index=False)
    print(f"Saved merged dataset with {len(dataset)} rows to {out_path}")


if __name__ == "__main__":
    main()
