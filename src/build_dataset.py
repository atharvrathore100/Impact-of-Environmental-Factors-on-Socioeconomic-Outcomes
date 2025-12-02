"""
Command-line entrypoint to merge LGII and environmental datasets and output
an enriched, analysis-ready CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from data_processing import build_dataset
from data_extraction import (
    load_country_shapes,
    extract_temperature,
    extract_population,
    extract_gdp,
    load_lgii_excel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LGII and environmental data and compute indices."
    )
    parser.add_argument("--lgii", required=True, help="Path to LGII Excel file")
    parser.add_argument(
        "--keggal_dir", required=True, help="Path to Keggal data directory"
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
    # Extract data into a DataFrame indexed by country ISO (if shapes has ISO)
    # naturalearth_lowres has 'iso_a3'
    
    # Create environment DataFrame
    env_df = pd.DataFrame(index=shapes.index)
    env_df["country_iso"] = shapes["iso_a3"]
    # We assume data is for a specific year or averaged. 
    # LGII is panel data (year-country), but our static rasters are single-year (mostly).
    # We will replicate the static env data for each year in LGII later or just merge.
    # For now, let's create a static environment lookup.
    
    env_df["temp_celsius"] = extract_temperature(shapes, args.keggal_dir)
    env_df["population"] = extract_population(shapes, args.keggal_dir)
    env_df["gdp"] = extract_gdp(shapes, args.keggal_dir)
    
    # Drop rows with no ISO
    env_df = env_df[env_df["country_iso"] != "-99"]

    dataset = build_dataset(lgii_df, env_df)
    dataset.to_csv(out_path, index=False)
    print(f"Saved merged dataset with {len(dataset)} rows to {out_path}")


if __name__ == "__main__":
    main()
