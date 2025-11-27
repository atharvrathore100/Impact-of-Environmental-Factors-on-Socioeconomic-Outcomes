"""
Command-line entrypoint to merge LGII and environmental datasets and output
an enriched, analysis-ready CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from data_processing import build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LGII and environmental data and compute indices."
    )
    parser.add_argument("--lgii", required=True, help="Path to LGII CSV")
    parser.add_argument("--environment", required=True, help="Path to environmental CSV")
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

    dataset = build_dataset(args.lgii, args.environment)
    dataset.to_csv(out_path, index=False)
    print(f"Saved merged dataset with {len(dataset)} rows to {out_path}")


if __name__ == "__main__":
    main()
