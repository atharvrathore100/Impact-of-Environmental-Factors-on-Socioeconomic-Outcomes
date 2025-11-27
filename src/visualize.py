"""
Lightweight visualization helpers for merged dataset.
Generates scatterplots and time-series charts for quick inspection.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def scatter_environment_vs_gini(df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.regplot(
        data=df,
        x="environmental_stress_index",
        y="gini",
        ax=ax,
        scatter_kws={"alpha": 0.7},
    )
    ax.set_title("Environmental Stress vs. Inequality")
    fig.tight_layout()
    fig.savefig(outdir / "stress_vs_gini.png", dpi=200)
    plt.close(fig)


def timeseries_gini(df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=df, x="year", y="gini", hue="country_iso", marker="o", ax=ax)
    ax.set_title("Gini by Country over Time")
    fig.tight_layout()
    fig.savefig(outdir / "gini_timeseries.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate visualizations.")
    parser.add_argument("--data", required=True, help="Merged dataset CSV")
    parser.add_argument("--outdir", default="reports/figures", help="Output directory")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)

    scatter_environment_vs_gini(df, outdir)
    timeseries_gini(df, outdir)
    print(f"Wrote figures to {outdir}")


if __name__ == "__main__":
    main()
