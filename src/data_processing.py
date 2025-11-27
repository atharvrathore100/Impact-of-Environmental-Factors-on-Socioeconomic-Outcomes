"""
Utilities for cleaning, merging, and enriching LGII and environmental datasets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def _standardize_iso(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def _coerce_year(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def load_lgii(path: Path | str) -> pd.DataFrame:
    """Load and minimally clean LGII-like tabular data."""
    df = pd.read_csv(path)
    rename_map = {
        "country": "country_iso",
        "iso": "country_iso",
        "gini_weighted": "gini",
    }
    df = df.rename(columns=rename_map)
    required_cols = {"country_iso", "year", "gini"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"LGII file missing required columns: {missing}")
    df["country_iso"] = _standardize_iso(df["country_iso"])
    df["year"] = _coerce_year(df["year"])
    num_cols = [c for c in df.columns if c not in {"country_iso"}]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["country_iso", "year"])
    return df


def load_environment(path: Path | str) -> pd.DataFrame:
    """Load and clean country-year environmental summaries."""
    df = pd.read_csv(path)
    if "country" in df.columns and "country_iso" not in df.columns:
        df = df.rename(columns={"country": "country_iso"})
    required_cols = {"country_iso", "year", "ndvi_mean", "pm25_mean"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Environmental file missing required columns: {missing}")
    df["country_iso"] = _standardize_iso(df["country_iso"])
    df["year"] = _coerce_year(df["year"])
    num_cols = [c for c in df.columns if c not in {"country_iso"}]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["country_iso", "year"])
    return df


def _zscore(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)


def merge_sources(lgii: pd.DataFrame, env: pd.DataFrame) -> pd.DataFrame:
    """Merge LGII and environmental data on country/year."""
    merged = pd.merge(
        lgii,
        env,
        on=["country_iso", "year"],
        how="inner",
        suffixes=("_lgii", "_env"),
    ).sort_values(["country_iso", "year"])
    return merged


def add_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived indices for stress and vulnerability."""
    out = df.copy()

    components = {
        "z_pm25": _zscore(out["pm25_mean"]),
        "z_hazard": _zscore(out.get("hazard_score", pd.Series(np.nan, index=out.index))),
        "z_temp": _zscore(out.get("temp_celsius", pd.Series(np.nan, index=out.index))),
        "z_ndvi": _zscore(out["ndvi_mean"]),
    }
    for name, values in components.items():
        out[name] = values

    out["environmental_stress_index"] = (
        out["z_pm25"] + out["z_hazard"] + out["z_temp"] - out["z_ndvi"]
    )

    socio_cols: Iterable[Tuple[str, float]] = [
        ("density_per_km2", 1.0),
        ("population_millions", 0.6),
        ("economic_openness", -0.8),
    ]
    socio_terms = []
    for col, sign in socio_cols:
        if col in out.columns:
            socio_terms.append(sign * _zscore(out[col]))
        else:
            socio_terms.append(pd.Series(0, index=out.index, dtype=float))
    out["socioeconomic_vulnerability"] = sum(socio_terms)

    # Combine for final CSVI. Scale components to 0-1 to keep the index interpretable.
    def _minmax(series: pd.Series) -> pd.Series:
        return (series - series.min()) / (series.max() - series.min() + 1e-9)

    out["csvi"] = 0.6 * _minmax(out["environmental_stress_index"]) + 0.4 * _minmax(
        out["socioeconomic_vulnerability"]
    )

    out["delta_gini"] = out.groupby("country_iso")["gini"].diff()
    return out


def build_dataset(lgii_path: Path | str, env_path: Path | str) -> pd.DataFrame:
    lgii = load_lgii(lgii_path)
    env = load_environment(env_path)
    merged = merge_sources(lgii, env)
    enriched = add_indices(merged)
    return enriched
