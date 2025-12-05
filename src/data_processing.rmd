"""
Utilities for cleaning, merging, and enriching LGII and environmental datasets.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def _standardize_iso(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def _coerce_year(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def clean_lgii(df: pd.DataFrame) -> pd.DataFrame:
    """Clean LGII tabular data."""
    # Ensure required columns exist
    if "country_iso" not in df.columns:
        raise ValueError("LGII dataframe missing 'country_iso' column")
    
    df = df.copy()
    df["country_iso"] = _standardize_iso(df["country_iso"])
    
    if "year" in df.columns:
        df["year"] = _coerce_year(df["year"])
        df = df.dropna(subset=["country_iso", "year"])
    else:
        df = df.dropna(subset=["country_iso"])
        
    return df


def clean_environment(df: pd.DataFrame) -> pd.DataFrame:
    """Clean environmental summaries."""
    df = df.copy()
    if "country_iso" not in df.columns:
        raise ValueError("Environmental dataframe missing 'country_iso' column")

    df["country_iso"] = _standardize_iso(df["country_iso"])
    
    if "year" in df.columns:
        df["year"] = _coerce_year(df["year"])
        df = df.dropna(subset=["country_iso", "year"])
    else:
        df = df.dropna(subset=["country_iso"])
        
    return df


def _zscore(series: pd.Series) -> pd.Series:
    if series.std(ddof=0) == 0:
        return series - series.mean()
    return (series - series.mean()) / series.std(ddof=0)


def merge_sources(lgii: pd.DataFrame, env: pd.DataFrame) -> pd.DataFrame:
    """Merge LGII and environmental data."""
    # Check if env has year
    if "year" in env.columns:
        on_cols = ["country_iso", "year"]
    else:
        on_cols = ["country_iso"]
        
    merged = pd.merge(
        lgii,
        env,
        on=on_cols,
        how="inner",
        suffixes=("_lgii", "_env"),
    )
    
    if "year" in merged.columns:
        merged = merged.sort_values(["country_iso", "year"])
    else:
        merged = merged.sort_values(["country_iso"])
        
    return merged


def add_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived indices for stress and vulnerability."""
    out = df.copy()

    # Map vegetation proxy to ndvi-like column for downstream use
    if "vegetation_fraction" in out.columns and "ndvi_mean" not in out.columns:
        out["ndvi_mean"] = out["vegetation_fraction"]

    # Handle missing columns by creating dummy Z-scores (0)
    def get_z(col_name):
        if col_name in out.columns and out[col_name].notna().any():
            return _zscore(out[col_name])
        return pd.Series(0, index=out.index)

    components = {
        "z_pm25": get_z("pm25_mean"),
        "z_hazard": get_z("hazard_score"),
        "z_temp": get_z("temp_celsius"),
        "z_ndvi": get_z("ndvi_mean"),
        "z_precip": get_z("precip_mm_month"),
    }
    for name, values in components.items():
        out[name] = values

    # Environmental stress index combines available factors.
    out["environmental_stress_index"] = (
        out["z_pm25"]
        + out["z_hazard"]
        + out["z_temp"]
        + out["z_precip"]
        - out["z_ndvi"]
    )

    # Socioeconomic factors
    # Map alternative column names to expected ones
    if "population" in out.columns and "population_millions" not in out.columns:
        out["population_millions"] = out["population"] / 1e6
    if "gdp_ppp" in out.columns and "gdp" not in out.columns:
        out["gdp"] = out["gdp_ppp"]

    socio_cols: Iterable[Tuple[str, float]] = [
        ("density_per_km2", 1.0),
        ("population_millions", 0.6),
        ("economic_openness", -0.8),
        ("gdp", -0.5),
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
        if series.max() == series.min():
            return pd.Series(0, index=series.index)
        return (series - series.min()) / (series.max() - series.min() + 1e-9)

    out["csvi"] = 0.6 * _minmax(out["environmental_stress_index"]) + 0.4 * _minmax(
        out["socioeconomic_vulnerability"]
    )

    if "gini" in out.columns:
        out["delta_gini"] = out.groupby("country_iso")["gini"].diff()
        
    return out


def build_dataset(lgii_df: pd.DataFrame, env_df: pd.DataFrame) -> pd.DataFrame:
    lgii = clean_lgii(lgii_df)
    env = clean_environment(env_df)
    merged = merge_sources(lgii, env)
    enriched = add_indices(merged)
    return enriched
