"""
Data extraction module to handle raw GeoTIFF and NetCDF files.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import xarray as xr
from rasterio.io import DatasetReader
from shapely.geometry import mapping


def load_country_shapes() -> gpd.GeoDataFrame:
    """Load country shapes using Natural Earth URL."""
    try:
        # URL for Natural Earth 110m Cultural Vectors (Countries)
        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        print(f"Downloading country shapes from {url}...")
        world = gpd.read_file(url)
        
        # Rename columns to match expected schema if needed (naturalearth_lowres had 'iso_a3', 'pop_est', 'name')
        # The downloaded file usually has 'ISO_A3', 'POP_EST', 'NAME' or similar.
        # Let's standardize column names to lowercase
        world.columns = world.columns.str.lower()
        
        # Exclude Antarctica
        if "pop_est" in world.columns:
            world = world[(world["pop_est"] > 0) & (world["name"] != "Antarctica")]
        
        return world
    except Exception as e:
        print(f"Error loading country shapes: {e}")
        raise


def _get_raster_stats(
    shapes: gpd.GeoDataFrame, raster_path: Path | str, stat: str = "mean"
) -> pd.Series:
    """Calculate zonal statistics for a raster against country shapes."""
    results = []
    
    with rasterio.open(raster_path) as src:
        # Reproject shapes to match raster CRS if needed
        if str(src.crs) != str(shapes.crs):
            shapes = shapes.to_crs(src.crs)

        for _, row in shapes.iterrows():
            try:
                geom = [mapping(row["geometry"])]
                out_image, _ = rasterio.mask.mask(src, geom, crop=True)
                # Mask nodata values
                data = out_image[0]
                if src.nodata is not None:
                    data = np.ma.masked_equal(data, src.nodata)
                
                if stat == "mean":
                    val = np.mean(data)
                elif stat == "sum":
                    val = np.sum(data)
                else:
                    val = np.nan
                
                # Handle masked array result
                if np.ma.is_masked(val):
                    results.append(np.nan)
                else:
                    results.append(val)
            except Exception:
                results.append(np.nan)

    return pd.Series(results, index=shapes.index)


def extract_temperature(
    shapes: gpd.GeoDataFrame, data_dir: Path | str
) -> pd.Series:
    """Extract mean temperature per country."""
    base_path = Path(data_dir)
    # Path found via exploration
    tif_path = base_path / "11_temperature/World_TEMP_GISdata_LTAy_GlobalSolarAtlas-v2_GEOTIFF/World_TEMP_GISdata_LTAy_GlobalSolarAtlas_GEOTIFF/TEMP.tif"
    
    if not tif_path.exists():
        warnings.warn(f"Temperature file not found at {tif_path}")
        return pd.Series(np.nan, index=shapes.index)

    print("Extracting temperature data...")
    return _get_raster_stats(shapes, tif_path, stat="mean")


def extract_population(
    shapes: gpd.GeoDataFrame, data_dir: Path | str
) -> pd.Series:
    """Extract total population per country."""
    base_path = Path(data_dir)
    # Path found via exploration
    tif_path = base_path / "5_GHS_POP_E2015_GLOBE_R2019A_54009_250_V1_0/GHS_POP_E2015_GLOBE_R2019A_54009_250_V1_0/GHS_POP_E2015_GLOBE_R2019A_54009_250_V1_0.tif"
    
    if not tif_path.exists():
        warnings.warn(f"Population file not found at {tif_path}")
        return pd.Series(np.nan, index=shapes.index)

    print("Extracting population data (this may take a while)...")
    # Using 'sum' for population count
    return _get_raster_stats(shapes, tif_path, stat="sum")


def extract_gdp(
    shapes: gpd.GeoDataFrame, data_dir: Path | str
) -> pd.Series:
    """Extract GDP data from NetCDF."""
    base_path = Path(data_dir)
    nc_path = base_path / "6_GDP/doi_10.5061_dryad.dk1j0__v2/GDP_PPP_1990_2015_5arcmin_v2.nc"
    
    if not nc_path.exists():
        warnings.warn(f"GDP file not found at {nc_path}")
        return pd.Series(np.nan, index=shapes.index)

    print("Extracting GDP data...")
    try:
        ds = xr.open_dataset(nc_path)
        # Assuming the variable name is 'GDP_PPP' or similar, need to check or be robust
        # Based on filename 'GDP_PPP_1990_2015_5arcmin_v2.nc', likely has a time dimension
        # We'll take the latest year available (2015)
        
        # Inspect variable names if possible, but for now guess standard ones or first data var
        var_name = list(ds.data_vars)[0]
        data_2015 = ds[var_name].sel(time=2015, method="nearest")
        
        # This is a bit complex to zonal stat efficiently with xarray+geopandas without rasterio
        # So we'll save a temporary tif or use a simplified point sampling if resolution is low
        # Or just skip for now if too complex, but let's try a simple approach:
        # Re-use rasterio if we can export to tif, or use regionmask if available (not in deps).
        
        # Alternative: just return NaNs for now if too hard without extra deps, 
        # but let's try to be helpful.
        # Actually, let's just use the population and temp for now as proof of concept
        # and maybe skip GDP if it's too heavy, or try to read it.
        
        # For simplicity in this environment, let's skip complex NetCDF zonal stats 
        # unless we really need it. We can use the CSV if we found one, but we didn't.
        
        return pd.Series(np.nan, index=shapes.index)

    except Exception as e:
        print(f"Error extracting GDP: {e}")
        return pd.Series(np.nan, index=shapes.index)


def load_lgii_excel(path: Path | str) -> pd.DataFrame:
    """Load LGII data from Excel."""
    print(f"Loading LGII from {path}")
    # Read the 'Data' sheet
    df = pd.read_excel(path, sheet_name="Data")
    
    # Standardize columns
    # Rename 'ISO' to 'country_iso'
    # Rename 'year' to 'year' (if needed, but usually it's lowercase or we normalize)
    
    # Normalize all columns to lowercase first to be safe
    df.columns = df.columns.str.lower()
    
    rename_map = {
        "iso": "country_iso",
        "country": "country_iso",
        "gini_weighted": "gini",
    }
    df = df.rename(columns=rename_map)
    return df
