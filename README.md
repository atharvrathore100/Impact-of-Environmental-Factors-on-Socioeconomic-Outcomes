## Impact of Environmental Factors on Socioeconomic Outcomes

This project operationalizes the proposal in `project 2 The A-team.pdf` by providing a reproducible Python pipeline to merge NASA SEDAC LGII inequality data with environmental indicators, compute a Climate–Socioeconomic Vulnerability Index (CSVI), and run baseline models/visualizations.

### Repository Layout
- `data/sample/` — toy LGII + environmental country-year CSVs to exercise the pipeline.
- `src/data_processing.py` — cleaning, merging, and index computation utilities.
- `src/build_dataset.py` — CLI to build an enriched merged CSV.
- `src/modeling.py` — linear regression + random forest baselines with metrics + feature importances.
- `src/visualize.py` — quick scatter + time-series plots.
- `requirements.txt` — minimal dependencies (no heavyweight geo stack by default).

### Setup
```bash
python -m venv .venv
.venv\\Scripts\\activate   # Windows
pip install -r requirements.txt
```

### Running the sample pipeline (no external data needed)
1) Build the merged dataset:
```bash
python src/build_dataset.py --lgii data/sample/lgii_sample.csv --environment data/sample/environment_sample.csv --out data/processed/merged_sample.csv
```
2) Train/evaluate models:
```bash
python src/modeling.py --data data/processed/merged_sample.csv --outdir reports
```
Outputs: `reports/metrics.json`, `reports/feature_importance.json`.
3) Generate visuals:
```bash
python src/visualize.py --data data/processed/merged_sample.csv --outdir reports/figures
```
Outputs: `reports/figures/stress_vs_gini.png`, `reports/figures/gini_timeseries.png`.

### Using the real datasets
- **Environmental data**: Kaggle “Geospatial Environmental and Socioeconomic Data” (requires Kaggle CLI & API token). Download and aggregate rasters to country level (mean NDVI, PM2.5, temp, precipitation, hazard intensity). Save a country-year table with at least the columns used in `data/sample/environment_sample.csv`.
- **LGII inequality data**: NASA SEDAC Light-based Geospatial Income Inequality (1992–2013). Export the country-year table with weighted Gini, population grids, density, and economic openness. Ensure ISO codes match the environmental file.

After preparing the real tables, point the scripts to them:
```bash
python src/build_dataset.py --lgii path/to/lgii.csv --environment path/to/environment.csv --out data/processed/merged.csv
python src/modeling.py --data data/processed/merged.csv --outdir reports
python src/visualize.py --data data/processed/merged.csv --outdir reports/figures
```

### Notes on alignment and cleaning
- Standardize country identifiers to ISO-3; the loader enforces uppercase and trims whitespace.
- Handle missing values before modeling; the scripts median-impute numeric predictors but you can swap in domain-specific imputation.
- If you add geospatial processing (CRS alignment, raster aggregation), install `geopandas`, `rasterio`, and `rasterstats` and plug the aggregated table into the same pipeline.

### Next steps
- Add Moran’s I or spatial lag models once geometries are available.
- Extend `modeling.py` with temporal ΔGini forecasting and cross-validation.
- Build a dashboard (e.g., Streamlit) using the merged dataset to interactively map CSVI.
