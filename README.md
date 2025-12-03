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
python3 -m venv DPA
source DPA/bin/activate      # macOS/Linux/WSL
pip install -r requirements.txt
```

### WSL prerequisites
When running inside Windows Subsystem for Linux, make sure your distro has Python 3 tooling available:
```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip
```
Then follow the same setup commands above (`python3 -m venv ...`).

### Running the pipeline with real datasets
- **Environmental data**: Kaggle “Geospatial Environmental and Socioeconomic Data” (requires Kaggle CLI & API token). Download and aggregate rasters to country level (mean NDVI, PM2.5, temp, precipitation, hazard intensity). Save a country-year table with at least the columns used in `data/sample/environment_sample.csv`.
- **LGII inequality data**: NASA SEDAC Light-based Geospatial Income Inequality (1992–2013). Export the country-year table with weighted Gini, population grids, density, and economic openness. Ensure ISO codes match the environmental file.

After preparing the real tables, point the scripts to them:
```bash
python3 src/build_dataset.py --lgii path/to/lgii.csv --environment path/to/environment.csv --out data/processed/merged.csv
python3 src/modeling.py --data data/processed/merged.csv --outdir reports
python3 src/visualize.py --data data/processed/merged.csv --outdir reports/figures
```

### Notes on alignment and cleaning
- Standardize country identifiers to ISO-3; the loader enforces uppercase and trims whitespace.
- Handle missing values before modeling; the scripts median-impute numeric predictors but you can swap in domain-specific imputation.
- If you add geospatial processing (CRS alignment, raster aggregation), install `geopandas`, `rasterio`, and `rasterstats` and plug the aggregated table into the same pipeline.

### Next steps
- Add Moran’s I or spatial lag models once geometries are available.
- Extend `modeling.py` with temporal ΔGini forecasting and cross-validation.
- Build a dashboard (e.g., Streamlit) using the merged dataset to interactively map CSVI.
