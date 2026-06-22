# Alberta wildfire ignition-risk dashboard

This is a read-only Streamlit dashboard for conference presentation and research review. It reads the latest saved outputs from the three-hour validation/prediction pipeline and does not run inference or modify model, raster, NPZ, validation-state, or raw data artifacts.

## What it shows

- Latest prediction run from `results/validation/latest_prediction_run.txt`
- Model B candidate count and Model A final-positive count
- Prospective validation hit rates at 1 km, 5 km, 10 km, and 25 km
- Median nearest distance and median lead time
- Interactive map of Model A candidate cells, current active fires, and validated prospective fires
- Optional research diagnostics when the CSVs are present:
  - Model A threshold sweep
  - prospective ranking diagnostic
  - ranked-vs-nearest feature comparison
  - hard-negative mining summary

## Install dashboard dependencies

From the repository root, inside the project Python environment:

```bash
pip install -r dashboard/requirements.txt
```

The project environment must already include the geospatial dependencies used by the main pipeline, especially `pandas`, `geopandas`, `shapely`, and `pyproj`.

## Run locally

```bash
streamlit run dashboard/app.py --server.port 8501
```

Then open the Streamlit URL printed in the terminal.

## Expected input files

The dashboard is useful when these files exist:

```text
results/validation/latest_prediction_run.txt
results/validation/prospective_validation_log.csv
data/validation/alberta_activefires_current.csv
results/runs/<latest_run>/model_b_candidates.csv
results/runs/<latest_run>/model_a_predictions.csv
results/runs/<latest_run>/model_a_candidate_cells.geojson
```

Optional diagnostic files:

```text
results/validation/model_a_threshold_sweep_summary.csv
results/validation/prospective_ranking_diagnostic_summary.csv
results/validation/prospective_ranking_diagnostic_detail.csv
results/validation/ranked_candidate_feature_comparison_summary.csv
results/validation/prospective_hard_negative_summary.csv
```

Missing optional diagnostics are shown as informational messages instead of causing the dashboard to fail.

## Presentation note

The dashboard should be described as a live research dashboard for a weather-conditioned regional ignition-risk screening model. Current prospective validation should be presented with the small-sample caveat.
