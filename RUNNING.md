# Running the Wildfire Ignition Pipeline

This guide records the current commands for the wildfire ignition project from the repository root.

```bash
cd /mnt/work/wildfire/25m/wildfire-ignition-prediction-alberta
```

## Current Model Settings

```text
Model B default:
models/model_B_1km_gatekeeper_hardneg_phase2.keras @ threshold 0.30

Model B recall backup:
models/model_B_1km_gatekeeper_phase2.keras @ threshold 0.40

Model A default:
models/model_A_25m_spatial_unet.keras @ threshold 0.50

Model A recall backup:
models/model_A_25m_spatial_unet.keras @ threshold 0.45

Model A operational minimum positive pixels:
1
```

## Important Local Paths

These are local machine paths. Do not commit them inside private config files.

```text
Alberta boundary shapefile:
/home/bondada.romith/wildfire/NFBD/Alberta_boundary.shp

1 km Model B channel stats:
/mnt/work/wildfire/1km/patches_1km_balanced/channel_stats.json

25 m Model A channel stats:
/mnt/work/wildfire/25m/patches_25m_balanced/channel_stats.json

Expected 1 km static raster folder:
/mnt/work/wildfire/1km/static/

Expected 25 m static raster folder:
/mnt/work/wildfire/25m/static_25/
```

## CRS Rule

The aligned static rasters are in:

```text
EPSG:3979
```

The geospatial prototype should therefore create the Alberta coarse grid and all downstream outputs in `EPSG:3979`.

Old `EPSG:3400` outputs were useful as code smoke tests only. Treat those outputs as stale geospatial artifacts.

## Open-Meteo API Safety Rule

Open-Meteo is used for live weather ingestion.

Known free API limits from project notes:

```text
10,000 calls/day
5,000 calls/hour
600 calls/minute
non-commercial use only
attribution required
```

The project client now throttles requests by default:

```text
Default request spacing: 0.25 seconds
Approximate max request rate from one process: 240 calls/minute
```

For safer province-wide validation runs, set:

```bash
export OPEN_METEO_MIN_REQUEST_INTERVAL_SECONDS=1.0
```

This slows the client to roughly:

```text
60 calls/minute
```

Recommended development sequence:

```text
3 patches
↓
100 patches
↓
full province once
```

Avoid repeated full-province runs on the same day unless needed.

## Quick Syntax Check

Run this after pulling or after code changes:

```bash
python -m py_compile src/geospatial/create_alberta_coarse_grid.py
python -m py_compile src/geospatial/extract_model_b_patch.py
python -m py_compile src/geospatial/validate_model_b_feature_config.py
python -m py_compile src/geospatial/validate_model_a_feature_config.py
python -m py_compile src/inference/run_model_b_geospatial.py
python -m py_compile src/geospatial/create_model_a_25m_patches_from_candidates.py
python -m py_compile src/inference/run_model_a_geospatial.py
python -m py_compile src/geospatial/stitch_model_a_heatmap.py
python -m py_compile src/geospatial/export_model_a_cells_geojson.py
python -m py_compile src/weather/open_meteo_client.py
python -m py_compile src/weather/weather_patch_generator.py
python -m py_compile src/inference/run_live_weather_model_b_scan.py
python -m py_compile src/inference/run_live_weather_model_a_candidates.py
python -m py_compile dashboard/app.py
python -m py_compile src/inference/run_paired_patch_pipeline.py
python -m py_compile src/evaluation/summarize_paired_pipeline_results.py
```

## Frozen Paired-Patch Research Baseline

This workflow matches 1 km and 25 m patches by filename. It is useful for controlled research evaluation, but it is not the final geospatial prototype.

```bash
python -m src.inference.run_paired_patch_pipeline \
  --model-b models/model_B_1km_gatekeeper_hardneg_phase2.keras \
  --model-a models/model_A_25m_spatial_unet.keras \
  --patches-1km-dir /mnt/work/wildfire/1km/patches_1km_balanced/test \
  --patches-25m-dir /mnt/work/wildfire/25m/patches_25m_balanced/test \
  --channel-stats-1km /mnt/work/wildfire/1km/patches_1km_balanced/channel_stats.json \
  --channel-stats-25m /mnt/work/wildfire/25m/patches_25m_balanced/channel_stats.json \
  --model-b-threshold 0.30 \
  --model-a-threshold 0.50 \
  --model-a-min-positive-pixels 1 \
  --output-csv results/paired_patch_pipeline_test_frozen_hardneg_phase2_t030.csv
```

Summarize:

```bash
python -m src.evaluation.summarize_paired_pipeline_results \
  --csv results/paired_patch_pipeline_test_frozen_hardneg_phase2_t030.csv
```

Recorded frozen held-out paired-test result:

```text
rows: 7205
completed: 4254
blocked_by_model_b: 2951
Model B passed patches: 4254 / 7205 = 0.5904
Model A ran patches:    4254 / 7205 = 0.5904
Final positive patches: 4220 / 7205 = 0.5857
Rows with 25 m labels:  4254 / 7205
Missing paired 25 m patches: 0
```

Do not tune thresholds or retrain from this held-out test result.

## Geospatial Prototype Overview

The geospatial prototype replaces filename pairing with real map coordinates.

```text
Alberta boundary
↓
32 km x 32 km coarse grid cells in EPSG:3979
↓
extract 32 x 32 Model B patches at 1 km resolution
↓
run Model B
↓
write selected 1 km candidate cells
↓
create 64 x 64 Model A context patches at 25 m resolution
↓
run Model A
↓
crop Model A output to the selected 1 km candidate cell, usually 40 x 40 pixels
↓
stitch cropped candidate-cell rasters into a heatmap
↓
export lightweight candidate-cell GeoJSON
↓
view GeoJSON in Streamlit dashboard
```

Terminology:

```text
Model B coarse patch:
32 x 32 pixels at 1 km = 32 km x 32 km footprint

Model B candidate cell:
one selected 1 km x 1 km pixel inside a coarse patch

Model A context patch:
64 x 64 pixels at 25 m = 1.6 km x 1.6 km footprint

Model A candidate-cell crop:
40 x 40 pixels at 25 m = 1 km x 1 km footprint
```

The final overview dashboard should use `cell_probability_tif_path` and candidate-cell GeoJSON, not full 1.6 km context-patch rasters.

## 1. Create Alberta Coarse Grid in EPSG:3979

```bash
python -m src.geospatial.create_alberta_coarse_grid \
  --boundary /home/bondada.romith/wildfire/NFBD/Alberta_boundary.shp \
  --output-geojson data/grids/alberta_coarse_grid_epsg3979.geojson \
  --output-parquet data/grids/alberta_coarse_grid_epsg3979.parquet \
  --patch-size 32 \
  --resolution-m 1000
```

Verify:

```bash
python - <<'PY'
import geopandas as gpd

grid = gpd.read_file("data/grids/alberta_coarse_grid_epsg3979.geojson")
print("grid crs:", grid.crs)
print("cells:", len(grid))
print("bounds:", grid.total_bounds)
print(grid.head().to_string())
PY
```

Expected validated province grid:

```text
grid rows: 692
crs: EPSG:3979
```

That means the full Alberta scan processes 692 Model B coarse patches. Each coarse patch is 32 km x 32 km and contains 1,024 possible 1 km cells.

## 2. Prepare and Validate Model B 1 km Feature Config

Create local config:

```bash
cp configs/model_b_1km_features.template.json configs/model_b_1km_features.json
```

The feature order must match the 1 km training channel stats:

```text
DEM_1km
cos_month
distance_to_road_1km
landcover_1km
municipalities_multiband_band1
municipalities_multiband_band2
municipalities_multiband_band3
municipalities_multiband_band4
municipalities_multiband_band5
municipalities_multiband_band6
municipalities_multiband_band7
municipalities_multiband_band8
relative_humidity
sin_month
temperature
water_1km
wind_speed
```

Validate:

```bash
python -m src.geospatial.validate_model_b_feature_config \
  --feature-config configs/model_b_1km_features.json \
  --channel-stats /mnt/work/wildfire/1km/patches_1km_balanced/channel_stats.json
```

Expected:

```text
Model B feature config validation passed
Feature count: 17
```

## 3. Extract 1 km Model B Geospatial Patches

One patch:

```bash
python -m src.geospatial.extract_model_b_patch \
  --grid data/grids/alberta_coarse_grid_epsg3979.geojson \
  --feature-config configs/model_b_1km_features.json \
  --output-dir data/patches/1km_epsg3979 \
  --patch-id ab_coarse_000001 \
  --manifest results/geospatial/model_b_patch_extraction_manifest_one_epsg3979.csv \
  --overwrite
```

100-patch smoke test:

```bash
python -m src.geospatial.extract_model_b_patch \
  --grid data/grids/alberta_coarse_grid_epsg3979.geojson \
  --feature-config configs/model_b_1km_features.json \
  --output-dir data/patches/1km_epsg3979 \
  --all \
  --max-patches 100 \
  --overwrite \
  --manifest results/geospatial/model_b_patch_extraction_manifest_100_epsg3979.csv
```

Expected from validated run:

```text
Completed: 100
Failed: 0
```

## 4. Run Model B Geospatial Inference

```bash
python -m src.inference.run_model_b_geospatial \
  --model models/model_B_1km_gatekeeper_hardneg_phase2.keras \
  --channel-stats /mnt/work/wildfire/1km/patches_1km_balanced/channel_stats.json \
  --manifest results/geospatial/model_b_patch_extraction_manifest_100_epsg3979.csv \
  --threshold 0.30 \
  --candidate-threshold 0.30 \
  --output-csv results/geospatial/model_b_geospatial_scores_100_epsg3979.csv \
  --candidate-csv results/geospatial/model_b_candidate_1km_cells_100_epsg3979.csv
```

Verify CRS:

```bash
python - <<'PY'
import pandas as pd

p = "results/geospatial/model_b_candidate_1km_cells_100_epsg3979.csv"
df = pd.read_csv(p)
print("rows:", len(df))
print("crs:", df["crs"].dropna().unique().tolist())
print(df.head().to_string(index=False))
PY
```

Expected:

```text
crs: ['EPSG:3979']
```

Validated Model B 100-patch EPSG:3979 candidate output:

```text
Candidate CSV: results/geospatial/model_b_candidate_1km_cells_100_epsg3979.csv
CRS: EPSG:3979
```

## 5. Prepare and Validate Model A 25 m Feature Config

Create local config:

```bash
cp configs/model_a_25m_features.template.json configs/model_a_25m_features.json
```

Model A expects this 25 m feature pattern:

```text
DEM_25m
cos_month
distance_to_road
landcover_25m
municipalities_multiband_band1
municipalities_multiband_band2
municipalities_multiband_band3
municipalities_multiband_band4
municipalities_multiband_band5
municipalities_multiband_band6
municipalities_multiband_band7
municipalities_multiband_band8
relative_humidity
sin_month
temperature
water_25m
wind_speed
```

Important:

```text
Model A expects the key distance_to_road, not distance_to_road_25m.
The raster path can still point to the 25 m road-distance raster.
```

Validate:

```bash
python -m src.geospatial.validate_model_a_feature_config \
  --feature-config configs/model_a_25m_features.json \
  --channel-stats /mnt/work/wildfire/25m/patches_25m_balanced/channel_stats.json
```

Expected:

```text
Model A feature config validation passed
Feature count: 17
```

## 6. Create 25 m Model A Context Patches

Create one 64 x 64, 25 m context patch per selected Model B 1 km candidate cell:

```bash
python -m src.geospatial.create_model_a_25m_patches_from_candidates \
  --candidates results/geospatial/model_b_candidate_1km_cells_100_epsg3979.csv \
  --feature-config configs/model_a_25m_features.json \
  --output-dir data/cache/model_a_25m_patches_100_epsg3979 \
  --manifest results/geospatial/model_a_25m_patch_manifest_100_epsg3979.csv \
  --overwrite
```

Expected:

```text
Completed > 0
Failed: 0
```

## 7. Run Model A Geospatial Inference With Cropped Candidate-Cell Outputs

This writes both full 64 x 64 context-patch rasters and cropped 40 x 40 candidate-cell rasters.

```bash
python -m src.inference.run_model_a_geospatial \
  --model models/model_A_25m_spatial_unet.keras \
  --channel-stats /mnt/work/wildfire/25m/patches_25m_balanced/channel_stats.json \
  --manifest results/geospatial/model_a_25m_patch_manifest_100_epsg3979.csv \
  --threshold 0.50 \
  --min-positive-pixels 1 \
  --batch-size 8 \
  --output-csv results/geospatial/model_a_geospatial_predictions_100_epsg3979.csv \
  --probability-dir results/geospatial/model_a_probability_tifs_100_epsg3979 \
  --binary-dir results/geospatial/model_a_binary_tifs_100_epsg3979 \
  --cell-probability-dir results/geospatial/model_a_cell_probability_tifs_100_epsg3979 \
  --cell-binary-dir results/geospatial/model_a_cell_binary_tifs_100_epsg3979
```

Inspect one cropped cell raster:

```bash
python - <<'PY'
import csv
import rasterio

csv_path = "results/geospatial/model_a_geospatial_predictions_100_epsg3979.csv"
with open(csv_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

completed = [r for r in rows if r["status"] == "completed"]
failed = [r for r in rows if r["status"] == "failed"]
print("completed:", len(completed))
print("failed:", len(failed))
if failed:
    print("first failure:", failed[0]["message"])

if completed:
    row = completed[0]
    print("candidate_id:", row["candidate_id"])
    print("cell_probability_tif_path:", row["cell_probability_tif_path"])
    with rasterio.open(row["cell_probability_tif_path"]) as src:
        arr = src.read(1)
        print("cell shape:", src.height, src.width)
        print("cell crs:", src.crs)
        print("cell bounds:", src.bounds)
        print("cell min/max:", float(arr.min()), float(arr.max()))
PY
```

Expected:

```text
Completed > 0
Failed: 0
cell shape: 40 x 40
cell crs: EPSG:3979
```

## 8. Stitch Cropped Candidate-Cell Heatmap

Default behavior stitches `cell_probability_tif_path`, not the full-patch `probability_tif_path`.

```bash
python -m src.geospatial.stitch_model_a_heatmap \
  --predictions-csv results/geospatial/model_a_geospatial_predictions_100_epsg3979.csv \
  --output-tif results/geospatial/alberta_model_a_cell_heatmap_100_epsg3979.tif \
  --binary-output-tif results/geospatial/alberta_model_a_cell_heatmap_100_epsg3979_binary.tif \
  --threshold 0.50
```

Inspect:

```bash
python - <<'PY'
import rasterio

for p in [
    "results/geospatial/alberta_model_a_cell_heatmap_100_epsg3979.tif",
    "results/geospatial/alberta_model_a_cell_heatmap_100_epsg3979_binary.tif",
]:
    print("\n", p)
    with rasterio.open(p) as src:
        arr = src.read(1)
        print("shape:", src.height, src.width)
        print("crs:", src.crs)
        print("bounds:", src.bounds)
        print("min/max:", float(arr.min()), float(arr.max()))
        if "binary" in p:
            print("unique:", sorted(set(arr.ravel().tolist())))
PY
```

## 9. Export Candidate-Cell GeoJSON for Dashboard

The dashboard uses lightweight candidate-cell polygons instead of a huge mostly-empty 25 m heatmap raster.

```bash
python -m src.geospatial.export_model_a_cells_geojson \
  --predictions-csv results/geospatial/model_a_geospatial_predictions_100_epsg3979.csv \
  --output-geojson results/geospatial/model_a_candidate_cells_100_epsg3979.geojson \
  --summary-json results/geospatial/model_a_candidate_cells_100_epsg3979_summary.json
```

Inspect:

```bash
python - <<'PY'
import geopandas as gpd
import json

geojson_path = "results/geospatial/model_a_candidate_cells_100_epsg3979.geojson"
summary_path = "results/geospatial/model_a_candidate_cells_100_epsg3979_summary.json"

gdf = gpd.read_file(geojson_path)
print("features:", len(gdf))
print("crs:", gdf.crs)
print("bounds:", gdf.total_bounds)
print("columns:", gdf.columns.tolist())
print("final_positive counts:")
print(gdf["final_positive"].value_counts(dropna=False))

with open(summary_path, "r", encoding="utf-8") as f:
    print(json.dumps(json.load(f), indent=2))
PY
```

Expected:

```text
features > 0
crs: EPSG:3979
final_positive counts printed
```

## 10. Run Streamlit Dashboard Through PuTTY Tunnel

Dashboard file:

```text
dashboard/app.py
```

Default dashboard input:

```text
results/geospatial/model_a_candidate_cells_100_epsg3979.geojson
results/geospatial/model_a_candidate_cells_100_epsg3979_summary.json
```

Run Streamlit on the remote machine:

```bash
streamlit run dashboard/app.py \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --server.headless true
```

In PuTTY on the local computer:

```text
Connection → SSH → Tunnels

Source port:
8501

Destination:
127.0.0.1:8501

Type:
Local
```

Click **Add**, then reconnect/login with that PuTTY session.

Open this on the local computer browser:

```text
http://localhost:8501
```

Expected dashboard behavior:

```text
Summary cards show candidate-cell counts.
Candidate-cell polygons appear on the map.
Sidebar probability filter works.
Final-positive filter works.
Candidate-cell table loads and can be downloaded as CSV.
```

Validated status:

```text
Dashboard accessible from local computer through PuTTY tunnel.
```

## 11. Live Weather Client Smoke Test

The live-weather client fetches model-ready weather values from Open-Meteo:

```text
temperature
relative_humidity
wind_speed
```

Syntax check:

```bash
python -m py_compile src/weather/open_meteo_client.py
python -m py_compile src/weather/weather_patch_generator.py
```

Fetch current weather for one point:

```bash
python -m src.weather.open_meteo_client \
  --lat 53.5461 \
  --lon -113.4938
```

Fetch with explicit request spacing:

```bash
python -m src.weather.open_meteo_client \
  --lat 53.5461 \
  --lon -113.4938 \
  --min-request-interval-seconds 1.0
```

Generate a 32 x 32 Model B weather patch:

```bash
python -m src.weather.weather_patch_generator \
  --temperature 20 \
  --relative-humidity 35 \
  --wind-speed 10 \
  --height 32 \
  --width 32 \
  --seed 42 \
  --output-npz results/geospatial/weather_patch_32x32_smoke.npz
```

Generate a 64 x 64 Model A weather patch:

```bash
python -m src.weather.weather_patch_generator \
  --temperature 20 \
  --relative-humidity 35 \
  --wind-speed 10 \
  --height 64 \
  --width 64 \
  --seed 42 \
  --output-npz results/geospatial/weather_patch_64x64_smoke.npz
```

Inspect weather patch outputs:

```bash
python - <<'PY'
import numpy as np

for p in [
    "results/geospatial/weather_patch_32x32_smoke.npz",
    "results/geospatial/weather_patch_64x64_smoke.npz",
]:
    print("\n", p)
    arr = np.load(p)
    print(arr.files)
    for k in arr.files:
        x = arr[k]
        print(k, x.shape, x.dtype, float(x.min()), float(x.max()), float(x.mean()))
PY
```

Expected:

```text
32 x 32 file:
temperature (32, 32)
relative_humidity (32, 32)
wind_speed (32, 32)

64 x 64 file:
temperature (64, 64)
relative_humidity (64, 64)
wind_speed (64, 64)
```

## 12. Live Weather Model B Saved-NPZ Validation

This is the validation-mode live-weather Model B runner. It saves generated NPZ patches so they can be inspected before switching to no-save streaming inference.

Output layout:

```text
data/runs/<run_id>/model_b_npz/
results/runs/<run_id>/model_b_manifest.csv
results/runs/<run_id>/model_b_scores.csv
results/runs/<run_id>/model_b_candidates.csv
results/runs/<run_id>/metadata.json
```

Run a 3-patch current-weather smoke test:

```bash
python -m src.inference.run_live_weather_model_b_scan \
  --grid data/grids/alberta_coarse_grid_epsg3979.geojson \
  --feature-config configs/model_b_1km_features.json \
  --model models/model_B_1km_gatekeeper_hardneg_phase2.keras \
  --channel-stats /mnt/work/wildfire/1km/patches_1km_balanced/channel_stats.json \
  --all \
  --max-patches 3 \
  --run-id live_weather_model_b_smoke_3_current \
  --threshold 0.30 \
  --candidate-threshold 0.30 \
  --batch-size 3 \
  --seed 42 \
  --overwrite
```

Run a 100-patch current-weather validation test:

```bash
python -m src.inference.run_live_weather_model_b_scan \
  --grid data/grids/alberta_coarse_grid_epsg3979.geojson \
  --feature-config configs/model_b_1km_features.json \
  --model models/model_B_1km_gatekeeper_hardneg_phase2.keras \
  --channel-stats /mnt/work/wildfire/1km/patches_1km_balanced/channel_stats.json \
  --all \
  --max-patches 100 \
  --run-id live_weather_model_b_100_current \
  --threshold 0.30 \
  --candidate-threshold 0.30 \
  --batch-size 32 \
  --seed 42 \
  --overwrite
```

Validated 100-patch current-weather Model B result:

```text
Selected patches: 100
NPZ extraction completed: 100
NPZ extraction failed: 0
Model B completed: 100
Model B failed: 0
Passed coarse patches: 7
Candidate 1 km cells: 7
```

Inspect one saved Model B live-weather NPZ:

```bash
python - <<'PY'
import json
import numpy as np
import pandas as pd

run_id = "live_weather_model_b_100_current"
manifest = f"results/runs/{run_id}/model_b_manifest.csv"
df = pd.read_csv(manifest)

print("manifest rows:", len(df))
print(df["status"].value_counts(dropna=False))

for _, row in df[df["status"] == "completed"].head(3).iterrows():
    print("\n==============================")
    print("patch_id:", row["patch_id"])
    print("npz:", row["npz_path"])

    arr = np.load(row["npz_path"])
    for key in ["temperature", "relative_humidity", "wind_speed"]:
        x = arr[key]
        print(key, x.shape, x.dtype, float(x.min()), float(x.max()), float(x.mean()))

    metadata = json.loads(row["metadata_json"])
    print("centroid lat/lon:", metadata["centroid_lat"], metadata["centroid_lon"])
    print("weather source:", metadata["weather"]["source"])
    print("weather source_time:", metadata["weather"]["source_time"])
    print("weather values:", {
        "temperature": metadata["weather"]["temperature"],
        "relative_humidity": metadata["weather"]["relative_humidity"],
        "wind_speed": metadata["weather"]["wind_speed"],
    })
PY
```

Inspect Model B scores and candidates:

```bash
python - <<'PY'
import json
import pandas as pd

run_id = "live_weather_model_b_100_current"

scores = pd.read_csv(f"results/runs/{run_id}/model_b_scores.csv")
cands = pd.read_csv(f"results/runs/{run_id}/model_b_candidates.csv")

print("scores:", len(scores))
print("candidates:", len(cands))

print("\nTop Model B coarse patches:")
print(scores.sort_values("model_b_max_prob", ascending=False)[[
    "patch_id",
    "model_b_max_prob",
    "model_b_mean_prob",
    "argmax_row",
    "argmax_col",
    "candidate_cell_count",
    "passed_gate",
]].head(10).to_string(index=False))

print("\nCandidate cells:")
print(cands.to_string(index=False))

with open(f"results/runs/{run_id}/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("\nmetadata counts:")
print(json.dumps(metadata["counts"], indent=2))
PY
```

## 13. Live Weather Model A Saved-NPZ Candidate Validation

This is the validation-mode live-weather Model A runner. It reads Model B candidate cells, saves generated Model A NPZ patches, runs Model A, and writes cropped candidate-cell rasters.

Output layout:

```text
data/runs/<run_id>/model_a_npz/
results/runs/<run_id>/model_a_manifest.csv
results/runs/<run_id>/model_a_predictions.csv
results/runs/<run_id>/model_a_metadata.json
results/runs/<run_id>/model_a_probability_tifs/
results/runs/<run_id>/model_a_binary_tifs/
results/runs/<run_id>/model_a_cell_probability_tifs/
results/runs/<run_id>/model_a_cell_binary_tifs/
```

Run Model A on the 100-patch current-weather Model B candidates:

```bash
python -m src.inference.run_live_weather_model_a_candidates \
  --candidates results/runs/live_weather_model_b_100_current/model_b_candidates.csv \
  --feature-config configs/model_a_25m_features.json \
  --model models/model_A_25m_spatial_unet.keras \
  --channel-stats /mnt/work/wildfire/25m/patches_25m_balanced/channel_stats.json \
  --run-id live_weather_model_b_100_current \
  --threshold 0.50 \
  --min-positive-pixels 1 \
  --batch-size 8 \
  --seed 42 \
  --overwrite
```

Validated 100-patch current-weather Model A result:

```text
Candidates: 7
Model A NPZ creation completed: 7
Model A NPZ creation failed: 0
Model A prediction rows: 7
Model A prediction completed: 7
Model A prediction failed: 0
Final positive candidate cells: 6
Final positive candidate-cell rate: 0.8571
```

Inspect one saved Model A live-weather NPZ and one cropped candidate-cell raster:

```bash
python - <<'PY'
import json
import numpy as np
import pandas as pd
import rasterio

run_id = "live_weather_model_b_100_current"

manifest = f"results/runs/{run_id}/model_a_manifest.csv"
predictions = f"results/runs/{run_id}/model_a_predictions.csv"

m = pd.read_csv(manifest)
p = pd.read_csv(predictions)

print("manifest rows:", len(m))
print(m["status"].value_counts(dropna=False))

print("\nprediction rows:", len(p))
print(p["status"].value_counts(dropna=False))
print("final_positive counts:")
print(p["final_positive"].value_counts(dropna=False))

row = m[m["status"] == "completed"].iloc[0]
print("\nNPZ:", row["npz_path"])
arr = np.load(row["npz_path"])

for key in ["temperature", "relative_humidity", "wind_speed"]:
    x = arr[key]
    print(key, x.shape, x.dtype, float(x.min()), float(x.max()), float(x.mean()))

metadata = json.loads(row["metadata_json"])
print("\ncentroid lat/lon:", metadata["centroid_lat"], metadata["centroid_lon"])
print("weather:", metadata["weather"])
print("weather_patch_summary:", metadata["weather_patch_summary"])

completed = p[p["status"] == "completed"]
r = completed.iloc[0]

print("\nfirst prediction:")
print(r[[
    "candidate_id",
    "coarse_patch_id",
    "cell_max_prob",
    "cell_mean_prob",
    "cell_positive_pixels",
    "final_positive",
    "cell_probability_tif_path",
]].to_string())

with rasterio.open(r["cell_probability_tif_path"]) as src:
    img = src.read(1)
    print("\ncell raster:")
    print("shape:", src.height, src.width)
    print("crs:", src.crs)
    print("bounds:", src.bounds)
    print("min/max:", float(img.min()), float(img.max()))
PY
```

Expected:

```text
Model A weather arrays: 64 x 64
cell raster shape: 40 x 40
crs: EPSG:3979
failed: 0
```

Export the live-weather dashboard GeoJSON:

```bash
python -m src.geospatial.export_model_a_cells_geojson \
  --predictions-csv results/runs/live_weather_model_b_100_current/model_a_predictions.csv \
  --output-geojson results/runs/live_weather_model_b_100_current/model_a_candidate_cells.geojson \
  --summary-json results/runs/live_weather_model_b_100_current/model_a_candidate_cells_summary.json
```

Inspect the live-weather GeoJSON:

```bash
python - <<'PY'
import geopandas as gpd
import json

run_id = "live_weather_model_b_100_current"
geojson = f"results/runs/{run_id}/model_a_candidate_cells.geojson"
summary = f"results/runs/{run_id}/model_a_candidate_cells_summary.json"

gdf = gpd.read_file(geojson)
print("features:", len(gdf))
print("crs:", gdf.crs)
print("bounds:", gdf.total_bounds)
print("final_positive counts:")
print(gdf["final_positive"].value_counts(dropna=False))

with open(summary, "r", encoding="utf-8") as f:
    print(json.dumps(json.load(f), indent=2))
PY
```

Validated 100-patch live-weather GeoJSON result:

```text
features: 7
crs: EPSG:3979
final_positive: 6 yes, 1 no
cell_max_prob_max: 0.70136905
cell_max_prob_mean: 0.54555394
```

## 14. Full Province Live Weather Saved-NPZ Validation

Before running full province, set conservative API throttling:

```bash
export OPEN_METEO_MIN_REQUEST_INTERVAL_SECONDS=1.0
```

Create and export one run ID for all commands:

```bash
RUN_ID=live_weather_province_current_$(date -u +%Y%m%d_%H%M%S)
export RUN_ID
echo $RUN_ID
```

Run full-province Model B with current weather:

```bash
python -m src.inference.run_live_weather_model_b_scan \
  --grid data/grids/alberta_coarse_grid_epsg3979.geojson \
  --feature-config configs/model_b_1km_features.json \
  --model models/model_B_1km_gatekeeper_hardneg_phase2.keras \
  --channel-stats /mnt/work/wildfire/1km/patches_1km_balanced/channel_stats.json \
  --all \
  --run-id $RUN_ID \
  --threshold 0.30 \
  --candidate-threshold 0.30 \
  --batch-size 128 \
  --seed 42 \
  --overwrite
```

Expected full Model B output:

```text
Coarse patch rows: 692
Completed: 692
Failed: 0
Passed coarse patches: some number
Candidate 1 km cells: some number
```

Summarize full Model B:

```bash
python - <<'PY'
import os
import json
import pandas as pd

run_id = os.environ["RUN_ID"]
print("RUN_ID:", run_id)

scores = pd.read_csv(f"results/runs/{run_id}/model_b_scores.csv")
cands = pd.read_csv(f"results/runs/{run_id}/model_b_candidates.csv")

print("\nModel B scores:", len(scores))
print(scores["status"].value_counts(dropna=False))

print("\npassed_gate counts:")
print(scores["passed_gate"].value_counts(dropna=False))

print("\ncandidate cells:", len(cands))

print("\nTop 20 coarse patches:")
print(scores.sort_values("model_b_max_prob", ascending=False)[[
    "patch_id",
    "model_b_max_prob",
    "model_b_mean_prob",
    "candidate_cell_count",
    "passed_gate",
]].head(20).to_string(index=False))

with open(f"results/runs/{run_id}/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("\nmetadata counts:")
print(json.dumps(metadata["counts"], indent=2))
PY
```

Run full Model A on the candidate cells from the same run:

```bash
python -m src.inference.run_live_weather_model_a_candidates \
  --candidates results/runs/$RUN_ID/model_b_candidates.csv \
  --feature-config configs/model_a_25m_features.json \
  --model models/model_A_25m_spatial_unet.keras \
  --channel-stats /mnt/work/wildfire/25m/patches_25m_balanced/channel_stats.json \
  --run-id $RUN_ID \
  --threshold 0.50 \
  --min-positive-pixels 1 \
  --batch-size 16 \
  --seed 42 \
  --overwrite
```

Export full live-weather dashboard GeoJSON:

```bash
python -m src.geospatial.export_model_a_cells_geojson \
  --predictions-csv results/runs/$RUN_ID/model_a_predictions.csv \
  --output-geojson results/runs/$RUN_ID/model_a_candidate_cells.geojson \
  --summary-json results/runs/$RUN_ID/model_a_candidate_cells_summary.json
```

Final full-run validation:

```bash
python - <<'PY'
import os
import json
import geopandas as gpd
import pandas as pd

run_id = os.environ["RUN_ID"]
print("RUN_ID:", run_id)

b = pd.read_csv(f"results/runs/{run_id}/model_b_candidates.csv")
a = pd.read_csv(f"results/runs/{run_id}/model_a_predictions.csv")
g = gpd.read_file(f"results/runs/{run_id}/model_a_candidate_cells.geojson")

print("\nModel B candidates:", len(b))
print("\nModel A predictions:", len(a))
print(a["status"].value_counts(dropna=False))
print("final_positive counts:")
print(a["final_positive"].value_counts(dropna=False))

print("\nGeoJSON features:", len(g))
print("GeoJSON CRS:", g.crs)
print("GeoJSON bounds:", g.total_bounds)

with open(f"results/runs/{run_id}/model_a_candidate_cells_summary.json", "r", encoding="utf-8") as f:
    print("\nGeoJSON summary:")
    print(json.dumps(json.load(f), indent=2))
PY
```

After the saved-NPZ full province run passes, the next implementation should be the no-save streaming version.

## 15. Run Streamlit Dashboard on a Live-Weather Run

Run Streamlit through the PuTTY tunnel as usual:

```bash
streamlit run dashboard/app.py \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --server.headless true
```

In the dashboard sidebar, use these paths for a live-weather run:

```text
Candidate-cell GeoJSON:
results/runs/<run_id>/model_a_candidate_cells.geojson

Summary JSON:
results/runs/<run_id>/model_a_candidate_cells_summary.json
```

For the validated 100-patch current-weather run:

```text
Candidate-cell GeoJSON:
results/runs/live_weather_model_b_100_current/model_a_candidate_cells.geojson

Summary JSON:
results/runs/live_weather_model_b_100_current/model_a_candidate_cells_summary.json
```

## Implemented Files

```text
src/geospatial/create_alberta_coarse_grid.py
src/geospatial/extract_model_b_patch.py
src/geospatial/validate_model_b_feature_config.py
src/geospatial/validate_model_a_feature_config.py
src/inference/run_model_b_geospatial.py
src/geospatial/create_model_a_25m_patches_from_candidates.py
src/inference/run_model_a_geospatial.py
src/geospatial/stitch_model_a_heatmap.py
src/geospatial/export_model_a_cells_geojson.py
src/weather/open_meteo_client.py
src/weather/weather_patch_generator.py
src/inference/run_live_weather_model_b_scan.py
src/inference/run_live_weather_model_a_candidates.py
dashboard/app.py
configs/model_b_1km_features.template.json
configs/model_a_25m_features.template.json
RUNNING.md
```

## Next Files Not Implemented Yet

```text
No-save streaming live-weather runner
api/main.py or src/api/main.py
Dockerfile
docker-compose.yml
```

Expected future flow:

```text
Saved-NPZ live-weather province validation
↓
No-save streaming live-weather inference
↓
FastAPI wrapper for inference/status endpoints
↓
Docker deployment
```

## Local Files To Avoid Committing

Do not commit generated data, model files, local private configs, or caches unless intentionally using Git LFS or an external artifact store.

Examples:

```text
configs/model_b_1km_features.json
configs/model_a_25m_features.json
data/grids/*.geojson
data/grids/*.parquet
data/patches/1km/*.npz
data/patches/1km_epsg3979/*.npz
data/cache/model_a_25m_patches*.npz
data/cache/model_a_25m_patches_*/
data/runs/*
results/geospatial/*.csv
results/geospatial/*.geojson
results/geospatial/*.json
results/geospatial/*.tif
results/geospatial/model_a_probability_tifs*/
results/geospatial/model_a_binary_tifs*/
results/geospatial/model_a_cell_probability_tifs*/
results/geospatial/model_a_cell_binary_tifs*/
results/runs/*
models/*.keras
```

## Document History

```text
2026-05-24:
Updated RUNNING.md with commands for the frozen paired-patch baseline and geospatial prototype.
Added Model A 25 m feature-config validator command.
Recorded successful 25 m candidate-to-Model-A patch smoke test.
Documented that Model A expects distance_to_road, not distance_to_road_25m.

2026-05-27:
Validated 100-coarse-patch geospatial smoke test.
Added cropped Model A candidate-cell raster workflow.
Added heatmap stitching from cell_probability_tif_path.
Added candidate-cell GeoJSON export workflow.

2026-05-30:
Documented EPSG:3979 as the required geospatial CRS for the aligned rasters and prototype outputs.
Added Streamlit dashboard through PuTTY tunnel workflow.
Recorded dashboard validation from local browser through remote PuTTY port forwarding.

2026-06-06:
Added Open-Meteo live-weather client and patch-shaped weather layer generator.
Added Open-Meteo request throttling using OPEN_METEO_MIN_REQUEST_INTERVAL_SECONDS.
Added saved-NPZ live-weather Model B scan runner.
Validated 3-patch and 100-patch current-weather Model B runs.
Added saved-NPZ live-weather Model A candidate runner.
Validated 100-patch current-weather two-stage pipeline: 7 Model B candidates, 7 Model A predictions, 6 final-positive cells.
Added full-province saved-NPZ live-weather validation commands.
```
