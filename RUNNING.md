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

Expected:

```text
grid crs: EPSG:3979
```

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
dashboard/app.py
configs/model_b_1km_features.template.json
configs/model_a_25m_features.template.json
RUNNING.md
```

## Next Files Not Implemented Yet

```text
api/main.py or src/api/main.py
Dockerfile
docker-compose.yml
```

Expected future flow:

```text
Dashboard prototype
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
results/geospatial/*.csv
results/geospatial/*.geojson
results/geospatial/*.json
results/geospatial/*.tif
results/geospatial/model_a_probability_tifs*/
results/geospatial/model_a_binary_tifs*/
results/geospatial/model_a_cell_probability_tifs*/
results/geospatial/model_a_cell_binary_tifs*/
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
```
