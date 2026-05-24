# Running the Wildfire Ignition Pipeline

This guide records how to run the current wildfire ignition project code from the repository root.

It includes two workflows:

1. The frozen research paired-patch baseline.
2. The new geospatial coarse-to-fine prototype.

All commands below assume you are in the repo root:

```bash
cd /mnt/work/wildfire/25m/wildfire-ignition-prediction-alberta
```

## Current Frozen Model Settings

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

These paths are local machine paths and should not be committed if copied into private config files.

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
/mnt/work/wildfire/25m/static/
```

## 1. Quick Syntax Check

Run this after pulling or after code changes:

```bash
python -m py_compile src/geospatial/create_alberta_coarse_grid.py
python -m py_compile src/geospatial/extract_model_b_patch.py
python -m py_compile src/geospatial/validate_model_b_feature_config.py
python -m py_compile src/geospatial/validate_model_a_feature_config.py
python -m py_compile src/inference/run_model_b_geospatial.py
python -m py_compile src/geospatial/create_model_a_25m_patches_from_candidates.py
python -m py_compile src/inference/run_model_a_geospatial.py
python -m py_compile src/inference/run_paired_patch_pipeline.py
python -m py_compile src/evaluation/summarize_paired_pipeline_results.py
```

## 2. Frozen Paired-Patch Research Baseline

This is the controlled test pipeline that matches 1 km and 25 m patches by filename.

It is useful for research evaluation, but it is not the final geospatial prototype.

```text
1 km patch file
↓
Model B gatekeeper
↓
if passed, find matching 25 m patch by filename
↓
Model A spatial refiner
↓
final patch-level summary
```

Run the frozen held-out paired test:

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

Summarize the paired-patch result:

```bash
python -m src.evaluation.summarize_paired_pipeline_results \
  --csv results/paired_patch_pipeline_test_frozen_hardneg_phase2_t030.csv
```

Frozen held-out paired test result recorded so far:

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

Model B gatekeeper result:

| Metric | Value |
|---|---:|
| TP | 3418 |
| FP | 836 |
| TN | 2741 |
| FN | 210 |
| patch precision | 0.8035 |
| patch recall | 0.9421 |
| patch FP rate | 0.2337 |
| patch accuracy | 0.8548 |

Model A patch outcome on Model-B-passed patches:

| Metric | Value |
|---|---:|
| TP | 3409 |
| FP | 811 |
| TN | 25 |
| FN | 9 |
| patch precision | 0.8078 |
| patch recall | 0.9974 |
| patch FP rate | 0.9701 |
| patch accuracy | 0.8072 |

Do not tune thresholds or retrain based on this held-out test result.

## 3. Geospatial Prototype Overview

The geospatial prototype replaces filename pairing with real map coordinates.

Correct coarse-to-fine flow:

```text
Alberta boundary
↓
32 km x 32 km coarse grid cells
↓
extract 32 x 32 Model B patches at 1 km resolution
↓
run Model B
↓
write selected 1 km candidate cells inside each coarse patch
↓
create 64 x 64 Model A patches at 25 m resolution around candidate cells
↓
run Model A on those 25 m patches
↓
stitch Model A predictions into a heatmap
```

Terminology:

```text
Model B coarse patch:
32 x 32 pixels at 1 km = 32 km x 32 km footprint

Model B candidate cell:
one selected 1 km x 1 km pixel inside the coarse patch

Model A fine patch:
64 x 64 pixels at 25 m = 1.6 km x 1.6 km footprint
```

## 4. Create the Alberta Coarse Grid

File:

```text
src/geospatial/create_alberta_coarse_grid.py
```

Create the Model-B-sized grid over Alberta:

```bash
python -m src.geospatial.create_alberta_coarse_grid \
  --boundary /home/bondada.romith/wildfire/NFBD/Alberta_boundary.shp \
  --output-geojson data/grids/alberta_coarse_grid.geojson \
  --output-parquet data/grids/alberta_coarse_grid.parquet \
  --patch-size 32 \
  --resolution-m 1000
```

Expected outputs:

```text
data/grids/alberta_coarse_grid.geojson
data/grids/alberta_coarse_grid.parquet
```

Optional grid inspection:

```bash
python - <<'PY'
import geopandas as gpd

grid = gpd.read_file("data/grids/alberta_coarse_grid.geojson")
print(grid.head())
print("cells:", len(grid))
print("crs:", grid.crs)
print("bounds:", grid.total_bounds)
PY
```

## 5. Prepare the 1 km Model B Feature Config

Template file committed to the repo:

```text
configs/model_b_1km_features.template.json
```

Create your local editable config:

```bash
cp configs/model_b_1km_features.template.json configs/model_b_1km_features.json
```

Edit the local file:

```text
configs/model_b_1km_features.json
```

The 1 km config must match this exact training key order:

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

Compute month constants for the target inference month:

```bash
python - <<'PY'
import math

month = 5  # change this to target inference month
print("sin_month:", math.sin(2 * math.pi * month / 12))
print("cos_month:", math.cos(2 * math.pi * month / 12))
PY
```

Update `sin_month` and `cos_month` in `configs/model_b_1km_features.json` before extraction.

Validate the local 1 km feature config:

```bash
python -m src.geospatial.validate_model_b_feature_config \
  --feature-config configs/model_b_1km_features.json \
  --channel-stats /mnt/work/wildfire/1km/patches_1km_balanced/channel_stats.json
```

Expected successful validation:

```text
Model B feature config validation passed
Feature count: 17
```

## 6. Extract 1 km Model B Geospatial Patches

File:

```text
src/geospatial/extract_model_b_patch.py
```

Extract one coarse patch first:

```bash
python -m src.geospatial.extract_model_b_patch \
  --grid data/grids/alberta_coarse_grid.geojson \
  --feature-config configs/model_b_1km_features.json \
  --output-dir data/patches/1km \
  --patch-id ab_coarse_000001 \
  --manifest results/geospatial/model_b_patch_extraction_manifest.csv \
  --overwrite
```

Inspect the extracted `.npz`:

```bash
python - <<'PY'
import numpy as np

p = "data/patches/1km/ab_coarse_000001.npz"
arr = np.load(p)
print(arr.files)
for k in arr.files:
    print(k, arr[k].shape, arr[k].dtype, float(arr[k].min()), float(arr[k].max()))
PY
```

Expected:

```text
17 feature keys
all feature arrays are (32, 32)
no metadata key inside the NPZ
```

Extract a small smoke-test batch of coarse patches:

```bash
python -m src.geospatial.extract_model_b_patch \
  --grid data/grids/alberta_coarse_grid.geojson \
  --feature-config configs/model_b_1km_features.json \
  --output-dir data/patches/1km \
  --all \
  --max-patches 25 \
  --overwrite \
  --manifest results/geospatial/model_b_patch_extraction_manifest_25.csv
```

Note: `--max-patches 25` means 25 coarse Model B patches for a smoke test. It does not mean 25 m patches.

## 7. Run Model B Geospatial Inference

File:

```text
src/inference/run_model_b_geospatial.py
```

This script writes two outputs:

```text
1. Coarse patch score CSV
2. Selected 1 km candidate cell CSV
```

Run on one extracted patch using the manifest:

```bash
python -m src.inference.run_model_b_geospatial \
  --model models/model_B_1km_gatekeeper_hardneg_phase2.keras \
  --channel-stats /mnt/work/wildfire/1km/patches_1km_balanced/channel_stats.json \
  --manifest results/geospatial/model_b_patch_extraction_manifest.csv \
  --threshold 0.30 \
  --candidate-threshold 0.30 \
  --output-csv results/geospatial/model_b_geospatial_scores_one_patch.csv \
  --candidate-csv results/geospatial/model_b_candidate_1km_cells_one_patch.csv
```

Inspect outputs:

```bash
cat results/geospatial/model_b_geospatial_scores_one_patch.csv
head results/geospatial/model_b_candidate_1km_cells_one_patch.csv
```

Run on the 25-coarse-patch smoke-test manifest:

```bash
python -m src.inference.run_model_b_geospatial \
  --model models/model_B_1km_gatekeeper_hardneg_phase2.keras \
  --channel-stats /mnt/work/wildfire/1km/patches_1km_balanced/channel_stats.json \
  --manifest results/geospatial/model_b_patch_extraction_manifest_25.csv \
  --threshold 0.30 \
  --candidate-threshold 0.30 \
  --output-csv results/geospatial/model_b_geospatial_scores_25.csv \
  --candidate-csv results/geospatial/model_b_candidate_1km_cells_25.csv
```

The candidate CSV is the bridge to Model A. Each row is one selected 1 km cell with map bounds:

```text
candidate_id
patch_id
row
col
probability
cell_xmin
cell_ymin
cell_xmax
cell_ymax
crs
```

## 8. Prepare and Validate the 25 m Model A Feature Config

Template file committed to the repo:

```text
configs/model_a_25m_features.template.json
```

Create your local editable config:

```bash
cp configs/model_a_25m_features.template.json configs/model_a_25m_features.json
```

Edit the local file:

```text
configs/model_a_25m_features.json
```

The current template expects this known 25 m feature pattern:

```text
DEM_25m
cos_month
distance_to_road_25m
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

Validate the local 25 m feature config before creating Model A patches:

```bash
python -m src.geospatial.validate_model_a_feature_config \
  --feature-config configs/model_a_25m_features.json \
  --channel-stats /mnt/work/wildfire/25m/patches_25m_balanced/channel_stats.json
```

Expected successful validation:

```text
Model A feature config validation passed
Feature count: 17
```

If validation fails, fix `configs/model_a_25m_features.json` before extraction.

## 9. Create 25 m Model A Patches From Model B Candidate Cells

File:

```text
src/geospatial/create_model_a_25m_patches_from_candidates.py
```

This reads selected 1 km candidate cells and creates one 64 x 64, 25 m patch centered on each candidate cell.

Each patch covers:

```text
64 pixels x 25 m = 1600 m
1.6 km x 1.6 km footprint
```

Run a small smoke test from the one-patch candidate CSV:

```bash
python -m src.geospatial.create_model_a_25m_patches_from_candidates \
  --candidates results/geospatial/model_b_candidate_1km_cells_one_patch.csv \
  --feature-config configs/model_a_25m_features.json \
  --output-dir data/cache/model_a_25m_patches \
  --manifest results/geospatial/model_a_25m_patch_manifest_one_patch.csv \
  --max-candidates 5 \
  --overwrite
```

Validated smoke-test result:

```text
completed: 1
failed: 0
example NPZ: data/cache/model_a_25m_patches/ab_coarse_000025_r16_c16.npz
feature count: 17
feature shape: 64 x 64
```

Inspect one generated 25 m `.npz`:

```bash
python - <<'PY'
import csv
import numpy as np

manifest = "results/geospatial/model_a_25m_patch_manifest_one_patch.csv"
with open(manifest, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

completed = [r for r in rows if r["status"] == "completed"]
failed = [r for r in rows if r["status"] == "failed"]

print("completed:", len(completed))
print("failed:", len(failed))
if failed:
    print("first failure:", failed[0]["message"])
if completed:
    p = completed[0]["npz_path"]
    arr = np.load(p)
    print("npz:", p)
    print(arr.files)
    for k in arr.files:
        print(k, arr[k].shape, arr[k].dtype, float(arr[k].min()), float(arr[k].max()))
PY
```

Expected:

```text
feature-only NPZ
all feature arrays are (64, 64)
feature keys match Model A training order
```

## 10. Run Model A Geospatial Inference

File:

```text
src/inference/run_model_a_geospatial.py
```

This script reads the Model A 25 m patch manifest, runs the frozen Model A spatial refiner, and writes:

```text
1. CSV summary per fine patch
2. georeferenced probability GeoTIFFs
3. georeferenced binary GeoTIFFs
```

Run a smoke test on the generated 25 m patch manifest:

```bash
python -m src.inference.run_model_a_geospatial \
  --model models/model_A_25m_spatial_unet.keras \
  --channel-stats /mnt/work/wildfire/25m/patches_25m_balanced/channel_stats.json \
  --manifest results/geospatial/model_a_25m_patch_manifest_one_patch.csv \
  --threshold 0.50 \
  --min-positive-pixels 1 \
  --batch-size 4 \
  --output-csv results/geospatial/model_a_geospatial_predictions_one_patch.csv \
  --probability-dir results/geospatial/model_a_probability_tifs \
  --binary-dir results/geospatial/model_a_binary_tifs
```

Inspect the CSV result:

```bash
cat results/geospatial/model_a_geospatial_predictions_one_patch.csv
```

Inspect generated GeoTIFF metadata:

```bash
python - <<'PY'
import csv
import rasterio

csv_path = "results/geospatial/model_a_geospatial_predictions_one_patch.csv"
with open(csv_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

completed = [r for r in rows if r["status"] == "completed"]
print("completed:", len(completed))
if completed:
    prob_path = completed[0]["probability_tif_path"]
    binary_path = completed[0]["binary_tif_path"]
    print("probability tif:", prob_path)
    print("binary tif:", binary_path)
    with rasterio.open(prob_path) as src:
        print("prob shape:", src.height, src.width)
        print("prob crs:", src.crs)
        print("prob bounds:", src.bounds)
        print("prob min/max:", float(src.read(1).min()), float(src.read(1).max()))
    with rasterio.open(binary_path) as src:
        print("binary shape:", src.height, src.width)
        print("binary crs:", src.crs)
        print("binary bounds:", src.bounds)
        print("binary unique:", sorted(set(src.read(1).ravel().tolist())))
PY
```

CSV-only smoke test without writing rasters:

```bash
python -m src.inference.run_model_a_geospatial \
  --model models/model_A_25m_spatial_unet.keras \
  --channel-stats /mnt/work/wildfire/25m/patches_25m_balanced/channel_stats.json \
  --manifest results/geospatial/model_a_25m_patch_manifest_one_patch.csv \
  --threshold 0.50 \
  --min-positive-pixels 1 \
  --output-csv results/geospatial/model_a_geospatial_predictions_one_patch.csv \
  --no-rasters
```

Expected:

```text
Rows: 1 or more
Completed: 1 or more
Failed: 0
probability GeoTIFF shape: 64 x 64
binary GeoTIFF shape: 64 x 64
CRS and bounds match the Model A patch manifest
```

## 11. Files Implemented So Far

```text
src/geospatial/create_alberta_coarse_grid.py
src/geospatial/extract_model_b_patch.py
src/geospatial/validate_model_b_feature_config.py
src/geospatial/validate_model_a_feature_config.py
src/inference/run_model_b_geospatial.py
src/geospatial/create_model_a_25m_patches_from_candidates.py
src/inference/run_model_a_geospatial.py
configs/model_b_1km_features.template.json
configs/model_a_25m_features.template.json
RUNNING.md
```

## 12. Files Not Implemented Yet

These are the next pieces required for the full heatmap dashboard pipeline:

```text
src/geospatial/stitch_model_a_heatmap.py
dashboard/app.py
```

Expected future flow:

```text
Model A geospatial prediction CSV + probability GeoTIFFs
↓
stitch rasters into Alberta heatmap GeoTIFF
↓
serve heatmap in dashboard
```

## 13. Local Files To Avoid Committing

Do not commit generated data, model files, local private configs, or caches unless intentionally using Git LFS or an external artifact store.

Examples:

```text
configs/model_b_1km_features.json
configs/model_a_25m_features.json
data/patches/1km/*.npz
data/cache/model_a_25m_patches/*.npz
results/geospatial/*.csv
results/geospatial/model_a_probability_tifs/*.tif
results/geospatial/model_a_binary_tifs/*.tif
models/*.keras
```

## 14. Document History

```text
2026-05-24:
Updated RUNNING.md with all commands for the frozen paired-patch baseline and the geospatial coarse-to-fine prototype implemented so far.
Added Model A 25 m feature-config validator command.
Recorded successful 25 m candidate-to-Model-A patch smoke test.
Added Model A geospatial inference command and GeoTIFF inspection commands.
```
