# Running the Streamlit Dashboard Remotely

This guide records how to run the wildfire ignition dashboard on the remote lab machine and access it from a personal computer through PuTTY.

All commands assume the repo root:

```bash
cd /mnt/work/wildfire/25m/wildfire-ignition-prediction-alberta
```

## Required Dashboard Input

The dashboard uses the lightweight candidate-cell GeoJSON layer:

```text
results/geospatial/model_a_candidate_cells_100_epsg3979.geojson
```

The optional summary file is:

```text
results/geospatial/model_a_candidate_cells_100_epsg3979_summary.json
```

The GeoJSON should be in EPSG:3979. The dashboard reprojects it to EPSG:4326 for web-map display.

## Create the GeoJSON if Needed

```bash
python -m src.geospatial.export_model_a_cells_geojson \
  --predictions-csv results/geospatial/model_a_geospatial_predictions_100_epsg3979.csv \
  --output-geojson results/geospatial/model_a_candidate_cells_100_epsg3979.geojson \
  --summary-json results/geospatial/model_a_candidate_cells_100_epsg3979_summary.json
```

Check it:

```bash
python - <<'PY'
import geopandas as gpd

p = "results/geospatial/model_a_candidate_cells_100_epsg3979.geojson"
gdf = gpd.read_file(p)
print("features:", len(gdf))
print("crs:", gdf.crs)
print("bounds:", gdf.total_bounds)
print("final_positive counts:")
print(gdf["final_positive"].value_counts(dropna=False))
PY
```

Expected:

```text
features > 0
crs: EPSG:3979
```

## Syntax Check

```bash
python -m py_compile dashboard/app.py
```

## Run Streamlit on the Remote Machine

Start the dashboard on the remote machine:

```bash
streamlit run dashboard/app.py \
  --server.address 127.0.0.1 \
  --server.port 8501 \
  --server.headless true
```

Keep this terminal running.

## PuTTY Port Forwarding

On the personal computer, open PuTTY.

Go to:

```text
Connection → SSH → Tunnels
```

Add this tunnel:

```text
Source port:
8501

Destination:
127.0.0.1:8501

Type:
Local
Auto
```

Click `Add`.

You should see:

```text
L8501 127.0.0.1:8501
```

Then log in to the remote machine through that PuTTY session.

## Open the Dashboard Locally

On the personal computer browser, open:

```text
http://localhost:8501
```

Do not use the remote machine IP in the browser. Use `localhost` because PuTTY forwards the local browser request to Streamlit on the remote machine.

## If Port 8501 Is Busy

Run Streamlit on port 8502:

```bash
streamlit run dashboard/app.py \
  --server.address 127.0.0.1 \
  --server.port 8502 \
  --server.headless true
```

Use this PuTTY tunnel:

```text
Source port: 8502
Destination: 127.0.0.1:8502
```

Open:

```text
http://localhost:8502
```

## Validated Result

This workflow has been validated:

```text
The dashboard opens from the personal computer through PuTTY.
The map displays Model A candidate-cell polygons.
Summary cards and sidebar filters work.
```

## Dashboard Design

The dashboard overview uses GeoJSON candidate-cell polygons rather than a large full-resolution 25 m raster.

```text
Overview layer:
results/geospatial/model_a_candidate_cells_100_epsg3979.geojson

Detailed raster outputs:
results/geospatial/model_a_cell_probability_tifs_100_epsg3979/*.tif

GIS/export heatmap:
results/geospatial/alberta_model_a_cell_heatmap_100_epsg3979.tif
```

The dashboard currently displays:

```text
candidate-cell polygons
summary metrics
probability filter
final-positive filter
candidate table
CSV download
```
