"""Streamlit dashboard for geospatial wildfire ignition candidate cells.

This dashboard displays the lightweight GeoJSON exported by
``src.geospatial.export_model_a_cells_geojson``. The GeoJSON is expected to use
the model/geospatial CRS, usually EPSG:3979, and is reprojected to EPSG:4326 for
web-map display.

Run from the repository root:

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import streamlit as st

try:
    import pydeck as pdk
except ImportError:  # pragma: no cover - runtime fallback for minimal environments.
    pdk = None


DEFAULT_GEOJSON_PATH = "results/runs/live_weather_province_current_20260607_211205/model_a_candidate_cells.geojson"
DEFAULT_SUMMARY_JSON_PATH = "results/runs/live_weather_province_current_20260607_211205/model_a_candidate_cells_summary.json"
WEB_CRS = "EPSG:4326"


st.set_page_config(
    page_title="Alberta Wildfire Ignition Risk",
    page_icon="🔥",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_candidate_cells(path: str) -> gpd.GeoDataFrame:
    """Load candidate-cell GeoJSON and normalize dashboard columns."""
    geojson_path = Path(path)
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")

    gdf = gpd.read_file(geojson_path)
    if gdf.empty:
        raise ValueError(f"GeoJSON has no features: {geojson_path}")
    if gdf.crs is None:
        raise ValueError(f"GeoJSON has no CRS: {geojson_path}")

    for column in ["cell_max_prob", "cell_mean_prob", "cell_positive_pixels", "final_positive"]:
        if column in gdf.columns:
            gdf[column] = pd.to_numeric(gdf[column], errors="coerce")

    if "final_positive" in gdf.columns:
        gdf["final_positive"] = gdf["final_positive"].fillna(0).astype(int)
    else:
        gdf["final_positive"] = 0

    if "cell_max_prob" not in gdf.columns:
        gdf["cell_max_prob"] = 0.0
    gdf["cell_max_prob"] = gdf["cell_max_prob"].fillna(0.0).astype(float)

    if "cell_mean_prob" not in gdf.columns:
        gdf["cell_mean_prob"] = 0.0
    gdf["cell_mean_prob"] = gdf["cell_mean_prob"].fillna(0.0).astype(float)

    if "cell_positive_pixels" not in gdf.columns:
        gdf["cell_positive_pixels"] = 0
    gdf["cell_positive_pixels"] = gdf["cell_positive_pixels"].fillna(0).astype(int)

    return gdf


@st.cache_data(show_spinner=False)
def load_summary_json(path: str) -> dict[str, Any]:
    """Load optional dashboard summary JSON."""
    summary_path = Path(path)
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def filter_candidate_cells(
    gdf: gpd.GeoDataFrame,
    min_probability: float,
    show_only_final_positive: bool,
) -> gpd.GeoDataFrame:
    """Apply sidebar filters."""
    filtered = gdf[gdf["cell_max_prob"] >= min_probability].copy()
    if show_only_final_positive:
        filtered = filtered[filtered["final_positive"] == 1].copy()
    return filtered


def build_display_dataframe(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Create a compact non-geometry table for display and download."""
    preferred_columns = [
        "candidate_id",
        "coarse_patch_id",
        "cell_max_prob",
        "cell_mean_prob",
        "cell_positive_pixels",
        "final_positive",
        "threshold",
        "cell_probability_tif_path",
        "cell_binary_tif_path",
    ]
    columns = [column for column in preferred_columns if column in gdf.columns]
    table = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))[columns].copy()
    if "cell_max_prob" in table.columns:
        table = table.sort_values("cell_max_prob", ascending=False)
    return table


def build_geojson_for_map(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    """Reproject candidate polygons to EPSG:4326 and return GeoJSON data."""
    display_gdf = gdf.to_crs(WEB_CRS)
    return json.loads(display_gdf.to_json())


def map_view_state(gdf: gpd.GeoDataFrame) -> dict[str, float]:
    """Compute an initial PyDeck view state from reprojected bounds."""
    display_gdf = gdf.to_crs(WEB_CRS)
    minx, miny, maxx, maxy = display_gdf.total_bounds
    longitude = float((minx + maxx) / 2.0)
    latitude = float((miny + maxy) / 2.0)

    # A simple zoom heuristic: smaller extents get a closer zoom.
    extent = max(float(maxx - minx), float(maxy - miny))
    if extent < 0.2:
        zoom = 8.5
    elif extent < 1.0:
        zoom = 7.0
    elif extent < 3.0:
        zoom = 5.5
    else:
        zoom = 4.5

    return {"latitude": latitude, "longitude": longitude, "zoom": zoom}


def render_pydeck_map(gdf: gpd.GeoDataFrame) -> None:
    """Render candidate-cell polygons using PyDeck."""
    if pdk is None:
        st.warning("pydeck is not installed. Showing table only.")
        return

    geojson_data = build_geojson_for_map(gdf)
    view_state_values = map_view_state(gdf)

    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson_data,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        get_fill_color="properties.final_positive == 1 ? [220, 70, 50, 150] : [70, 120, 220, 80]",
        get_line_color=[40, 40, 40, 180],
        line_width_min_pixels=1,
    )

    tooltip = {
        "html": "<b>{candidate_id}</b><br/>"
        "Max probability: {cell_max_prob}<br/>"
        "Mean probability: {cell_mean_prob}<br/>"
        "Positive pixels: {cell_positive_pixels}<br/>"
        "Final positive: {final_positive}",
        "style": {"backgroundColor": "white", "color": "black"},
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(
            latitude=view_state_values["latitude"],
            longitude=view_state_values["longitude"],
            zoom=view_state_values["zoom"],
        ),
        tooltip=tooltip,
        map_style=None,
    )
    st.pydeck_chart(deck, use_container_width=True)


def render_metrics(gdf: gpd.GeoDataFrame, filtered: gpd.GeoDataFrame) -> None:
    """Render summary metric cards."""
    total_cells = int(len(gdf))
    shown_cells = int(len(filtered))
    final_positive = int(filtered["final_positive"].sum()) if not filtered.empty else 0
    max_prob = float(filtered["cell_max_prob"].max()) if not filtered.empty else 0.0
    mean_prob = float(filtered["cell_max_prob"].mean()) if not filtered.empty else 0.0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total cells", total_cells)
    col2.metric("Shown cells", shown_cells)
    col3.metric("Final positives", final_positive)
    col4.metric("Max probability", f"{max_prob:.3f}")
    col5.metric("Mean max-prob", f"{mean_prob:.3f}")


def render_sidebar() -> tuple[str, str, float, bool]:
    """Render sidebar controls and return selected settings."""
    st.sidebar.header("Inputs")
    geojson_path = st.sidebar.text_input("Candidate-cell GeoJSON", DEFAULT_GEOJSON_PATH)
    summary_path = st.sidebar.text_input("Summary JSON", DEFAULT_SUMMARY_JSON_PATH)

    st.sidebar.header("Filters")
    min_probability = st.sidebar.slider(
        "Minimum cell max probability",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
    )
    show_only_final_positive = st.sidebar.checkbox("Show only final-positive cells", value=False)

    return geojson_path, summary_path, min_probability, show_only_final_positive


def main() -> None:
    st.title("Alberta Wildfire Ignition Risk Dashboard")
    st.caption("Prototype dashboard for Model B candidate cells refined by Model A.")

    geojson_path, summary_path, min_probability, show_only_final_positive = render_sidebar()

    try:
        gdf = load_candidate_cells(geojson_path)
    except Exception as exc:  # noqa: BLE001 - show user-friendly dashboard error.
        st.error(str(exc))
        st.stop()

    summary = load_summary_json(summary_path)
    filtered = filter_candidate_cells(
        gdf=gdf,
        min_probability=min_probability,
        show_only_final_positive=show_only_final_positive,
    )

    st.subheader("Summary")
    render_metrics(gdf, filtered)

    with st.expander("Input metadata", expanded=False):
        st.write("GeoJSON path:", geojson_path)
        st.write("Input CRS:", str(gdf.crs))
        st.write("Map display CRS:", WEB_CRS)
        if summary:
            st.json(summary)
        else:
            st.info("No summary JSON found. The dashboard can still run from the GeoJSON.")

    st.subheader("Candidate-cell map")
    if filtered.empty:
        st.warning("No candidate cells match the current filters.")
    else:
        render_pydeck_map(filtered)

    st.subheader("Candidate-cell table")
    table = build_display_dataframe(filtered)
    st.dataframe(table, use_container_width=True, hide_index=True)

    csv_bytes = table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered candidate table",
        data=csv_bytes,
        file_name="filtered_model_a_candidate_cells.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
