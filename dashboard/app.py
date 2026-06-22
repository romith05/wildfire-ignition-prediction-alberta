"""Conference dashboard for Alberta wildfire ignition-risk results.

The dashboard is intentionally read-only. It visualizes the latest saved outputs
from the automated prospective-validation pipeline and does not run inference,
modify model artifacts, or edit validation state.

Run from the repository root:
    streamlit run dashboard/app.py --server.port 8501
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import streamlit as st

try:  # Optional map dependencies used by the dashboard.
    import folium
    from streamlit_folium import st_folium
except Exception:  # pragma: no cover - handled in the UI.
    folium = None
    st_folium = None


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
RUNS_ROOT = RESULTS_ROOT / "runs"
VALIDATION_ROOT = RESULTS_ROOT / "validation"
DATA_VALIDATION_ROOT = REPO_ROOT / "data" / "validation"

LATEST_RUN_FILE = VALIDATION_ROOT / "latest_prediction_run.txt"
VALIDATION_LOG_CSV = VALIDATION_ROOT / "prospective_validation_log.csv"
ACTIVE_FIRES_CSV = DATA_VALIDATION_ROOT / "alberta_activefires_current.csv"

THRESHOLD_SWEEP_CSV = VALIDATION_ROOT / "model_a_threshold_sweep_summary.csv"
RANKING_DETAIL_CSV = VALIDATION_ROOT / "prospective_ranking_diagnostic_detail.csv"
RANKING_SUMMARY_CSV = VALIDATION_ROOT / "prospective_ranking_diagnostic_summary.csv"
FEATURE_COMPARISON_SUMMARY_CSV = VALIDATION_ROOT / "ranked_candidate_feature_comparison_summary.csv"
HARD_NEGATIVE_SUMMARY_CSV = VALIDATION_ROOT / "prospective_hard_negative_summary.csv"

DISTANCE_THRESHOLDS_M = [1000, 5000, 10000, 25000]


st.set_page_config(
    page_title="Alberta Wildfire Ignition Risk Dashboard",
    page_icon="🔥",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def read_text(path: str) -> str | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    return file_path.read_text(encoding="utf-8").strip()


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


@st.cache_data(show_spinner=False)
def read_geojson(path: str) -> gpd.GeoDataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return gpd.GeoDataFrame()
    return gpd.read_file(file_path)


def find_latest_run_id() -> str | None:
    latest = read_text(str(LATEST_RUN_FILE))
    if latest:
        return latest
    if not RUNS_ROOT.exists():
        return None
    run_dirs = sorted(path.name for path in RUNS_ROOT.iterdir() if path.is_dir())
    return run_dirs[-1] if run_dirs else None


def validated_fires(validation_df: pd.DataFrame) -> pd.DataFrame:
    if validation_df.empty:
        return validation_df
    if "status" not in validation_df.columns:
        return validation_df.copy()
    return validation_df[validation_df["status"].astype(str).str.lower().eq("validated")].copy()


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").dropna()


def hit_rate(validated: pd.DataFrame, prefix: str, threshold_m: int) -> tuple[int, int, float | None]:
    column = f"{prefix}_hit_within_{threshold_m}m"
    if column not in validated.columns or validated.empty:
        return 0, 0, None
    hits = pd.to_numeric(validated[column], errors="coerce").fillna(0).astype(int)
    total = int(len(hits))
    hit_count = int(hits.sum())
    return hit_count, total, None if total == 0 else hit_count / total


def fmt_count(value: int | float | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{int(value):,}"


def fmt_float(value: float | None, suffix: str = "", digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{digits}f}{suffix}"


def infer_lat_lon_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    lat_candidates = ["latitude", "Latitude", "lat", "LATITUDE", "LAT", "latitude_normalized"]
    lon_candidates = ["longitude", "Longitude", "lon", "lng", "LONGITUDE", "LON", "longitude_normalized"]
    lower_lookup = {column.lower(): column for column in df.columns}
    lat_col = next((column for column in lat_candidates if column in df.columns), None)
    lon_col = next((column for column in lon_candidates if column in df.columns), None)
    if lat_col is None:
        lat_col = next((lower_lookup.get(column.lower()) for column in lat_candidates if lower_lookup.get(column.lower())), None)
    if lon_col is None:
        lon_col = next((lower_lookup.get(column.lower()) for column in lon_candidates if lower_lookup.get(column.lower())), None)
    return lat_col, lon_col


def risk_tier(probability: float | None) -> str:
    if probability is None or pd.isna(probability):
        return "Unknown"
    if probability >= 0.90:
        return "Very high"
    if probability >= 0.75:
        return "High"
    if probability >= 0.60:
        return "Medium"
    return "Candidate"


def risk_color(tier: str, final_positive: int | None = None) -> str:
    if final_positive == 1:
        return "#d73027"
    return {
        "Very high": "#d73027",
        "High": "#fc8d59",
        "Medium": "#fee08b",
        "Candidate": "#91bfdb",
        "Unknown": "#999999",
    }.get(tier, "#999999")


def prepare_map_cells(run_dir: Path) -> gpd.GeoDataFrame:
    geojson_path = run_dir / "model_a_candidate_cells.geojson"
    cells = read_geojson(str(geojson_path))
    if cells.empty:
        return cells
    if cells.crs is None:
        st.warning(f"Map layer has no CRS: {geojson_path}")
        return gpd.GeoDataFrame()

    cells = cells.copy()
    for prob_col in ["cell_max_prob", "probability"]:
        if prob_col in cells.columns:
            cells[prob_col] = pd.to_numeric(cells[prob_col], errors="coerce")
    if "cell_max_prob" not in cells.columns and "probability" in cells.columns:
        cells["cell_max_prob"] = cells["probability"]
    if "cell_max_prob" not in cells.columns:
        cells["cell_max_prob"] = pd.NA
    if "final_positive" in cells.columns:
        cells["final_positive"] = pd.to_numeric(cells["final_positive"], errors="coerce").fillna(0).astype(int)
    else:
        cells["final_positive"] = 0
    cells["risk_tier"] = cells["cell_max_prob"].apply(risk_tier)
    return cells.to_crs("EPSG:4326")


def add_candidate_cells_to_map(m: Any, cells: gpd.GeoDataFrame, max_features: int) -> None:
    if cells.empty:
        return
    layer = cells.copy()
    if len(layer) > max_features:
        layer = layer.sort_values("cell_max_prob", ascending=False).head(max_features).copy()

    def style_function(feature: dict[str, Any]) -> dict[str, Any]:
        props = feature.get("properties", {})
        final_positive = props.get("final_positive")
        tier = props.get("risk_tier", "Unknown")
        color = risk_color(tier, final_positive)
        return {"fillColor": color, "color": color, "weight": 1, "fillOpacity": 0.45 if final_positive == 1 else 0.28}

    tooltip_fields = [field for field in ["candidate_id", "cell_max_prob", "risk_tier", "final_positive"] if field in layer.columns]
    folium.GeoJson(
        layer,
        name="Model A candidate cells",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields) if tooltip_fields else None,
    ).add_to(m)


def add_fire_points_to_map(m: Any, df: pd.DataFrame, name: str, color: str, radius: int = 5) -> None:
    if df.empty:
        return
    lat_col, lon_col = infer_lat_lon_columns(df)
    if lat_col is None or lon_col is None:
        return
    group = folium.FeatureGroup(name=name, show=True)
    for row in df.itertuples(index=False):
        lat = pd.to_numeric(getattr(row, lat_col), errors="coerce")
        lon = pd.to_numeric(getattr(row, lon_col), errors="coerce")
        if pd.isna(lat) or pd.isna(lon):
            continue
        fire_id = getattr(row, "fire_id", None) or getattr(row, "fire_id_normalized", None) or getattr(row, "Fire_Name", None) or "Fire"
        popup_parts = [f"<b>{fire_id}</b>"]
        for field in ["lead_time_hours", "model_a_positive_nearest_distance_m", "model_b_candidate_nearest_distance_m"]:
            if field in df.columns:
                value = getattr(row, field, None)
                if pd.notna(value):
                    popup_parts.append(f"{field}: {value}")
        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup="<br>".join(popup_parts),
        ).add_to(group)
        if name.lower().startswith("validated"):
            folium.Circle(
                location=[float(lat), float(lon)],
                radius=25_000,
                color=color,
                fill=False,
                weight=1,
                opacity=0.45,
            ).add_to(group)
    group.add_to(m)


def render_map(run_id: str | None, validation_df: pd.DataFrame, active_fires: pd.DataFrame, max_features: int) -> None:
    if folium is None or st_folium is None:
        st.info("Install map dependencies to enable the interactive map: pip install streamlit-folium folium")
        return
    m = folium.Map(location=[54.7, -115.0], zoom_start=5, tiles="CartoDB positron")
    if run_id:
        cells = prepare_map_cells(RUNS_ROOT / run_id)
        add_candidate_cells_to_map(m, cells, max_features=max_features)
    add_fire_points_to_map(m, active_fires, "Current active fires", "#54278f", radius=4)
    add_fire_points_to_map(m, validated_fires(validation_df), "Validated prospective fires + 25 km buffer", "#006d2c", radius=6)
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, height=650, use_container_width=True)


def render_metric_cards(run_id: str | None, validation_df: pd.DataFrame, model_b: pd.DataFrame, model_a: pd.DataFrame) -> None:
    valid = validated_fires(validation_df)
    lead = numeric_series(valid, "lead_time_hours")
    dist = numeric_series(valid, "model_a_positive_nearest_distance_m")
    hit_count, total, hit_fraction = hit_rate(valid, "model_a_positive", 25000)

    final_positive_count = None
    if not model_a.empty and "final_positive" in model_a.columns:
        final_positive_count = int(pd.to_numeric(model_a["final_positive"], errors="coerce").fillna(0).astype(int).sum())

    cols = st.columns(6)
    cols[0].metric("Latest run", run_id or "Missing")
    cols[1].metric("Model B candidates", fmt_count(len(model_b)) if not model_b.empty else "—")
    cols[2].metric("Model A positives", fmt_count(final_positive_count))
    cols[3].metric("Validated fires", fmt_count(len(valid)))
    cols[4].metric("25 km hit rate", "—" if hit_fraction is None else f"{hit_count}/{total} ({100 * hit_fraction:.1f}%)")
    cols[5].metric("Median lead", fmt_float(float(lead.median()), " hr") if not lead.empty else "—")

    cols = st.columns(3)
    cols[0].metric("Median nearest distance", fmt_float(float(dist.median() / 1000.0), " km") if not dist.empty else "—")
    if not model_a.empty and "cell_max_prob" in model_a.columns:
        prob = pd.to_numeric(model_a["cell_max_prob"], errors="coerce").dropna()
        cols[1].metric("Median Model A probability", fmt_float(float(prob.median()), digits=3) if not prob.empty else "—")
    else:
        cols[1].metric("Median Model A probability", "—")
    cols[2].metric("Validation log rows", fmt_count(len(validation_df)) if not validation_df.empty else "—")


def render_validation_summary(validation_df: pd.DataFrame) -> None:
    valid = validated_fires(validation_df)
    if valid.empty:
        st.info("No prospective validation rows found yet.")
        return
    hit_rows = []
    for prefix, label in [("model_b_candidate", "Model B candidate"), ("model_a_candidate", "Model A candidate"), ("model_a_positive", "Model A positive")]:
        for threshold in DISTANCE_THRESHOLDS_M:
            hits, total, frac = hit_rate(valid, prefix, threshold)
            hit_rows.append({"source": label, "distance_km": int(threshold / 1000), "hits": hits, "total": total, "hit_rate_percent": None if frac is None else 100 * frac})
    hit_df = pd.DataFrame(hit_rows)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Prospective hit rates")
        st.dataframe(hit_df, use_container_width=True, hide_index=True)
        chart = hit_df[hit_df["source"].eq("Model A positive")].set_index("distance_km")[["hit_rate_percent"]]
        st.bar_chart(chart)
    with right:
        st.subheader("Validated fire table")
        useful_cols = [col for col in ["fire_id", "prediction_run_id", "lead_time_hours", "model_b_candidate_nearest_distance_m", "model_a_candidate_nearest_distance_m", "model_a_positive_nearest_distance_m"] if col in valid.columns]
        table = valid[useful_cols].copy() if useful_cols else valid.copy()
        for col in list(table.columns):
            if col.endswith("_distance_m"):
                table[col.replace("_m", "_km")] = pd.to_numeric(table[col], errors="coerce") / 1000.0
                table = table.drop(columns=[col])
        st.dataframe(table, use_container_width=True, hide_index=True)


def render_threshold_sweep() -> None:
    df = read_csv(str(THRESHOLD_SWEEP_CSV))
    st.subheader("Model A threshold sweep")
    if df.empty:
        st.info(f"Threshold sweep CSV not found yet: {THRESHOLD_SWEEP_CSV.relative_to(REPO_ROOT)}")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)
    x_col = next((col for col in ["threshold", "model_a_threshold"] if col in df.columns), None)
    if x_col:
        numeric_cols = [col for col in df.columns if col != x_col and pd.to_numeric(df[col], errors="coerce").notna().any()]
        hit_cols = [col for col in numeric_cols if "25" in col or "hit" in col.lower()]
        chart_cols = hit_cols[:2] or numeric_cols[:2]
        if chart_cols:
            st.line_chart(df[[x_col] + chart_cols].copy().set_index(x_col))


def render_ranking_diagnostic() -> None:
    st.subheader("Prospective ranking diagnostic")
    summary = read_csv(str(RANKING_SUMMARY_CSV))
    detail = read_csv(str(RANKING_DETAIL_CSV))
    if summary.empty and detail.empty:
        st.info("Ranking diagnostic CSVs are optional and were not found yet.")
        return
    if not summary.empty:
        st.caption("Summary")
        st.dataframe(summary, use_container_width=True, hide_index=True)
    if not detail.empty:
        st.caption("Per-fire detail")
        st.dataframe(detail, use_container_width=True, hide_index=True)
        rank_cols = [col for col in detail.columns if "rank" in col.lower()]
        if rank_cols:
            fire_col = "fire_id" if "fire_id" in detail.columns else detail.columns[0]
            st.bar_chart(detail[[fire_col, rank_cols[0]]].copy().set_index(fire_col))


def render_feature_comparison() -> None:
    st.subheader("Top-ranked vs near-fire feature comparison")
    df = read_csv(str(FEATURE_COMPARISON_SUMMARY_CSV))
    if df.empty:
        st.info(f"Feature comparison summary not found yet: {FEATURE_COMPARISON_SUMMARY_CSV.relative_to(REPO_ROOT)}")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)
    if {"feature", "top_minus_nearest"}.issubset(df.columns):
        st.bar_chart(df[["feature", "top_minus_nearest"]].copy().set_index("feature"))


def render_hard_negative_summary() -> None:
    st.subheader("Hard-negative mining summary")
    df = read_csv(str(HARD_NEGATIVE_SUMMARY_CSV))
    if df.empty:
        st.info(f"Hard-negative summary not found yet: {HARD_NEGATIVE_SUMMARY_CSV.relative_to(REPO_ROOT)}")
        return
    st.dataframe(df, use_container_width=True, hide_index=True)
    numeric = df[df["summary_type"].astype(str).eq("numeric")] if "summary_type" in df.columns else df
    if not numeric.empty and {"sample_role", "median_probability"}.issubset(numeric.columns):
        st.bar_chart(numeric[["sample_role", "median_probability"]].copy().set_index("sample_role"))


def render_active_fire_table(active_fires: pd.DataFrame) -> None:
    st.subheader("Current active-fire feed")
    if active_fires.empty:
        st.info(f"Active-fire CSV not found yet: {ACTIVE_FIRES_CSV.relative_to(REPO_ROOT)}")
        return
    columns = [col for col in ["Fire_Name", "Agency", "Start_Date", "Stage_of_Control", "Latitude", "Longitude", "Current_Size"] if col in active_fires.columns]
    st.dataframe(active_fires[columns] if columns else active_fires, use_container_width=True, hide_index=True)


def render_context_panel(run_id: str | None) -> None:
    st.markdown(
        """
        **Interpretation:** This dashboard shows saved outputs from the automated wildfire ignition-risk pipeline. The model is currently strongest as a regional ignition-risk screening system, not a precise ignition-point locator. Validation metrics are prospective only when a prediction run was created before the reported fire start time.
        """
    )
    if run_id:
        run_dir = RUNS_ROOT / run_id
        with st.expander("Latest run files"):
            for path in [run_dir / "model_b_scores.csv", run_dir / "model_b_candidates.csv", run_dir / "model_a_predictions.csv", run_dir / "model_a_candidate_cells.geojson", run_dir / "model_a_candidate_cells_summary.json"]:
                st.write("✅" if path.exists() else "⚠️", path.relative_to(REPO_ROOT))


def main() -> None:
    st.title("🔥 Alberta wildfire ignition-risk dashboard")
    st.caption("Conference view of the latest live-weather prediction snapshot and prospective validation results.")

    with st.sidebar:
        st.header("Dashboard controls")
        run_id = find_latest_run_id()
        manual_run = st.text_input("Prediction run ID", value=run_id or "")
        run_id = manual_run.strip() or run_id
        max_map_features = st.slider("Max mapped candidate cells", 50, 1000, 500, 50)
        st.write("Repository root")
        st.code(str(REPO_ROOT))
        st.write("Latest-run pointer")
        st.code(str(LATEST_RUN_FILE.relative_to(REPO_ROOT)))

    run_dir = RUNS_ROOT / run_id if run_id else None
    validation_df = read_csv(str(VALIDATION_LOG_CSV))
    active_fires = read_csv(str(ACTIVE_FIRES_CSV))
    model_b = read_csv(str(run_dir / "model_b_candidates.csv")) if run_dir else pd.DataFrame()
    model_a = read_csv(str(run_dir / "model_a_predictions.csv")) if run_dir else pd.DataFrame()

    render_context_panel(run_id)
    render_metric_cards(run_id, validation_df, model_b, model_a)

    tab_map, tab_validation, tab_diagnostics, tab_active = st.tabs(["Latest map", "Prospective validation", "Research diagnostics", "Active fires"])
    with tab_map:
        st.subheader("Latest model output map")
        render_map(run_id, validation_df, active_fires, max_features=max_map_features)
    with tab_validation:
        render_validation_summary(validation_df)
    with tab_diagnostics:
        diag_tabs = st.tabs(["Threshold sweep", "Ranking", "Feature comparison", "Hard negatives"])
        with diag_tabs[0]:
            render_threshold_sweep()
        with diag_tabs[1]:
            render_ranking_diagnostic()
        with diag_tabs[2]:
            render_feature_comparison()
        with diag_tabs[3]:
            render_hard_negative_summary()
    with tab_active:
        render_active_fire_table(active_fires)


if __name__ == "__main__":
    main()
