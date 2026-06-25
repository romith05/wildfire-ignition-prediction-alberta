
"""Conference dashboard for Alberta wildfire ignition-risk results.

Read-only: visualizes the latest saved outputs from the automated
prospective-validation pipeline. No inference, no edits.

Run from the repository root:
    streamlit run dashboard/app.py --server.port 8501
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import folium
    from streamlit_folium import st_folium
except Exception:
    folium = None
    st_folium = None

try:
    import pydeck as pdk
except Exception:
    pdk = None


# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Theme injection
# ----------------------------------------------------------------------------
def inject_theme() -> None:
    st.markdown(
        """
        <style>
          html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
          .block-container { padding-top: 1.6rem; }
          h1 { letter-spacing:-.03em; font-weight:900; }

          /* Tab styling */
          .stTabs [data-baseweb="tab-list"] {
              gap: 4px; border-bottom: 1px solid #27272A;
          }
          .stTabs [data-baseweb="tab"] {
              padding: 8px 16px; font-family: 'JetBrains Mono', monospace;
              font-size: 0.78rem; letter-spacing: 0.12em; text-transform: uppercase;
              color: #A1A1AA;
          }
          .stTabs [aria-selected="true"] {
              color: #FF5722 !important;
              border-bottom: 2px solid #FF5722 !important;
          }

          /* Default Streamlit metrics, when we still use them */
          [data-testid="stMetric"] {
              background: #121214; border: 1px solid #27272A;
              padding: 14px 16px; border-radius: 2px;
          }
          [data-testid="stMetricLabel"] {
              color: #A1A1AA !important; text-transform: uppercase;
              letter-spacing: .18em; font-size: .7rem !important;
          }
          [data-testid="stMetricValue"] {
              color: #FF5722; font-family: 'JetBrains Mono', monospace;
          }

          /* KPI custom cards */
          .kpi-card {
              background: #121214; border: 1px solid #27272A;
              border-left-width: 3px; padding: 14px 16px;
              border-radius: 2px; height: 100%;
          }
          .kpi-label {
              color: #A1A1AA; font-size: .68rem;
              letter-spacing: .18em; text-transform: uppercase;
              font-family: 'JetBrains Mono', monospace;
          }
          .kpi-value {
              color: #fff; font-family: 'JetBrains Mono', monospace;
              font-size: 1.7rem; font-weight: 800; margin-top: 4px;
              letter-spacing: -.02em;
          }
          .kpi-sub {
              color: #52525B; font-size: .72rem; margin-top: 3px;
              font-family: 'JetBrains Mono', monospace;
          }

          /* Freshness chip */
          .freshness-chip {
              display: inline-flex; align-items: center; gap: 8px;
              padding: 4px 10px; border: 1px solid #27272A;
              font-family: 'JetBrains Mono', monospace; font-size: .68rem;
              letter-spacing: .14em; text-transform: uppercase;
          }
          .pulse-dot {
              width: 7px; height: 7px; border-radius: 50%;
              animation: pulse 1.8s infinite;
          }
          @keyframes pulse {
              0%, 100% { opacity: 1; transform: scale(1); }
              50%      { opacity: .35; transform: scale(1.4); }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# File I/O (cached)
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_text(path: str) -> str | None:
    p = Path(path)
    return p.read_text(encoding="utf-8").strip() if p.exists() else None


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def read_geojson(path: str) -> gpd.GeoDataFrame:
    p = Path(path)
    return gpd.read_file(p) if p.exists() else gpd.GeoDataFrame()


def find_latest_run_id() -> str | None:
    latest = read_text(str(LATEST_RUN_FILE))
    if latest:
        return latest
    if not RUNS_ROOT.exists():
        return None
    run_dirs = sorted(p.name for p in RUNS_ROOT.iterdir() if p.is_dir())
    return run_dirs[-1] if run_dirs else None


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def validated_fires(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "status" not in df.columns:
        return df.copy() if not df.empty else df
    return df[df["status"].astype(str).str.lower().eq("validated")].copy()


def numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").dropna() if col in df.columns else pd.Series(dtype=float)


def hit_rate(df: pd.DataFrame, prefix: str, threshold_m: int) -> tuple[int, int, float | None]:
    col = f"{prefix}_hit_within_{threshold_m}m"
    if col not in df.columns or df.empty:
        return 0, 0, None
    hits = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    total = int(len(hits))
    n = int(hits.sum())
    return n, total, None if total == 0 else n / total


def fmt_count(v) -> str:
    return "—" if v is None or pd.isna(v) else f"{int(v):,}"


def fmt_float(v, suffix="", digits=2) -> str:
    return "—" if v is None or pd.isna(v) else f"{v:.{digits}f}{suffix}"


def infer_lat_lon_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    lat_c = ["latitude", "Latitude", "lat", "LATITUDE", "LAT", "latitude_normalized"]
    lon_c = ["longitude", "Longitude", "lon", "lng", "LONGITUDE", "LON", "longitude_normalized"]
    lat = next((c for c in lat_c if c in df.columns), None)
    lon = next((c for c in lon_c if c in df.columns), None)
    return lat, lon


def risk_tier(p) -> str:
    if p is None or pd.isna(p):
        return "Unknown"
    if p >= 0.90: return "Very high"
    if p >= 0.75: return "High"
    if p >= 0.60: return "Medium"
    return "Candidate"


def risk_color(tier: str, final_positive: int | None = None) -> str:
    if final_positive == 1:
        return "#FF3B30"
    return {
        "Very high": "#FF3B30",
        "High":      "#FF5722",
        "Medium":    "#FF9F0A",
        "Candidate": "#84CC16",
        "Unknown":   "#71717A",
    }.get(tier, "#71717A")


def hit_rate_accent(frac: float | None) -> str:
    if frac is None or pd.isna(frac): return "#71717A"
    if frac >= 0.90: return "#84CC16"
    if frac >= 0.70: return "#FF9F0A"
    return "#FF3B30"


def parse_run_timestamp(run_id: str | None) -> datetime | None:
    """Extract a timestamp from run ids like 'live_weather_province_current_20260625T1730Z'."""
    if not run_id:
        return None
    m = re.search(r"(\d{8})[T_](\d{4,6})", run_id)
    if not m:
        return None
    try:
        date = datetime.strptime(m.group(1) + m.group(2)[:4], "%Y%m%d%H%M")
        return date.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def humanize_age(dt: datetime | None) -> str:
    if dt is None:
        return "unknown"
    delta = datetime.now(timezone.utc) - dt
    secs = int(delta.total_seconds())
    if secs < 60:    return f"{secs}s ago"
    if secs < 3600:  return f"{secs // 60}m ago"
    if secs < 86400: return f"{secs // 3600}h {(secs % 3600) // 60}m ago"
    return f"{secs // 86400}d ago"


# ----------------------------------------------------------------------------
# KPI cards
# ----------------------------------------------------------------------------
def kpi_card(label: str, value: str, sub: str | None = None, accent: str = "#FF5722") -> None:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="kpi-card" style="border-left-color:{accent}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value" style="color:{accent}">{value}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_freshness_chip(run_id: str | None) -> None:
    ts = parse_run_timestamp(run_id)
    age_secs = (datetime.now(timezone.utc) - ts).total_seconds() if ts else None
    if age_secs is None:
        color, label = "#71717A", "Snapshot age unknown"
    elif age_secs < 4 * 3600:
        color, label = "#84CC16", f"Fresh · {humanize_age(ts)}"
    elif age_secs < 12 * 3600:
        color, label = "#FF9F0A", f"Stale · {humanize_age(ts)}"
    else:
        color, label = "#FF3B30", f"Outdated · {humanize_age(ts)}"
    st.markdown(
        f"""
        <div class="freshness-chip" style="color:{color};border-color:{color}55;background:{color}10">
            <span class="pulse-dot" style="background:{color}"></span> {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_strip(run_id: str | None, validation_df: pd.DataFrame,
                    model_b: pd.DataFrame, model_a: pd.DataFrame) -> None:
    valid = validated_fires(validation_df)
    lead = numeric_series(valid, "lead_time_hours")
    dist = numeric_series(valid, "model_a_positive_nearest_distance_m")
    hits, total, frac = hit_rate(valid, "model_a_positive", 25000)

    final_pos = None
    if not model_a.empty and "final_positive" in model_a.columns:
        final_pos = int(pd.to_numeric(model_a["final_positive"], errors="coerce").fillna(0).astype(int).sum())

    prob_median = None
    if not model_a.empty and "cell_max_prob" in model_a.columns:
        prob = pd.to_numeric(model_a["cell_max_prob"], errors="coerce").dropna()
        prob_median = float(prob.median()) if not prob.empty else None

    short_run = (run_id[:18] + "…") if run_id and len(run_id) > 20 else (run_id or "missing")
    hit_value = "—" if frac is None else f"{100 * frac:.1f}%"
    hit_sub = None if frac is None else f"{hits}/{total} fires"

    cols = st.columns(4)
    with cols[0]:
        kpi_card("Latest run", short_run,
                 sub=run_id if run_id else "no pointer",
                 accent="#A1A1AA")
    with cols[1]:
        kpi_card("Model B candidates", fmt_count(len(model_b)) if not model_b.empty else "—",
                 sub="1 km gatekeeper", accent="#2563EB")
    with cols[2]:
        kpi_card("Model A positives", fmt_count(final_pos),
                 sub="25 m refinement", accent="#FF5722")
    with cols[3]:
        kpi_card("25 km hit rate", hit_value, sub=hit_sub, accent=hit_rate_accent(frac))

    cols = st.columns(4)
    with cols[0]:
        kpi_card("Validated fires", fmt_count(len(valid)),
                 sub="prospective only", accent="#84CC16")
    with cols[1]:
        kpi_card("Median lead", fmt_float(float(lead.median()), " hr") if not lead.empty else "—",
                 sub="time before reported start", accent="#FF9F0A")
    with cols[2]:
        kpi_card("Median nearest dist.",
                 fmt_float(float(dist.median() / 1000.0), " km") if not dist.empty else "—",
                 sub="Model A positive → fire", accent="#FF5722")
    with cols[3]:
        kpi_card("Median Model A prob.", fmt_float(prob_median, digits=3),
                 sub="across candidate cells", accent="#A1A1AA")


# ----------------------------------------------------------------------------
# Map (Folium)
# ----------------------------------------------------------------------------
def prepare_map_cells(run_dir: Path) -> gpd.GeoDataFrame:
    cells = read_geojson(str(run_dir / "model_a_candidate_cells.geojson"))
    if cells.empty:
        return cells
    if cells.crs is None:
        st.warning("Map layer has no CRS")
        return gpd.GeoDataFrame()
    cells = cells.copy()
    for c in ["cell_max_prob", "probability"]:
        if c in cells.columns:
            cells[c] = pd.to_numeric(cells[c], errors="coerce")
    if "cell_max_prob" not in cells.columns and "probability" in cells.columns:
        cells["cell_max_prob"] = cells["probability"]
    if "cell_max_prob" not in cells.columns:
        cells["cell_max_prob"] = pd.NA
    cells["final_positive"] = (
        pd.to_numeric(cells["final_positive"], errors="coerce").fillna(0).astype(int)
        if "final_positive" in cells.columns else 0
    )
    cells["risk_tier"] = cells["cell_max_prob"].apply(risk_tier)
    return cells.to_crs("EPSG:4326")


def add_candidate_cells_to_map(m, cells, max_features):
    if cells.empty: return
    layer = cells.copy()
    if len(layer) > max_features:
        layer = layer.sort_values("cell_max_prob", ascending=False).head(max_features).copy()

    def style(feat):
        p = feat.get("properties", {})
        c = risk_color(p.get("risk_tier", "Unknown"), p.get("final_positive"))
        return {"fillColor": c, "color": c, "weight": 1,
                "fillOpacity": 0.55 if p.get("final_positive") == 1 else 0.32}

    tooltip_fields = [f for f in ["candidate_id", "cell_max_prob", "risk_tier", "final_positive"] if f in layer.columns]
    folium.GeoJson(
        layer, name="Model A candidate cells", style_function=style,
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields) if tooltip_fields else None,
    ).add_to(m)


def add_fire_points_to_map(m, df, name, color, radius=5):
    if df.empty: return
    lat_col, lon_col = infer_lat_lon_columns(df)
    if lat_col is None or lon_col is None: return
    group = folium.FeatureGroup(name=name, show=True)
    for row in df.itertuples(index=False):
        lat = pd.to_numeric(getattr(row, lat_col), errors="coerce")
        lon = pd.to_numeric(getattr(row, lon_col), errors="coerce")
        if pd.isna(lat) or pd.isna(lon): continue
        fid = (getattr(row, "fire_id", None) or getattr(row, "fire_id_normalized", None)
               or getattr(row, "Fire_Name", None) or "Fire")
        popup = [f"<b>{fid}</b>"]
        for f in ["lead_time_hours", "model_a_positive_nearest_distance_m",
                  "model_b_candidate_nearest_distance_m"]:
            if f in df.columns:
                v = getattr(row, f, None)
                if pd.notna(v): popup.append(f"{f}: {v}")
        folium.CircleMarker(
            location=[float(lat), float(lon)], radius=radius,
            color=color, fill=True, fill_color=color, fill_opacity=0.9,
            popup="<br>".join(popup),
        ).add_to(group)
        if name.lower().startswith("validated"):
            folium.Circle(location=[float(lat), float(lon)], radius=25_000,
                          color=color, fill=False, weight=1, opacity=0.4).add_to(group)
    group.add_to(m)


def render_map(run_id, validation_df, active_fires, max_features):
    if folium is None or st_folium is None:
        st.info("pip install streamlit-folium folium to enable the interactive map.")
        return
    m = folium.Map(location=[54.7, -115.0], zoom_start=5, tiles="CartoDB dark_matter")
    if run_id:
        cells = prepare_map_cells(RUNS_ROOT / run_id)
        add_candidate_cells_to_map(m, cells, max_features=max_features)
    add_fire_points_to_map(m, active_fires, "Current active fires", "#A78BFA", radius=4)
    add_fire_points_to_map(m, validated_fires(validation_df),
                           "Validated prospective fires + 25 km buffer", "#84CC16", radius=6)
    folium.LayerControl(collapsed=False).add_to(m)
    st_folium(m, height=650, use_container_width=True)


# ----------------------------------------------------------------------------
# 3D risk surface (pydeck)
# ----------------------------------------------------------------------------
def render_3d_risk_surface(run_id: str | None) -> None:
    if pdk is None:
        st.info("pip install pydeck to enable the 3D risk surface.")
        return
    if not run_id:
        st.info("No latest run selected.")
        return
    cells = prepare_map_cells(RUNS_ROOT / run_id)
    if cells.empty:
        st.info("No candidate cells available for this run.")
        return
    pts = cells.copy()
    pts["lon"] = pts.geometry.centroid.x
    pts["lat"] = pts.geometry.centroid.y
    pts["prob"] = pd.to_numeric(pts["cell_max_prob"], errors="coerce").fillna(0.0)
    pts = pts[pts["prob"] > 0]
    if pts.empty:
        st.info("All candidate probabilities are zero; nothing to render in 3D.")
        return

    layer = pdk.Layer(
        "HexagonLayer",
        data=pts[["lon", "lat", "prob"]],
        get_position=["lon", "lat"],
        get_elevation_weight="prob",
        elevation_scale=12000,
        elevation_range=[0, 80000],
        radius=8000,
        extruded=True,
        pickable=True,
        coverage=0.9,
        color_range=[
            [132, 204, 22],   # safe
            [255, 159, 10],   # warning
            [255, 87, 34],    # blaze
            [255, 59, 48],    # danger
        ],
    )
    view = pdk.ViewState(
        latitude=float(pts["lat"].mean()),
        longitude=float(pts["lon"].mean()),
        zoom=4.2, pitch=52, bearing=18,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        map_style="dark",
        tooltip={"text": "Ignition risk density\nlat: {position[1]}\nlon: {position[0]}"},
    )
    st.pydeck_chart(deck, use_container_width=True)
    st.caption(
        "Hex columns aggregate Model A candidate-cell probabilities. "
        "Height ∝ summed probability; color from low (lime) to extreme (red). "
        "Drag to rotate · Scroll to zoom · Shift+drag to tilt."
    )


# ----------------------------------------------------------------------------
# Validation timeline
# ----------------------------------------------------------------------------
def render_validation_timeline(validation_df: pd.DataFrame) -> None:
    df = validated_fires(validation_df).copy()
    if df.empty:
        return
    ts_col = next((c for c in ["validated_at", "validation_time", "report_date",
                               "fire_start_time"] if c in df.columns), None)
    if ts_col is None:
        return
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)
    if "model_a_positive_hit_within_25000m" not in df.columns:
        return
    df["hit_25km"] = pd.to_numeric(df["model_a_positive_hit_within_25000m"],
                                   errors="coerce").fillna(0)
    window = min(10, max(3, len(df) // 3))
    df["rolling_hit_rate"] = df["hit_25km"].rolling(window, min_periods=2).mean() * 100
    st.markdown("**Rolling 25 km hit rate** &nbsp;·&nbsp; "
                f"window: {window} fires")
    st.area_chart(df.set_index(ts_col)[["rolling_hit_rate"]], height=220)


# ----------------------------------------------------------------------------
# Validation tab content
# ----------------------------------------------------------------------------
def render_validation_summary(validation_df: pd.DataFrame) -> None:
    valid = validated_fires(validation_df)
    if valid.empty:
        st.info("No prospective validation rows yet. "
                "Validation runs append after each scheduled cycle.")
        return

    render_validation_timeline(validation_df)

    rows = []
    for prefix, label in [
        ("model_b_candidate", "Model B candidate"),
        ("model_a_candidate", "Model A candidate"),
        ("model_a_positive",  "Model A positive"),
    ]:
        for thr in DISTANCE_THRESHOLDS_M:
            h, t, f = hit_rate(valid, prefix, thr)
            rows.append({
                "source": label, "distance_km": int(thr / 1000),
                "hits": h, "total": t,
                "hit_rate_percent": None if f is None else round(100 * f, 1),
            })
    hit_df = pd.DataFrame(rows)

    left, right = st.columns(2)
    with left:
        st.subheader("Prospective hit rates")
        st.dataframe(hit_df, use_container_width=True, hide_index=True)
        chart = (hit_df[hit_df["source"].eq("Model A positive")]
                 .set_index("distance_km")[["hit_rate_percent"]])
        st.bar_chart(chart, height=220)

    with right:
        st.subheader("Validated fires")
        useful_cols = [c for c in [
            "fire_id", "prediction_run_id", "lead_time_hours",
            "model_b_candidate_nearest_distance_m",
            "model_a_candidate_nearest_distance_m",
            "model_a_positive_nearest_distance_m",
        ] if c in valid.columns]
        table = valid[useful_cols].copy() if useful_cols else valid.copy()
        for c in list(table.columns):
            if c.endswith("_distance_m"):
                table[c.replace("_m", "_km")] = pd.to_numeric(table[c], errors="coerce") / 1000.0
                table = table.drop(columns=[c])
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.download_button(
            "Export validated fires (CSV)",
            valid.to_csv(index=False).encode("utf-8"),
            file_name="validated_fires.csv",
            mime="text/csv",
        )


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------
def _csv_table(df: pd.DataFrame, csv_path: Path, missing_msg: str) -> None:
    if df.empty:
        st.info(missing_msg)
        return
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        f"Export {csv_path.name}",
        df.to_csv(index=False).encode("utf-8"),
        file_name=csv_path.name, mime="text/csv",
    )


def render_threshold_sweep() -> None:
    st.subheader("Model A threshold sweep")
    df = read_csv(str(THRESHOLD_SWEEP_CSV))
    if df.empty:
        st.info(f"Not generated yet: {THRESHOLD_SWEEP_CSV.relative_to(REPO_ROOT)}")
        return
    _csv_table(df, THRESHOLD_SWEEP_CSV, "missing")
    x_col = next((c for c in ["threshold", "model_a_threshold"] if c in df.columns), None)
    if x_col:
        numeric_cols = [c for c in df.columns
                        if c != x_col and pd.to_numeric(df[c], errors="coerce").notna().any()]
        hit_cols = [c for c in numeric_cols if "25" in c or "hit" in c.lower()]
        chart_cols = hit_cols[:2] or numeric_cols[:2]
        if chart_cols:
            st.line_chart(df[[x_col] + chart_cols].set_index(x_col), height=260)


def render_ranking_diagnostic() -> None:
    st.subheader("Prospective ranking diagnostic")
    summary = read_csv(str(RANKING_SUMMARY_CSV))
    detail = read_csv(str(RANKING_DETAIL_CSV))
    if summary.empty and detail.empty:
        st.info("Ranking diagnostic CSVs are optional and not generated yet.")
        return
    if not summary.empty:
        st.caption("Summary")
        _csv_table(summary, RANKING_SUMMARY_CSV, "missing")
    if not detail.empty:
        st.caption("Per-fire detail")
        _csv_table(detail, RANKING_DETAIL_CSV, "missing")
        rank_cols = [c for c in detail.columns if "rank" in c.lower()]
        if rank_cols:
            fire_col = "fire_id" if "fire_id" in detail.columns else detail.columns[0]
            st.bar_chart(detail[[fire_col, rank_cols[0]]].set_index(fire_col), height=240)


def render_feature_comparison() -> None:
    st.subheader("Top-ranked vs near-fire feature comparison")
    df = read_csv(str(FEATURE_COMPARISON_SUMMARY_CSV))
    if df.empty:
        st.info(f"Not generated yet: {FEATURE_COMPARISON_SUMMARY_CSV.relative_to(REPO_ROOT)}")
        return
    _csv_table(df, FEATURE_COMPARISON_SUMMARY_CSV, "missing")
    if {"feature", "top_minus_nearest"}.issubset(df.columns):
        st.bar_chart(df[["feature", "top_minus_nearest"]].set_index("feature"), height=300)


def render_hard_negative_summary() -> None:
    st.subheader("Hard-negative mining summary")
    df = read_csv(str(HARD_NEGATIVE_SUMMARY_CSV))
    if df.empty:
        st.info(f"Not generated yet: {HARD_NEGATIVE_SUMMARY_CSV.relative_to(REPO_ROOT)}")
        return
    _csv_table(df, HARD_NEGATIVE_SUMMARY_CSV, "missing")
    if "summary_type" in df.columns:
        numeric = df[df["summary_type"].astype(str).eq("numeric")]
        if not numeric.empty and {"sample_role", "median_probability"}.issubset(numeric.columns):
            st.bar_chart(numeric[["sample_role", "median_probability"]]
                         .set_index("sample_role"), height=240)


# ----------------------------------------------------------------------------
# Active fires
# ----------------------------------------------------------------------------
def render_active_fire_table(active_fires: pd.DataFrame) -> None:
    st.subheader("Current active-fire feed")
    if active_fires.empty:
        st.info(f"Active-fire CSV not found: {ACTIVE_FIRES_CSV.relative_to(REPO_ROOT)}")
        return
    cols = [c for c in ["Fire_Name", "Agency", "Start_Date", "Stage_of_Control",
                        "Latitude", "Longitude", "Current_Size"] if c in active_fires.columns]
    st.dataframe(active_fires[cols] if cols else active_fires,
                 use_container_width=True, hide_index=True)
    st.download_button(
        "Export active fires (CSV)",
        active_fires.to_csv(index=False).encode("utf-8"),
        file_name="active_fires.csv", mime="text/csv",
    )


# ----------------------------------------------------------------------------
# Operational flow — grouped stage bands
# ----------------------------------------------------------------------------
STAGES = [
    ("SCHEDULE", "#6b7280", [
        ("1", "3-hour automated cycle",
         "Runs every three hours: refresh active fires, validate, regenerate snapshot."),
    ]),
    ("VALIDATE", "#7c3aed", [
        ("2", "Refresh active-fire feed",
         "Downloads Alberta active-fire feed and stores current snapshot."),
        ("3", "Prospective validation",
         "New fires compared against snapshots created before reported start."),
        ("4", "Append validation log",
         "Logs lead time, nearest distance, and hit status at 1/5/10/25 km."),
    ]),
    ("PREDICT", "#2563eb", [
        ("5", "Start prediction snapshot",
         "Begin a new full-province prediction run for the next validation cycle."),
        ("6", "Live weather + geospatial",
         "Weather + static layers: elevation, landcover, roads, water, municipal bands."),
    ]),
    ("REFINE", "#0f766e", [
        ("7", "Model B · 1 km gatekeeper",
         "Coarse provincial scan flagging candidate cells."),
        ("8", "Candidate 1 km cells",
         "Cells above threshold carried forward — broad screening, not final."),
        ("9", "Model A · 25 m refinement",
         "Fine-grained confirmation over candidate areas."),
    ]),
    ("PUBLISH", "#15803d", [
        ("10", "Export outputs",
         "CSV, GeoJSON, summary files written to results/runs/."),
        ("11", "Update latest-run pointer",
         "Dashboard auto-reads the newest completed run."),
    ]),
    ("DISPLAY", "#ea580c", [
        ("12", "Conference dashboard",
         "Read-only display of saved outputs, validation, and diagnostics."),
    ]),
]


def render_interactive_operational_flow() -> None:
    bands_html = []
    for band_label, accent, steps in STAGES:
        cards = "".join(
            f"""
            <div class="flow-card" style="--accent:{accent}">
                <div class="step-badge">{n}</div>
                <div class="step-title">{title}</div>
                <div class="tooltip-box">{detail}</div>
            </div>
            """ for n, title, detail in steps
        )
        bands_html.append(f"""
        <div class="stage-band" style="--band:{accent}">
            <div class="stage-label">{band_label}</div>
            <div class="stage-cards">{cards}</div>
        </div>
        """)

    html = f"""
    <style>
        .flow-wrapper {{
            font-family: 'IBM Plex Sans', -apple-system, sans-serif;
            padding: 18px 4px 30px 4px;
            color: #fafafa;
        }}
        .flow-title {{
            font-family: 'Chivo', sans-serif;
            text-align: center; font-size: 28px; font-weight: 900;
            letter-spacing: -0.02em; color: #fafafa; margin-bottom: 6px;
        }}
        .flow-subtitle {{
            text-align: center; font-size: 12px; color: #71717A;
            font-family: 'JetBrains Mono', monospace; letter-spacing: 0.14em;
            text-transform: uppercase; margin-bottom: 24px;
        }}
        .stage-band {{
            display: flex; align-items: stretch; gap: 14px;
            margin-bottom: 14px; padding: 10px 12px;
            border-left: 3px solid var(--band);
            background: rgba(255,255,255,0.02);
        }}
        .stage-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px; letter-spacing: 0.28em;
            color: var(--band); writing-mode: vertical-rl;
            transform: rotate(180deg); padding: 6px 4px;
            font-weight: 700;
        }}
        .stage-cards {{
            display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr));
            gap: 12px; flex: 1;
        }}
        .flow-card {{
            position: relative;
            border: 1px solid var(--accent);
            background: #121214; padding: 14px 12px;
            min-height: 92px; text-align: left;
            transition: transform 0.15s, box-shadow 0.15s;
            cursor: help;
        }}
        .flow-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 10px 26px rgba(0,0,0,0.6);
            z-index: 30;
        }}
        .step-badge {{
            width: 24px; height: 24px;
            background: var(--accent); color: #fff;
            display: inline-flex; align-items: center; justify-content: center;
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px; font-weight: 800; margin-bottom: 8px;
        }}
        .step-title {{
            font-family: 'Chivo', sans-serif; font-size: 13px;
            font-weight: 700; line-height: 1.25; color: #fafafa;
        }}
        .tooltip-box {{
            display: none; position: absolute;
            left: 50%; top: 102%; transform: translateX(-50%);
            width: 260px; background: #0a0a0a; color: #fafafa;
            border: 1px solid var(--accent);
            padding: 12px 14px; font-size: 12px; line-height: 1.4;
            font-family: 'IBM Plex Sans', sans-serif;
        }}
        .flow-card:hover .tooltip-box {{ display: block; }}
        .flow-footer {{
            margin-top: 18px; text-align: center;
            color: #52525B; font-size: 11px;
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 0.18em; text-transform: uppercase;
        }}
        @media (max-width: 1100px) {{
            .stage-cards {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
    <div class="flow-wrapper">
      <div class="flow-title">Operational Flow · Live Wildfire Ignition-Risk System</div>
      <div class="flow-subtitle">Hover any step for detail</div>
      {''.join(bands_html)}
      <div class="flow-footer">Read-only dashboard · saved pipeline outputs only</div>
    </div>
    """
    components.html(html, height=860, scrolling=True)


# ----------------------------------------------------------------------------
# Sidebar status panel
# ----------------------------------------------------------------------------
def render_sidebar(run_id: str | None) -> tuple[str | None, int]:
    with st.sidebar:
        st.header("Controls")
        manual = st.text_input("Prediction run ID", value=run_id or "")
        selected_run = manual.strip() or run_id
        max_features = st.slider("Max mapped candidate cells", 50, 1000, 500, 50)

        st.divider()
        st.markdown("##### Pipeline file status")
        run_dir = (RUNS_ROOT / selected_run) if selected_run else None
        files = [
            ("Latest-run pointer", LATEST_RUN_FILE),
            ("Validation log",     VALIDATION_LOG_CSV),
            ("Active fires",       ACTIVE_FIRES_CSV),
            ("Model B candidates", run_dir / "model_b_candidates.csv" if run_dir else None),
            ("Model A predictions", run_dir / "model_a_predictions.csv" if run_dir else None),
            ("Candidate cells geo", run_dir / "model_a_candidate_cells.geojson" if run_dir else None),
        ]
        for label, path in files:
            mark = "✅" if (path and Path(path).exists()) else "⚠️"
            st.markdown(f"<span style='font-family:JetBrains Mono;font-size:12px'>"
                        f"{mark}&nbsp;{label}</span>", unsafe_allow_html=True)

        if selected_run:
            with st.expander("Latest run files (full paths)"):
                if run_dir:
                    for p in [run_dir / "model_b_scores.csv",
                              run_dir / "model_b_candidates.csv",
                              run_dir / "model_a_predictions.csv",
                              run_dir / "model_a_candidate_cells.geojson",
                              run_dir / "model_a_candidate_cells_summary.json"]:
                        st.write("✅" if p.exists() else "⚠️", p.relative_to(REPO_ROOT))

        st.divider()
        st.caption(f"Repo root\n`{REPO_ROOT}`")
    return selected_run, max_features


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    inject_theme()

    initial_run = find_latest_run_id()
    run_id, max_map_features = render_sidebar(initial_run)

    # Header row
    header_left, header_right = st.columns([5, 2])
    with header_left:
        st.markdown(
            "<h1 style='margin-bottom:0'>🔥 Alberta wildfire ignition-risk</h1>"
            "<div style='color:#A1A1AA;font-size:.85rem;margin-top:2px'>"
            "Conference view · latest live-weather snapshot + prospective validation"
            "</div>", unsafe_allow_html=True,
        )
    with header_right:
        st.markdown("<div style='text-align:right;padding-top:18px'>", unsafe_allow_html=True)
        render_freshness_chip(run_id)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        "<div style='color:#A1A1AA;font-size:.8rem;margin-top:14px;line-height:1.5'>"
        "<b style='color:#FF5722'>Interpretation:</b> read-only outputs from the automated "
        "ignition-risk pipeline. The model currently behaves as a regional ignition-risk "
        "screening system, not a precise ignition-point locator. Validation metrics are "
        "prospective only when the prediction snapshot pre-dates the reported fire start time."
        "</div>", unsafe_allow_html=True,
    )

    st.markdown("&nbsp;")

    # Load data
    run_dir = RUNS_ROOT / run_id if run_id else None
    validation_df = read_csv(str(VALIDATION_LOG_CSV))
    active_fires = read_csv(str(ACTIVE_FIRES_CSV))
    model_b = read_csv(str(run_dir / "model_b_candidates.csv")) if run_dir else pd.DataFrame()
    model_a = read_csv(str(run_dir / "model_a_predictions.csv")) if run_dir else pd.DataFrame()

    # KPI strip
    render_kpi_strip(run_id, validation_df, model_b, model_a)
    st.markdown("&nbsp;")

    # Tabs
    tab_flow, tab_3d, tab_map, tab_validation, tab_diagnostics, tab_active = st.tabs([
        "Operational flow",
        "3D risk surface",
        "Latest map",
        "Prospective validation",
        "Research diagnostics",
        "Active fires",
    ])

    with tab_flow:
        render_interactive_operational_flow()

    with tab_3d:
        st.subheader("3D ignition-risk density surface")
        render_3d_risk_surface(run_id)

    with tab_map:
        st.subheader("Latest model output map")
        render_map(run_id, validation_df, active_fires, max_features=max_map_features)

    with tab_validation:
        render_validation_summary(validation_df)

    with tab_diagnostics:
        diag_tabs = st.tabs(["Threshold sweep", "Ranking",
                             "Feature comparison", "Hard negatives"])
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
