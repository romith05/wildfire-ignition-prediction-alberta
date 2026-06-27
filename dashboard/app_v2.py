
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
    from folium.plugins import HeatMap
    from streamlit_folium import st_folium
except Exception:
    folium = None
    HeatMap = None
    st_folium = None



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

# Alberta map display bounds.

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


def probability_heat_color(probability) -> str:
    """Continuous heatmap-like color for Model A candidate-cell probability."""
    p = pd.to_numeric(probability, errors="coerce")
    if pd.isna(p):
        return "#71717A"
    p = float(max(0.0, min(1.0, p)))
    if p >= 0.90:
        return "#7F0000"
    if p >= 0.80:
        return "#D7301F"
    if p >= 0.70:
        return "#FC8D59"
    if p >= 0.60:
        return "#FDBB84"
    if p >= 0.50:
        return "#FEE08B"
    if p >= 0.30:
        return "#D9EF8B"
    return "#91CF60"


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

    for col in ["cell_max_prob", "probability"]:
        if col in cells.columns:
            cells[col] = pd.to_numeric(cells[col], errors="coerce")

    if "cell_max_prob" not in cells.columns and "probability" in cells.columns:
        cells["cell_max_prob"] = cells["probability"]

    if "cell_max_prob" not in cells.columns:
        cells["cell_max_prob"] = pd.NA

    if "final_positive" in cells.columns:
        cells["final_positive"] = pd.to_numeric(
            cells["final_positive"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        cells["final_positive"] = 0

    cells["risk_tier"] = cells["cell_max_prob"].apply(risk_tier)
    return cells.to_crs("EPSG:4326")


ALBERTA_BOUNDS = [[48.85, -120.35], [60.15, -109.45]]
ALBERTA_CENTER = [54.7, -115.0]
ALBERTA_MIN_ZOOM = 5  # Set to 6 to tightly wrap the province borders on wide layouts
ALBERTA_MAX_ZOOM = 18

def create_bounded_alberta_map():
    """Create an OpenStreetMap map strictly locked and constrained to Alberta."""
    m = folium.Map(
        location=ALBERTA_CENTER,
        tiles="OpenStreetMap",
        min_zoom=ALBERTA_MIN_ZOOM,
        max_zoom=ALBERTA_MAX_ZOOM,
        control_scale=True,
        prefer_canvas=True,
        max_bounds=True,
    )
    
    # Force the initial viewport window to tightly frame the boundary box
    m.fit_bounds(ALBERTA_BOUNDS)

    folium.Rectangle(
        bounds=ALBERTA_BOUNDS,
        color="#111111",
        weight=1,
        fill=False,
        opacity=0.65,
        name="Alberta map boundary",
        show=True,
    ).add_to(m)

    map_name = m.get_name()
    bounds_js = "[[48.85, -120.35], [60.15, -109.45]]"

    # Strict boundary enforcement script to eliminate dragging or scrolling out
    custom_js = f"""
    setTimeout(function() {{
        var map = {map_name};
        var bounds = L.latLngBounds({bounds_js});

        map.setMaxBounds(bounds);
        map.options.maxBoundsViscosity = 1.0; // Removes the bouncy/elastic drag margin
        map.setMinZoom({ALBERTA_MIN_ZOOM});

        // Ensure current zoom isn't accidentally broken during st_folium mounting
        if (map.getZoom() < {ALBERTA_MIN_ZOOM}) {{
            map.setZoom({ALBERTA_MIN_ZOOM});
        }}

        // Catch map adjustments instantly and snap back to safety bounds
        map.on('zoomend dragend moveend', function() {{
            if (map.getZoom() < {ALBERTA_MIN_ZOOM}) {{
                map.setZoom({ALBERTA_MIN_ZOOM});
            }}
            map.panInsideBounds(bounds, {{animate: false}});
        }});
    }}, 200);
    """
    m.get_root().script.add_child(folium.Element(custom_js))

    return m


def _heatmap_points_from_cells(cells: gpd.GeoDataFrame, max_features: int) -> list[list[float]]:
    if cells.empty or "cell_max_prob" not in cells.columns:
        return []

    layer = cells.copy()
    layer["cell_max_prob"] = pd.to_numeric(layer["cell_max_prob"], errors="coerce").fillna(0).clip(0, 1)

    if len(layer) > max_features:
        layer = layer.sort_values("cell_max_prob", ascending=False).head(max_features).copy()

    pts = layer.geometry.representative_point()

    return [
        [float(pt.y), float(pt.x), float(prob)]
        for pt, prob in zip(pts, layer["cell_max_prob"])
        if float(prob) > 0
    ]


def add_model_a_probability_heatmap(m, cells: gpd.GeoDataFrame, max_features: int) -> None:
    if cells.empty or HeatMap is None:
        return

    heat_data = _heatmap_points_from_cells(cells, max_features)
    if not heat_data:
        return

    HeatMap(
        heat_data,
        name="Model A probability heatmap",
        min_opacity=0.22,
        max_opacity=0.86,
        radius=24,
        blur=20,
        max_zoom=9,
        gradient={
            0.10: "#91CF60",
            0.35: "#D9EF8B",
            0.55: "#FEE08B",
            0.70: "#FDBB84",
            0.85: "#FC8D59",
            1.00: "#D7301F",
        },
        show=True,
    ).add_to(m)


def infer_probability_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "model_b_probability",
        "probability",
        "score",
        "risk_score",
        "prediction",
        "pred_prob",
        "candidate_probability",
        "max_probability",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    probability_like = [
        col for col in df.columns
        if ("prob" in col.lower() or "score" in col.lower())
        and pd.to_numeric(df[col], errors="coerce").notna().any()
    ]
    return probability_like[0] if probability_like else None



def add_model_b_candidate_probability_heatmap(m, model_b_candidates: pd.DataFrame, max_features: int) -> None:
    """Add Model B 1 km candidate probabilities as a heatmap.

    model_b_candidates.csv stores coordinates in EPSG:3979:
      cell_xmin, cell_ymin, cell_xmax, cell_ymax, probability

    This function converts the candidate-cell centers to EPSG:4326 for Folium.
    """
    required = {"cell_xmin", "cell_ymin", "cell_xmax", "cell_ymax", "probability"}
    if model_b_candidates.empty or HeatMap is None or not required.issubset(model_b_candidates.columns):
        return

    df = model_b_candidates.copy()

    for col in ["cell_xmin", "cell_ymin", "cell_xmax", "cell_ymax", "probability"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["cell_xmin", "cell_ymin", "cell_xmax", "cell_ymax", "probability"])
    if df.empty:
        return

    df["probability"] = df["probability"].clip(0.0, 1.0)
    df["x_center"] = (df["cell_xmin"] + df["cell_xmax"]) / 2.0
    df["y_center"] = (df["cell_ymin"] + df["cell_ymax"]) / 2.0

    if len(df) > max_features:
        df = df.sort_values("probability", ascending=False).head(max_features).copy()

    points = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["x_center"], df["y_center"]),
        crs="EPSG:3979",
    ).to_crs("EPSG:4326")

    heat_data = [
        [float(row.geometry.y), float(row.geometry.x), float(row["probability"])]
        for _, row in points.iterrows()
        if float(row["probability"]) > 0
    ]

    if not heat_data:
        return

    HeatMap(
        heat_data,
        name="Model B 1 km candidate probability heatmap",
        min_opacity=0.20,
        max_opacity=0.82,
        radius=34,
        blur=24,
        max_zoom=9,
        gradient={
            0.10: "#DBEAFE",
            0.35: "#93C5FD",
            0.55: "#3B82F6",
            0.75: "#1D4ED8",
            1.00: "#312E81",
        },
        show=False,
    ).add_to(m)


def add_model_b_candidate_grid_to_map(m, model_b_candidates: pd.DataFrame, max_features: int) -> None:
    """Add exact Model B 1 km candidate cells as a toggleable polygon layer."""
    required = {"cell_xmin", "cell_ymin", "cell_xmax", "cell_ymax", "probability"}
    if model_b_candidates.empty or not required.issubset(model_b_candidates.columns):
        return

    df = model_b_candidates.copy()

    for col in ["cell_xmin", "cell_ymin", "cell_xmax", "cell_ymax", "probability"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["cell_xmin", "cell_ymin", "cell_xmax", "cell_ymax", "probability"])
    if df.empty:
        return

    df["probability"] = df["probability"].clip(0.0, 1.0)

    if len(df) > max_features:
        df = df.sort_values("probability", ascending=False).head(max_features).copy()

    from shapely.geometry import box

    geoms = [
        box(row["cell_xmin"], row["cell_ymin"], row["cell_xmax"], row["cell_ymax"])
        for _, row in df.iterrows()
    ]

    cells = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:3979").to_crs("EPSG:4326")

    def style(feature):
        p = feature.get("properties", {})
        prob = p.get("probability", 0.0)
        color = "#312E81"
        try:
            prob = float(prob)
        except Exception:
            prob = 0.0

        if prob >= 0.90:
            color = "#312E81"
        elif prob >= 0.75:
            color = "#1D4ED8"
        elif prob >= 0.55:
            color = "#3B82F6"
        elif prob >= 0.35:
            color = "#93C5FD"
        else:
            color = "#DBEAFE"

        return {
            "fillColor": color,
            "color": color,
            "weight": 0.7,
            "fillOpacity": 0.30,
        }

    tooltip_fields = [
        col for col in ["candidate_id", "patch_id", "probability", "row", "col"]
        if col in cells.columns
    ]

    folium.GeoJson(
        cells,
        name="Exact Model B 1 km candidate grid",
        style_function=style,
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields) if tooltip_fields else None,
        show=False,
    ).add_to(m)

def add_candidate_cells_to_map(m, cells: gpd.GeoDataFrame, max_features: int) -> None:
    if cells.empty:
        return

    layer = cells.copy()
    if len(layer) > max_features:
        layer = layer.sort_values("cell_max_prob", ascending=False).head(max_features).copy()

    def style(feat):
        p = feat.get("properties", {})
        c = probability_heat_color(p.get("cell_max_prob"))
        return {
            "fillColor": c,
            "color": c,
            "weight": 0.7,
            "fillOpacity": 0.34,
        }

    tooltip_fields = [
        col for col in ["candidate_id", "cell_max_prob", "risk_tier", "final_positive"]
        if col in layer.columns
    ]

    folium.GeoJson(
        layer,
        name="Exact Model A probability grid",
        style_function=style,
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields) if tooltip_fields else None,
        show=False,
    ).add_to(m)


def add_probability_legend(m) -> None:
    legend_html = """
    <div style="
        position: fixed;
        bottom: 34px;
        left: 34px;
        z-index: 9999;
        background: rgba(255,255,255,0.94);
        border: 1px solid #cfcfcf;
        border-radius: 8px;
        padding: 10px 12px;
        font-family: Arial, sans-serif;
        font-size: 12px;
        color: #111;
        box-shadow: 0 4px 14px rgba(0,0,0,0.18);
    ">
        <div style="font-weight:700;margin-bottom:6px;">Probability heatmap</div>
        <div style="width:185px;height:12px;background:linear-gradient(to right,#91CF60,#D9EF8B,#FEE08B,#FDBB84,#FC8D59,#D7301F);border-radius:4px;"></div>
        <div style="display:flex;justify-content:space-between;margin-top:4px;">
            <span>Lower</span><span>Higher</span>
        </div>
        <div style="margin-top:6px;color:#444;">Model A shown by default. Model B is toggleable.</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def add_fire_points_to_map(m, df: pd.DataFrame, name: str, color: str, radius: int = 5) -> None:
    if df.empty:
        return

    lat_col, lon_col = infer_lat_lon_columns(df)
    if lat_col is None or lon_col is None:
        return

    group = folium.FeatureGroup(name=name, show=True)

    for _, row in df.iterrows():
        lat = pd.to_numeric(row.get(lat_col), errors="coerce")
        lon = pd.to_numeric(row.get(lon_col), errors="coerce")

        if pd.isna(lat) or pd.isna(lon):
            continue

        fid = (
            row.get("fire_id")
            or row.get("fire_id_normalized")
            or row.get("Fire_Name")
            or "Fire"
        )

        popup = [f"<b>{fid}</b>"]
        for field in [
            "lead_time_hours",
            "model_a_positive_nearest_distance_m",
            "model_b_candidate_nearest_distance_m",
        ]:
            if field in df.columns and pd.notna(row.get(field)):
                popup.append(f"{field}: {row.get(field)}")

        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup="<br>".join(popup),
        ).add_to(group)

        if name.lower().startswith("validated"):
            folium.Circle(
                location=[float(lat), float(lon)],
                radius=25_000,
                color=color,
                fill=False,
                weight=1,
                opacity=0.4,
            ).add_to(group)

    group.add_to(m)


def render_map(run_id, validation_df, active_fires, model_b, max_features):
    if folium is None or st_folium is None:
        st.info("Install map dependencies: pip install streamlit-folium folium")
        return

    m = create_bounded_alberta_map()

    if run_id:
        cells = prepare_map_cells(RUNS_ROOT / run_id)
        add_model_a_probability_heatmap(m, cells, max_features=max_features)
        #add_candidate_cells_to_map(m, cells, max_features=max_features)
        add_model_b_candidate_probability_heatmap(m, model_b, max_features=max_features)
        #add_model_b_candidate_grid_to_map(m, model_b, max_features=max_features)
        add_probability_legend(m)

    add_fire_points_to_map(m, active_fires, "Current active fires", "#7C3AED", radius=4)
    # add_fire_points_to_map(
    #     m,
    #     validated_fires(validation_df),
    #     "Validated prospective fires + 25 km buffer",
    #     "#15803D",
    #     radius=6,
    # )

    folium.LayerControl(collapsed=False).add_to(m)

    st_folium(
        m,
        height=650,
        use_container_width=True,
        returned_objects=[],
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
# Operational flow — image-like hover tile flow
# ----------------------------------------------------------------------------
FLOW_STEPS = [
    {
        "number": "1",
        "title": "3-hour automated cycle",
        "section": "Schedule",
        "accent": "#6b7280",
        "icon": "⏱",
        "detail": "The operational system runs every three hours. Each cycle refreshes fire data, validates new fires, and creates a fresh prediction snapshot.",
    },
    {
        "number": "2",
        "title": "Refresh Alberta active-fire feed",
        "section": "Validation",
        "accent": "#7c3aed",
        "icon": "🔥",
        "detail": "Downloads the latest Alberta active-fire feed and stores the current fire snapshot for validation and dashboard display.",
    },
    {
        "number": "3",
        "title": "Prospective validation",
        "section": "Validation",
        "accent": "#7c3aed",
        "icon": "✓",
        "detail": "Only newly detected fires are validated. Each fire is compared against prediction snapshots created before the reported fire start time.",
    },
    {
        "number": "4",
        "title": "Append validation results",
        "section": "Validation",
        "accent": "#7c3aed",
        "icon": "📋",
        "detail": "Adds lead time, nearest prediction distance, and hit status at 1, 5, 10, and 25 km to the prospective validation log.",
    },
    {
        "number": "5",
        "title": "Start latest prediction snapshot",
        "section": "Prediction",
        "accent": "#2563eb",
        "icon": "▶",
        "detail": "Starts a new full-province prediction run after validation. This saved run becomes the next candidate snapshot for future validation.",
    },
    {
        "number": "6",
        "title": "Live weather + geospatial inputs",
        "section": "Prediction",
        "accent": "#2563eb",
        "icon": "🌦",
        "detail": "Combines live weather variables with static geospatial layers such as elevation, landcover, roads, water, and municipal bands.",
    },
    {
        "number": "7",
        "title": "Model B: 1 km gatekeeper",
        "section": "Screening",
        "accent": "#1d4ed8",
        "icon": "🔎",
        "detail": "Scans the province at 1 km resolution and identifies coarse candidate cells with elevated ignition susceptibility.",
    },
    {
        "number": "8",
        "title": "Candidate 1 km cells",
        "section": "Screening",
        "accent": "#1d4ed8",
        "icon": "▦",
        "detail": "Cells passing the Model B threshold are carried forward as broad regional candidates. These are screening outputs, not final ignition masks.",
    },
    {
        "number": "9",
        "title": "Model A: 25 m spatial refinement",
        "section": "Refinement",
        "accent": "#0f766e",
        "icon": "◎",
        "detail": "Runs only on Model B candidate areas and provides finer local spatial assessment at 25 m resolution.",
    },
    {
        "number": "10",
        "title": "Export outputs",
        "section": "Publish",
        "accent": "#15803d",
        "icon": "⬇",
        "detail": "Writes CSV, GeoJSON, and summary files into the results folder. These files power the map, metrics, and diagnostics.",
    },
    {
        "number": "11",
        "title": "Update latest-run pointer",
        "section": "Publish",
        "accent": "#15803d",
        "icon": "↻",
        "detail": "Updates the latest-run pointer so the dashboard automatically reads the newest completed prediction snapshot.",
    },
    {
        "number": "12",
        "title": "Conference dashboard",
        "section": "Display",
        "accent": "#ea580c",
        "icon": "▣",
        "detail": "Read-only web view showing the latest map, prospective validation metrics, active-fire table, and research diagnostics.",
    },
]


def render_interactive_operational_flow() -> None:
    """Pixel-faithful reproduction of the operational-flow diagram.
    Independent style — does not depend on or affect the rest of the dashboard CSS.
    """

    # Palette per stage group (matches the reference image)
    GRAY   = {"border": "#9CA3AF", "fill": "#F3F4F6", "badge": "#6B7280"}
    PURPLE = {"border": "#A78BFA", "fill": "#F5F3FF", "badge": "#7C3AED"}
    BLUE_L = {"border": "#93C5FD", "fill": "#EFF6FF", "badge": "#2563EB"}
    BLUE_D = {"border": "#2563EB", "fill": "#DBEAFE", "badge": "#1D4ED8"}
    GREEN  = {"border": "#34D399", "fill": "#ECFDF5", "badge": "#15803D"}
    ORANGE = {"border": "#FB923C", "fill": "#FFF7ED", "badge": "#EA580C"}

    # Top-row steps (1..7, 9) — step 8 sits below step 7 as a dashed callout
    top_row = [
        ("1", "🕐", ["3-hour", "automated cycle"], "", GRAY),
        ("2", "🔥", ["Refresh Alberta", "active-fire feed"], "", GRAY),
        ("3", "🔍", ["Prospective", "validation"],
            "compare new fires against earlier prediction snapshots", PURPLE),
        ("4", "📋", ["Append validation", "results"], "", PURPLE),
        ("5", "🗄️", ["Start latest", "prediction snapshot"], "", BLUE_L),
        ("6", "⛅", ["Live weather +", "geospatial inputs"], "", BLUE_L),
        ("7", "📍", ["Model B:", "1 km gatekeeper"], "", BLUE_D),
        ("9", "🗺️", ["Model A:", "25 m spatial refinement"], "", BLUE_D),
    ]

    # Geometry
    TILE_W, TILE_H, GAP = 160, 230, 38
    X0, Y0 = 40, 70  # top-left of first tile

    def tile_svg(x, y, w, h, num, icon, lines, sub, theme, dashed=False, badge_inside=True):
        stroke_dash = 'stroke-dasharray="8 6"' if dashed else ''
        # Card
        card = (
            f'<rect x="{x}" y="{y}" rx="14" ry="14" width="{w}" height="{h}" '
            f'fill="{theme["fill"]}" stroke="{theme["border"]}" stroke-width="2.4" {stroke_dash}/>'
        )
        # Icon
        icon_svg = (
            f'<text x="{x + w/2}" y="{y + 56}" text-anchor="middle" '
            f'font-size="34">{icon}</text>'
        )
        # Numbered badge (circle + number)
        badge_cx, badge_cy = x + w/2, y + 100
        badge = (
            f'<circle cx="{badge_cx}" cy="{badge_cy}" r="17" fill="{theme["badge"]}"/>'
            f'<text x="{badge_cx}" y="{badge_cy + 5}" text-anchor="middle" '
            f'fill="#fff" font-size="14" font-weight="800" '
            f'font-family="Inter, system-ui, sans-serif">{num}</text>'
        )
        # Title lines
        title = ""
        title_y = y + 145
        for i, line in enumerate(lines):
            title += (
                f'<text x="{x + w/2}" y="{title_y + i*20}" text-anchor="middle" '
                f'fill="#0F172A" font-size="14.5" font-weight="700" '
                f'font-family="Inter, system-ui, sans-serif">{line}</text>'
            )
        # Sub caption (wrapped)
        sub_svg = ""
        if sub:
            sub_y = title_y + len(lines) * 20 + 14
            words = sub.split()
            chunks, line_words, char_count = [], [], 0
            for w_ in words:
                if char_count + len(w_) > 22 and line_words:
                    chunks.append(" ".join(line_words))
                    line_words, char_count = [w_], len(w_)
                else:
                    line_words.append(w_)
                    char_count += len(w_) + 1
            if line_words:
                chunks.append(" ".join(line_words))
            for i, chunk in enumerate(chunks):
                sub_svg += (
                    f'<text x="{x + w/2}" y="{sub_y + i*14}" text-anchor="middle" '
                    f'fill="#475569" font-size="11" font-style="italic" '
                    f'font-family="Inter, system-ui, sans-serif">{chunk}</text>'
                )
        return card + icon_svg + badge + title + sub_svg

    # Build top row tiles + horizontal arrows between consecutive top-row tiles
    tiles_svg = ""
    arrows_svg = ""
    positions = []
    for i, (num, icon, lines, sub, theme) in enumerate(top_row):
        x = X0 + i * (TILE_W + GAP)
        y = Y0
        positions.append((x, y))
        tiles_svg += tile_svg(x, y, TILE_W, TILE_H, num, icon, lines, sub, theme)

    # Arrows between top row tiles (1→2, 2→3, ... 6→7, 7→9 with skip handled by index)
    for i in range(len(top_row) - 1):
        x_from = positions[i][0] + TILE_W
        x_to = positions[i + 1][0]
        y_mid = Y0 + TILE_H / 2
        arrows_svg += (
            f'<line x1="{x_from + 2}" y1="{y_mid}" x2="{x_to - 6}" y2="{y_mid}" '
            f'stroke="#334155" stroke-width="2.2" marker-end="url(#arrow)"/>'
        )

    # ── Step 8: dashed callout below step 7
    step7_x, step7_y = positions[6]
    step9_x, step9_y = positions[7]
    callout_x = step7_x + TILE_W / 2 - 95
    callout_y = step7_y + TILE_H + 70
    callout_w, callout_h = 190, 110
    callout = (
        f'<rect x="{callout_x}" y="{callout_y}" rx="18" ry="18" '
        f'width="{callout_w}" height="{callout_h}" fill="#EFF6FF" '
        f'stroke="#2563EB" stroke-width="2.2" stroke-dasharray="6 5"/>'
        f'<text x="{callout_x + callout_w/2}" y="{callout_y + 40}" text-anchor="middle" '
        f'font-size="28">🎯</text>'
        f'<text x="{callout_x + callout_w/2}" y="{callout_y + 72}" text-anchor="middle" '
        f'fill="#0F172A" font-size="14" font-weight="700" '
        f'font-family="Inter, system-ui, sans-serif">Candidate</text>'
        f'<text x="{callout_x + callout_w/2}" y="{callout_y + 90}" text-anchor="middle" '
        f'fill="#0F172A" font-size="14" font-weight="700" '
        f'font-family="Inter, system-ui, sans-serif">1 km cells</text>'
    )
    arrow_7_to_8 = (
        f'<line x1="{step7_x + TILE_W/2}" y1="{step7_y + TILE_H + 2}" '
        f'x2="{step7_x + TILE_W/2}" y2="{callout_y - 6}" '
        f'stroke="#334155" stroke-width="2.2" marker-end="url(#arrow)"/>'
    )

    # ── Steps 10 & 11 stacked below step 9 (green)
    step10_x = step9_x
    step10_y = step9_y + TILE_H + 70
    step10 = tile_svg(step10_x, step10_y, TILE_W, TILE_H, "10", "📄",
                      ["Export outputs"], "CSV • GeoJSON • summary files", GREEN)
    arrow_9_to_10 = (
        f'<line x1="{step9_x + TILE_W/2}" y1="{step9_y + TILE_H + 2}" '
        f'x2="{step10_x + TILE_W/2}" y2="{step10_y - 6}" '
        f'stroke="#15803D" stroke-width="2.4" marker-end="url(#arrow-green)"/>'
    )
    step11_x = step10_x
    step11_y = step10_y + TILE_H + 60
    step11 = tile_svg(step11_x, step11_y, TILE_W, TILE_H, "11", "🔄",
                      ["Update", "latest-run pointer"], "", GREEN)
    arrow_10_to_11 = (
        f'<line x1="{step10_x + TILE_W/2}" y1="{step10_y + TILE_H + 2}" '
        f'x2="{step11_x + TILE_W/2}" y2="{step11_y - 6}" '
        f'stroke="#15803D" stroke-width="2.4" marker-end="url(#arrow-green)"/>'
    )

    # ── Step 12: orange container with 4 sub-tiles
    cont_x, cont_y = 280, step11_y + 40
    cont_w, cont_h = 1050, 240
    container = (
        f'<rect x="{cont_x}" y="{cont_y}" rx="20" ry="20" '
        f'width="{cont_w}" height="{cont_h}" fill="#FFF7ED" '
        f'stroke="#FB923C" stroke-width="2.6"/>'
        # Header
        f'<text x="{cont_x + 40}" y="{cont_y + 42}" font-size="26">🖥️</text>'
        f'<circle cx="{cont_x + 92}" cy="{cont_y + 34}" r="15" fill="#EA580C"/>'
        f'<text x="{cont_x + 92}" y="{cont_y + 39}" text-anchor="middle" fill="#fff" '
        f'font-size="13" font-weight="800" font-family="Inter, system-ui, sans-serif">12</text>'
        f'<text x="{cont_x + 118}" y="{cont_y + 41}" fill="#0F172A" font-size="18" '
        f'font-weight="800" font-family="Inter, system-ui, sans-serif">Conference dashboard</text>'
    )
    sub_tiles = [
        ("🗺️", ["Latest", "prediction map"]),
        ("📈", ["Prospective", "validation metrics"]),
        ("⚗️", ["Research", "diagnostics"]),
        ("📊", ["Active-fire", "table"]),
    ]
    sub_w, sub_h, sub_gap = 220, 130, 28
    sub_total = 4 * sub_w + 3 * sub_gap
    sub_start_x = cont_x + (cont_w - sub_total) / 2
    sub_y = cont_y + 80
    sub_svg = ""
    branch_svg = ""
    parent_x = cont_x + cont_w / 2
    parent_y = cont_y + 70
    for i, (icon, lines) in enumerate(sub_tiles):
        sx = sub_start_x + i * (sub_w + sub_gap)
        sub_svg += (
            f'<rect x="{sx}" y="{sub_y}" rx="14" ry="14" width="{sub_w}" height="{sub_h}" '
            f'fill="#FFFFFF" stroke="#FB923C" stroke-width="2"/>'
            f'<text x="{sx + 30}" y="{sub_y + 50}" font-size="26">{icon}</text>'
            f'<text x="{sx + 70}" y="{sub_y + 55}" fill="#0F172A" font-size="14" '
            f'font-weight="700" font-family="Inter, system-ui, sans-serif">{lines[0]}</text>'
            f'<text x="{sx + 70}" y="{sub_y + 75}" fill="#0F172A" font-size="14" '
            f'font-weight="700" font-family="Inter, system-ui, sans-serif">{lines[1]}</text>'
        )
        # Branch line from header center down to each sub-tile top
        cx = sx + sub_w / 2
        branch_svg += (
            f'<path d="M {parent_x} {parent_y} L {parent_x} {sub_y - 12} '
            f'L {cx} {sub_y - 12} L {cx} {sub_y - 2}" '
            f'fill="none" stroke="#EA580C" stroke-width="2" marker-end="url(#arrow-orange)"/>'
        )

    # Dashed purple arrow: step 4 → step 12
    s4_x = positions[3][0] + TILE_W / 2
    s4_y = positions[3][1] + TILE_H
    purple_arrow = (
        f'<path d="M {s4_x} {s4_y + 2} L {s4_x} {cont_y - 40} L {cont_x + 80} {cont_y - 40} '
        f'L {cont_x + 80} {cont_y - 4}" fill="none" stroke="#7C3AED" stroke-width="2.2" '
        f'stroke-dasharray="8 6" marker-end="url(#arrow-purple)"/>'
    )

    # Green arrow: step 11 → step 12 (right side into container)
    s11_cx = step11_x + TILE_W / 2
    s11_cy = step11_y + TILE_H / 2
    green_arrow = (
        f'<path d="M {step11_x - 6} {s11_cy} L {cont_x + cont_w + 30} {s11_cy} '
        f'L {cont_x + cont_w + 30} {cont_y + cont_h / 2} L {cont_x + cont_w + 4} '
        f'{cont_y + cont_h / 2}" fill="none" stroke="#15803D" stroke-width="2.4" '
        f'marker-end="url(#arrow-green)"/>'
    )

    # Assemble SVG
    svg_w = X0 + len(top_row) * (TILE_W + GAP) + 40
    svg_h = step11_y + TILE_H + 320

    svg = f"""
    <svg viewBox="0 0 {svg_w} {svg_h}" xmlns="http://www.w3.org/2000/svg"
         style="width:100%;height:auto;background:#FFFFFF;border-radius:8px;">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#334155"/>
        </marker>
        <marker id="arrow-green" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#15803D"/>
        </marker>
        <marker id="arrow-purple" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#7C3AED"/>
        </marker>
        <marker id="arrow-orange" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#EA580C"/>
        </marker>
      </defs>

      <!-- Title -->
      <text x="{svg_w/2}" y="42" text-anchor="middle"
            fill="#0F172A" font-size="30" font-weight="900"
            font-family="Inter, system-ui, sans-serif">
        Operational Flow of the Live Wildfire Ignition-Risk System
      </text>

      <!-- Top row tiles + arrows -->
      {tiles_svg}
      {arrows_svg}

      <!-- Step 8 callout + arrow from 7 -->
      {arrow_7_to_8}
      {callout}

      <!-- Steps 10, 11 (green) + arrows -->
      {arrow_9_to_10}
      {step10}
      {arrow_10_to_11}
      {step11}

      <!-- Dashed purple arrow 4 → 12 -->
      {purple_arrow}

      <!-- Green arrow 11 → 12 -->
      {green_arrow}

      <!-- Step 12 container + branches + sub-tiles -->
      {container}
      {branch_svg}
      {sub_svg}

      <!-- Footer caption -->
      <text x="{svg_w/2}" y="{svg_h - 24}" text-anchor="middle"
            fill="#64748B" font-size="14" font-style="italic"
            font-family="Inter, system-ui, sans-serif">
        Dashboard is read-only: it displays saved pipeline outputs and does not run inference live.
      </text>
    </svg>
    """

    # Hover layer only. This keeps the SVG flowchart style/layout unchanged.
    flow_details = {str(step["number"]): step["detail"] for step in FLOW_STEPS}
    flow_titles = {str(step["number"]): step["title"] for step in FLOW_STEPS}

    def html_escape(value) -> str:
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    def hotspot(num, x, y, w, h):
        detail = html_escape(flow_details.get(str(num), ""))
        title = html_escape(flow_titles.get(str(num), f"Step {num}"))

        if not detail:
            return ""

        left = 100.0 * x / svg_w
        top = 100.0 * y / svg_h
        width = 100.0 * w / svg_w
        height = 100.0 * h / svg_h

        return f"""
        <div class="flow-hotspot"
             style="left:{left:.4f}%; top:{top:.4f}%; width:{width:.4f}%; height:{height:.4f}%;">
            <div class="flow-popover">
                <div class="flow-popover-title">{title}</div>
                <div class="flow-popover-body">{detail}</div>
            </div>
        </div>
        """

    hotspots = []

    # Top-row tiles: steps 1,2,3,4,5,6,7,9.
    for i, (num, icon, lines, sub, theme) in enumerate(top_row):
        x, y = positions[i]
        hotspots.append(hotspot(num, x, y, TILE_W, TILE_H))

    # Step 8 dashed candidate-cell callout.
    hotspots.append(hotspot("8", callout_x, callout_y, callout_w, callout_h))

    # Steps 10 and 11.
    hotspots.append(hotspot("10", step10_x, step10_y, TILE_W, TILE_H))
    hotspots.append(hotspot("11", step11_x, step11_y, TILE_W, TILE_H))

    # Step 12 dashboard container.
    hotspots.append(hotspot("12", cont_x, cont_y, cont_w, cont_h))

    hotspots_html = "".join(hotspots)

    flow_html = f"""
    <style>
        .flow-shell {{
            background: #FFFFFF;
            padding: 20px 16px;
            overflow: visible;
        }}

        .flow-canvas {{
            position: relative;
            width: 100%;
            overflow: visible;
        }}

        .flow-canvas svg {{
            display: block;
            width: 100%;
            height: auto;
        }}

        .flow-hotspot {{
            position: absolute;
            background: rgba(255,255,255,0);
            cursor: help;
            z-index: 10;
            overflow: visible;
        }}

        .flow-hotspot:hover {{
            outline: 2px solid rgba(255,87,34,0.28);
            outline-offset: 3px;
            border-radius: 14px;
        }}

        .flow-popover {{
            position: absolute;
            left: 50%;
            top: 100%;
            transform: translateX(-50%) translateY(10px);
            width: 310px;
            max-width: 340px;
            background: #0F172A;
            color: #F8FAFC;
            border: 1px solid rgba(255,87,34,0.75);
            border-left: 5px solid #FF5722;
            border-radius: 8px;
            box-shadow: 0 18px 42px rgba(15,23,42,0.38);
            padding: 12px 14px;
            opacity: 0;
            visibility: hidden;
            pointer-events: none;
            transition: opacity 0.15s ease, transform 0.15s ease, visibility 0.15s ease;
            z-index: 999;
            font-family: Inter, system-ui, sans-serif;
        }}

        .flow-popover::before {{
            content: "";
            position: absolute;
            top: -9px;
            left: 50%;
            transform: translateX(-50%);
            border-left: 9px solid transparent;
            border-right: 9px solid transparent;
            border-bottom: 9px solid #FF5722;
        }}

        .flow-hotspot:hover .flow-popover {{
            opacity: 1;
            visibility: visible;
            transform: translateX(-50%) translateY(0);
        }}

        .flow-popover-title {{
            color: #FFB088;
            font-size: 12px;
            font-weight: 900;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 7px;
        }}

        .flow-popover-body {{
            color: #F8FAFC;
            font-size: 12px;
            line-height: 1.45;
        }}
    </style>

    <div class="flow-shell">
        <div class="flow-canvas">
            {svg}
            {hotspots_html}
        </div>
    </div>
    """

    components.html(
        flow_html,
        height=int(svg_h * 0.78) + 140,
        scrolling=True,
    )




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
    tab_flow, tab_map, tab_validation, tab_diagnostics, tab_active = st.tabs([
        "Operational flow",
        
        "Latest map",
        "Prospective validation",
        "Research diagnostics",
        "Active fires",
    ])

    with tab_flow:
        render_interactive_operational_flow()

    with tab_map:
        st.subheader("Latest model output map")
        render_map(run_id, validation_df, active_fires, model_b, max_features=max_map_features)

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
