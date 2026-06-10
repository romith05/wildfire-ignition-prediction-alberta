"""Validate predicted wildfire candidate cells against active-fire observations.

This utility is intentionally source-agnostic. It accepts:

- Model A candidate-cell prediction GeoJSON from
  ``src.geospatial.export_model_a_cells_geojson``.
- Active fire or hotspot observations as GeoJSON, Shapefile, or CSV with
  latitude/longitude columns.

It writes lightweight validation artifacts that can be used to make a live run
more credible without treating active-fire observations as perfect ignition
labels.

Example:
    python -m src.validation.validate_predictions_against_active_fires \
      --predictions-geojson results/runs/<RUN_ID>/model_a_candidate_cells.geojson \
      --active-fires data/validation/firms_viirs_alberta_last24h.csv \
      --active-crs EPSG:4326 \
      --run-id <RUN_ID> \
      --match-distance-m 5000 \
      --distance-thresholds-m 1000,5000,10000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


DEFAULT_MATCH_DISTANCE_M = 5000.0
DEFAULT_DISTANCE_THRESHOLDS_M = "1000,5000,10000"
DEFAULT_ACTIVE_CRS = "EPSG:4326"
PREDICTION_FILTER_FINAL_POSITIVE = "final-positive"
PREDICTION_FILTER_ALL = "all"

LATITUDE_CANDIDATES = ["latitude", "lat", "y", "LATITUDE", "LAT", "Y"]
LONGITUDE_CANDIDATES = ["longitude", "lon", "lng", "x", "LONGITUDE", "LON", "LNG", "X"]
ACTIVE_ID_CANDIDATES = [
    "fire_id",
    "fire_number",
    "fire_name",
    "incident_id",
    "id",
    "name",
    "FIRE_ID",
    "FIRE_NUMBER",
    "FIRE_NAME",
    "ID",
    "NAME",
]


def parse_distance_thresholds(value: str) -> list[float]:
    """Parse comma-separated distance thresholds in metres."""
    thresholds: list[float] = []
    for item in value.split(","):
        clean = item.strip()
        if not clean:
            continue
        threshold = float(clean)
        if threshold < 0:
            raise ValueError(f"Distance thresholds must be >= 0, got {threshold}")
        thresholds.append(threshold)
    if not thresholds:
        raise ValueError("At least one distance threshold is required.")
    return sorted(set(thresholds))


def first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    """Return the first column matching candidates, case-sensitive first then lower-case."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    lower_lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        found = lower_lookup.get(candidate.lower())
        if found is not None:
            return found
    return None


def read_predictions(path: str | Path, prediction_filter: str) -> gpd.GeoDataFrame:
    """Read candidate-cell prediction polygons and optionally keep final positives."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions GeoJSON not found: {path}")

    predictions = gpd.read_file(path)
    if predictions.empty:
        raise ValueError(f"Predictions GeoJSON is empty: {path}")
    if predictions.crs is None:
        raise ValueError(f"Predictions GeoJSON has no CRS: {path}")
    if "candidate_id" not in predictions.columns:
        raise ValueError("Predictions GeoJSON must contain candidate_id.")

    if prediction_filter == PREDICTION_FILTER_FINAL_POSITIVE:
        if "final_positive" not in predictions.columns:
            raise ValueError("Prediction filter final-positive requires final_positive column.")
        final_positive = pd.to_numeric(predictions["final_positive"], errors="coerce").fillna(0).astype(int)
        predictions = predictions[final_positive == 1].copy()
        if predictions.empty:
            raise ValueError("No final-positive prediction cells found after filtering.")
    elif prediction_filter != PREDICTION_FILTER_ALL:
        raise ValueError(f"Unknown prediction filter: {prediction_filter}")

    return predictions.reset_index(drop=True)


def read_active_csv(
    path: str | Path,
    active_crs: str,
    lat_column: str | None,
    lon_column: str | None,
) -> gpd.GeoDataFrame:
    """Read active-fire point observations from CSV."""
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Active-fire CSV is empty: {path}")

    columns = list(df.columns)
    lat_column = lat_column or first_existing_column(columns, LATITUDE_CANDIDATES)
    lon_column = lon_column or first_existing_column(columns, LONGITUDE_CANDIDATES)
    if not lat_column or not lon_column:
        raise ValueError(
            "Could not infer latitude/longitude columns from active-fire CSV. "
            "Pass --lat-column and --lon-column explicitly."
        )

    lat = pd.to_numeric(df[lat_column], errors="coerce")
    lon = pd.to_numeric(df[lon_column], errors="coerce")
    valid = lat.notna() & lon.notna()
    if not valid.any():
        raise ValueError(f"No valid latitude/longitude rows found in active-fire CSV: {path}")

    df = df.loc[valid].copy()
    geometries = [Point(float(x), float(y)) for x, y in zip(lon.loc[valid], lat.loc[valid], strict=True)]
    return gpd.GeoDataFrame(df, geometry=geometries, crs=active_crs).reset_index(drop=True)


def read_active_fires(
    path: str | Path,
    active_crs: str,
    lat_column: str | None,
    lon_column: str | None,
) -> gpd.GeoDataFrame:
    """Read active-fire observations from CSV or a geospatial vector file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Active-fire input not found: {path}")

    if path.suffix.lower() == ".csv":
        active = read_active_csv(path, active_crs=active_crs, lat_column=lat_column, lon_column=lon_column)
    else:
        active = gpd.read_file(path)
        if active.empty:
            raise ValueError(f"Active-fire vector file is empty: {path}")
        if active.crs is None:
            active = active.set_crs(active_crs)

    active = active[~active.geometry.is_empty & active.geometry.notna()].copy()
    if active.empty:
        raise ValueError(f"No valid active-fire geometries found: {path}")
    return active.reset_index(drop=True)


def active_identifier(row: pd.Series, index: int) -> str:
    """Choose a stable active-fire identifier when possible."""
    for column in ACTIVE_ID_CANDIDATES:
        if column in row.index and pd.notna(row[column]) and str(row[column]).strip():
            return str(row[column])
    return f"active_{index:06d}"


def nearest_geometry_distance(target_geometry: Any, source: gpd.GeoDataFrame) -> tuple[int, float]:
    """Return nearest source row index and distance in CRS units."""
    distances = source.geometry.distance(target_geometry)
    nearest_index = int(distances.idxmin())
    return nearest_index, float(distances.loc[nearest_index])


def build_active_matches(
    active: gpd.GeoDataFrame,
    predictions: gpd.GeoDataFrame,
    thresholds_m: list[float],
    match_distance_m: float,
) -> pd.DataFrame:
    """Create one row per active-fire observation with nearest prediction info."""
    rows: list[dict[str, Any]] = []
    for active_index, active_row in active.iterrows():
        nearest_prediction_index, distance_m = nearest_geometry_distance(active_row.geometry, predictions)
        prediction_row = predictions.loc[nearest_prediction_index]
        row: dict[str, Any] = {
            "active_index": int(active_index),
            "active_id": active_identifier(active_row, int(active_index)),
            "nearest_candidate_id": prediction_row.get("candidate_id", ""),
            "nearest_distance_m": distance_m,
            "matched_within_match_distance": int(distance_m <= match_distance_m),
        }
        for threshold in thresholds_m:
            row[f"within_{int(threshold)}m"] = int(distance_m <= threshold)
        rows.append(row)
    return pd.DataFrame(rows)


def build_prediction_matches(
    predictions: gpd.GeoDataFrame,
    active: gpd.GeoDataFrame,
    match_distance_m: float,
) -> pd.DataFrame:
    """Create one row per prediction with nearest active-fire info."""
    rows: list[dict[str, Any]] = []
    active_ids = [active_identifier(row, int(index)) for index, row in active.iterrows()]
    for prediction_index, prediction_row in predictions.iterrows():
        distances = active.geometry.distance(prediction_row.geometry)
        nearest_active_index = int(distances.idxmin())
        nearest_distance_m = float(distances.loc[nearest_active_index])
        count_within = int((distances <= match_distance_m).sum())
        rows.append(
            {
                "prediction_index": int(prediction_index),
                "candidate_id": prediction_row.get("candidate_id", ""),
                "final_positive": prediction_row.get("final_positive", ""),
                "cell_max_prob": prediction_row.get("cell_max_prob", ""),
                "nearest_active_id": active_ids[nearest_active_index],
                "nearest_active_distance_m": nearest_distance_m,
                "active_count_within_match_distance": count_within,
                "matched_within_match_distance": int(count_within > 0),
            }
        )
    return pd.DataFrame(rows)


def build_summary(
    predictions: gpd.GeoDataFrame,
    active: gpd.GeoDataFrame,
    active_matches: pd.DataFrame,
    prediction_matches: pd.DataFrame,
    thresholds_m: list[float],
    match_distance_m: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build compact validation summary metrics."""
    active_count = len(active_matches)
    prediction_count = len(prediction_matches)
    matched_predictions = int(prediction_matches["matched_within_match_distance"].sum()) if prediction_count else 0
    matched_active = int(active_matches["matched_within_match_distance"].sum()) if active_count else 0

    threshold_recall = {
        f"active_recall_within_{int(threshold)}m": (
            float(active_matches[f"within_{int(threshold)}m"].sum() / active_count) if active_count else None
        )
        for threshold in thresholds_m
    }

    bounds = [float(value) for value in predictions.total_bounds]
    return {
        "run_id": args.run_id or "",
        "predictions_geojson": str(args.predictions_geojson),
        "active_fires": str(args.active_fires),
        "prediction_filter": args.prediction_filter,
        "prediction_count": int(prediction_count),
        "active_fire_count": int(active_count),
        "match_distance_m": float(match_distance_m),
        "matched_prediction_count": matched_predictions,
        "unmatched_prediction_count": int(prediction_count - matched_predictions),
        "matched_active_fire_count": matched_active,
        "unmatched_active_fire_count": int(active_count - matched_active),
        "prediction_match_rate_proxy": float(matched_predictions / prediction_count) if prediction_count else None,
        "active_fire_recall_proxy": float(matched_active / active_count) if active_count else None,
        "mean_active_to_nearest_prediction_distance_m": (
            float(active_matches["nearest_distance_m"].mean()) if active_count else None
        ),
        "median_active_to_nearest_prediction_distance_m": (
            float(active_matches["nearest_distance_m"].median()) if active_count else None
        ),
        "mean_prediction_to_nearest_active_distance_m": (
            float(prediction_matches["nearest_active_distance_m"].mean()) if prediction_count else None
        ),
        "median_prediction_to_nearest_active_distance_m": (
            float(prediction_matches["nearest_active_distance_m"].median()) if prediction_count else None
        ),
        "distance_thresholds": threshold_recall,
        "prediction_crs": str(predictions.crs),
        "prediction_bounds": {
            "left": bounds[0],
            "bottom": bounds[1],
            "right": bounds[2],
            "top": bounds[3],
        },
        "notes": [
            "This is a near-real-time credibility check, not final scientific validation.",
            "Active fires and satellite hotspots are imperfect proxies for ignition labels.",
        ],
    }


def write_summary(summary: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def write_overlay_geojson(
    predictions: gpd.GeoDataFrame,
    active: gpd.GeoDataFrame,
    prediction_matches: pd.DataFrame,
    active_matches: pd.DataFrame,
    path: str | Path,
) -> None:
    """Write one mixed GeoJSON with prediction polygons and active-fire geometries."""
    prediction_overlay = predictions.copy()
    prediction_overlay = prediction_overlay.merge(
        prediction_matches,
        on="candidate_id",
        how="left",
        suffixes=("", "_validation"),
    )
    prediction_overlay["validation_layer"] = "prediction_cell"

    active_overlay = active.copy()
    active_overlay["active_index"] = list(range(len(active_overlay)))
    active_overlay = active_overlay.merge(active_matches, on="active_index", how="left")
    active_overlay["validation_layer"] = "active_fire"

    shared_columns = [
        "validation_layer",
        "candidate_id",
        "final_positive",
        "cell_max_prob",
        "active_id",
        "nearest_candidate_id",
        "nearest_active_id",
        "nearest_distance_m",
        "nearest_active_distance_m",
        "matched_within_match_distance",
        "geometry",
    ]

    for frame in [prediction_overlay, active_overlay]:
        for column in shared_columns:
            if column not in frame.columns:
                frame[column] = None

    overlay = pd.concat(
        [prediction_overlay[shared_columns], active_overlay[shared_columns]],
        ignore_index=True,
    )
    overlay_gdf = gpd.GeoDataFrame(overlay, geometry="geometry", crs=predictions.crs)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay_gdf.to_file(path, driver="GeoJSON")


def apply_default_outputs(args: argparse.Namespace) -> argparse.Namespace:
    if args.output_active_matches_csv is None:
        args.output_active_matches_csv = f"results/runs/{args.run_id}/validation_active_matches.csv"
    if args.output_prediction_matches_csv is None:
        args.output_prediction_matches_csv = f"results/runs/{args.run_id}/validation_prediction_matches.csv"
    if args.output_summary_json is None:
        args.output_summary_json = f"results/runs/{args.run_id}/validation_summary.json"
    if args.output_overlay_geojson is None:
        args.output_overlay_geojson = f"results/runs/{args.run_id}/validation_overlay.geojson"
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate prediction cells against active-fire observations.")
    parser.add_argument("--predictions-geojson", required=True, help="Model A candidate-cell prediction GeoJSON.")
    parser.add_argument("--active-fires", required=True, help="Active-fire observations as GeoJSON/Shapefile or CSV.")
    parser.add_argument("--run-id", required=True, help="Run ID used for default output paths.")
    parser.add_argument("--active-crs", default=DEFAULT_ACTIVE_CRS, help="CRS for active-fire CSV or vector files without CRS.")
    parser.add_argument("--lat-column", default=None, help="Latitude column for active-fire CSV input.")
    parser.add_argument("--lon-column", default=None, help="Longitude column for active-fire CSV input.")
    parser.add_argument(
        "--prediction-filter",
        choices=[PREDICTION_FILTER_FINAL_POSITIVE, PREDICTION_FILTER_ALL],
        default=PREDICTION_FILTER_FINAL_POSITIVE,
        help="Prediction cells to validate against active fires.",
    )
    parser.add_argument("--match-distance-m", type=float, default=DEFAULT_MATCH_DISTANCE_M)
    parser.add_argument("--distance-thresholds-m", default=DEFAULT_DISTANCE_THRESHOLDS_M)
    parser.add_argument("--output-active-matches-csv", default=None)
    parser.add_argument("--output-prediction-matches-csv", default=None)
    parser.add_argument("--output-summary-json", default=None)
    parser.add_argument("--output-overlay-geojson", default=None)
    return parser.parse_args()


def main() -> None:
    args = apply_default_outputs(parse_args())
    if args.match_distance_m < 0:
        raise ValueError("--match-distance-m must be >= 0")

    thresholds_m = parse_distance_thresholds(args.distance_thresholds_m)
    predictions = read_predictions(args.predictions_geojson, args.prediction_filter)
    active = read_active_fires(
        args.active_fires,
        active_crs=args.active_crs,
        lat_column=args.lat_column,
        lon_column=args.lon_column,
    ).to_crs(predictions.crs)

    active_matches = build_active_matches(active, predictions, thresholds_m, args.match_distance_m)
    prediction_matches = build_prediction_matches(predictions, active, args.match_distance_m)
    summary = build_summary(
        predictions=predictions,
        active=active,
        active_matches=active_matches,
        prediction_matches=prediction_matches,
        thresholds_m=thresholds_m,
        match_distance_m=args.match_distance_m,
        args=args,
    )

    Path(args.output_active_matches_csv).parent.mkdir(parents=True, exist_ok=True)
    active_matches.to_csv(args.output_active_matches_csv, index=False)
    Path(args.output_prediction_matches_csv).parent.mkdir(parents=True, exist_ok=True)
    prediction_matches.to_csv(args.output_prediction_matches_csv, index=False)
    write_summary(summary, args.output_summary_json)
    write_overlay_geojson(predictions, active, prediction_matches, active_matches, args.output_overlay_geojson)

    print("Active-fire validation complete")
    print(f"Run ID: {args.run_id}")
    print(f"Prediction cells evaluated: {summary['prediction_count']}")
    print(f"Active-fire observations: {summary['active_fire_count']}")
    print(f"Match distance: {args.match_distance_m:.1f} m")
    print(f"Matched prediction cells: {summary['matched_prediction_count']}")
    print(f"Matched active fires: {summary['matched_active_fire_count']}")
    print(f"Active-fire recall proxy: {summary['active_fire_recall_proxy']}")
    print(f"Prediction match-rate proxy: {summary['prediction_match_rate_proxy']}")
    print(f"Active matches CSV: {args.output_active_matches_csv}")
    print(f"Prediction matches CSV: {args.output_prediction_matches_csv}")
    print(f"Summary JSON: {args.output_summary_json}")
    print(f"Overlay GeoJSON: {args.output_overlay_geojson}")


if __name__ == "__main__":
    main()
