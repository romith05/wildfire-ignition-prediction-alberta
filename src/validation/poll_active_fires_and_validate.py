"""Prospectively validate new Alberta fires against earlier prediction runs.

The script is designed for a three-hour polling cycle:

1. Read the latest active-fire CSV snapshot.
2. Keep Alberta agency records.
3. Compare incident IDs with a persistent seen-fire state file.
4. For every newly observed incident, select the latest completed prediction run
   created before the reported fire start time.
5. Measure distance to Model B candidate cells, all Model A candidate cells, and
   Model A final-positive cells.
6. Append one immutable prospective validation record per new incident.

The first execution must use ``--initialize-state``. This records the currently
active fires as the baseline without treating them as future ignitions.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box


RUN_TIMESTAMP_RE = re.compile(r"(?P<stamp>\d{8}_\d{6})$")
DEFAULT_THRESHOLDS_M = "1000,5000,10000,25000"
DEFAULT_STATE_PATH = "data/validation/state/seen_active_fires.json"
DEFAULT_REGISTRY_PATH = "data/validation/state/prediction_runs.csv"
DEFAULT_LOG_PATH = "results/validation/prospective_validation_log.csv"

FIRE_ID_CANDIDATES = [
    "Fire_Name",
    "fire_name",
    "FireNumber",
    "fire_number",
    "incident_id",
    "OBJECTID",
    "id",
]
LATITUDE_CANDIDATES = ["latitude", "Latitude", "lat", "LATITUDE", "LAT"]
LONGITUDE_CANDIDATES = ["longitude", "Longitude", "lon", "lng", "LONGITUDE", "LON"]


@dataclass(frozen=True)
class PredictionRun:
    run_id: str
    created_at: pd.Timestamp
    run_dir: Path
    model_b_candidates: Path
    model_a_geojson: Path


def utc_now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def parse_thresholds(value: str) -> list[float]:
    thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not thresholds or any(value < 0 for value in thresholds):
        raise ValueError("Distance thresholds must contain one or more non-negative values.")
    return thresholds


def first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    lower_lookup = {column.lower(): column for column in columns}
    for candidate in candidates:
        found = lower_lookup.get(candidate.lower())
        if found is not None:
            return found
    return None


def parse_run_created_at(run_id: str) -> pd.Timestamp | None:
    match = RUN_TIMESTAMP_RE.search(run_id)
    if match is None:
        return None
    parsed = datetime.strptime(match.group("stamp"), "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    return pd.Timestamp(parsed)


def discover_prediction_runs(runs_root: Path) -> list[PredictionRun]:
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")

    runs: list[PredictionRun] = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        created_at = parse_run_created_at(run_dir.name)
        if created_at is None:
            continue

        model_b_candidates = run_dir / "model_b_candidates.csv"
        model_a_geojson = run_dir / "model_a_candidate_cells.geojson"
        if not model_b_candidates.exists() or not model_a_geojson.exists():
            continue

        runs.append(
            PredictionRun(
                run_id=run_dir.name,
                created_at=created_at,
                run_dir=run_dir,
                model_b_candidates=model_b_candidates,
                model_a_geojson=model_a_geojson,
            )
        )

    runs.sort(key=lambda item: item.created_at)
    return runs


def write_prediction_registry(runs: list[PredictionRun], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "run_id": run.run_id,
            "created_at_utc": run.created_at.isoformat(),
            "run_dir": str(run.run_dir),
            "model_b_candidates": str(run.model_b_candidates),
            "model_a_geojson": str(run.model_a_geojson),
        }
        for run in runs
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def select_latest_prior_run(runs: list[PredictionRun], fire_start: pd.Timestamp) -> PredictionRun | None:
    eligible = [run for run in runs if run.created_at < fire_start]
    return eligible[-1] if eligible else None


def read_active_fires(
    csv_path: Path,
    agency: str,
    fire_id_column: str | None,
    start_date_column: str,
    start_date_unit: str,
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Active-fire CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        return df

    if agency:
        if "Agency" not in df.columns:
            raise ValueError("Agency filtering requested, but the CSV has no Agency column.")
        df = df[df["Agency"].astype(str).str.strip().str.upper().eq(agency.upper())].copy()

    columns = list(df.columns)
    fire_id_column = fire_id_column or first_existing_column(columns, FIRE_ID_CANDIDATES)
    latitude_column = first_existing_column(columns, LATITUDE_CANDIDATES)
    longitude_column = first_existing_column(columns, LONGITUDE_CANDIDATES)

    if fire_id_column is None:
        raise ValueError("Could not infer a stable fire ID column. Pass --fire-id-column.")
    if latitude_column is None or longitude_column is None:
        raise ValueError("Could not infer latitude/longitude columns from the active-fire CSV.")
    if start_date_column not in df.columns:
        raise ValueError(f"Start-date column not found: {start_date_column}")

    df["_fire_id"] = df[fire_id_column].astype(str).str.strip()
    df["_latitude"] = pd.to_numeric(df[latitude_column], errors="coerce")
    df["_longitude"] = pd.to_numeric(df[longitude_column], errors="coerce")
    df["_fire_start_utc"] = pd.to_datetime(
        df[start_date_column],
        unit=start_date_unit,
        utc=True,
        errors="coerce",
    )

    df = df[
        df["_fire_id"].ne("")
        & df["_latitude"].notna()
        & df["_longitude"].notna()
    ].copy()
    df = df.drop_duplicates(subset=["_fire_id"], keep="last").reset_index(drop=True)
    return df


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "last_poll_time_utc": None, "seen_fires": {}}
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    state.setdefault("version", 1)
    state.setdefault("last_poll_time_utc", None)
    state.setdefault("seen_fires", {})
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def load_model_b_candidates(path: Path) -> gpd.GeoDataFrame:
    df = pd.read_csv(path)
    required = ["cell_xmin", "cell_ymin", "cell_xmax", "cell_ymax", "crs"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Model B candidates missing required columns {missing}: {path}")

    crs_values = [str(value) for value in df["crs"].dropna().unique()]
    if len(crs_values) != 1:
        raise ValueError(f"Expected one CRS in Model B candidates, found {crs_values}: {path}")

    geometry = [
        box(float(row.cell_xmin), float(row.cell_ymin), float(row.cell_xmax), float(row.cell_ymax))
        for row in df.itertuples(index=False)
    ]
    return gpd.GeoDataFrame(df, geometry=geometry, crs=crs_values[0])


def load_model_a_candidates(path: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    all_candidates = gpd.read_file(path)
    if all_candidates.empty or all_candidates.crs is None:
        raise ValueError(f"Model A candidate GeoJSON is empty or has no CRS: {path}")
    if "final_positive" not in all_candidates.columns:
        raise ValueError(f"Model A candidate GeoJSON has no final_positive column: {path}")

    positive_mask = pd.to_numeric(all_candidates["final_positive"], errors="coerce").fillna(0).astype(int).eq(1)
    positives = all_candidates[positive_mask].copy()
    return all_candidates, positives


def nearest_distance_m(point_wgs84: Point, layer: gpd.GeoDataFrame) -> float | None:
    if layer.empty:
        return None
    point = gpd.GeoSeries([point_wgs84], crs="EPSG:4326").to_crs(layer.crs).iloc[0]
    distances = layer.geometry.distance(point)
    return float(distances.min()) if distances.notna().any() else None


def distance_fields(prefix: str, distance_m: float | None, thresholds: list[float]) -> dict[str, Any]:
    fields: dict[str, Any] = {f"{prefix}_nearest_distance_m": distance_m}
    for threshold in thresholds:
        fields[f"{prefix}_hit_within_{int(threshold)}m"] = (
            None if distance_m is None else int(distance_m <= threshold)
        )
    return fields


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if path.exists() and path.stat().st_size > 0:
        old_df = pd.read_csv(path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["fire_id"], keep="first")
    else:
        combined = new_df
    combined.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prospectively validate newly observed Alberta fires.")
    parser.add_argument("--active-fires-csv", required=True)
    parser.add_argument("--runs-root", default="results/runs")
    parser.add_argument("--state-json", default=DEFAULT_STATE_PATH)
    parser.add_argument("--prediction-registry-csv", default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--validation-log-csv", default=DEFAULT_LOG_PATH)
    parser.add_argument("--agency", default="AB")
    parser.add_argument("--fire-id-column", default=None)
    parser.add_argument("--start-date-column", default="Start_Date")
    parser.add_argument("--start-date-unit", default="ms")
    parser.add_argument("--distance-thresholds-m", default=DEFAULT_THRESHOLDS_M)
    parser.add_argument(
        "--initialize-state",
        action="store_true",
        help="Record current fires as the baseline without validating them.",
    )
    parser.add_argument(
        "--poll-time-utc",
        default=None,
        help="Optional ISO-8601 poll time for reproducible testing. Defaults to current UTC time.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not write state or validation rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.distance_thresholds_m)
    poll_time = pd.to_datetime(args.poll_time_utc, utc=True) if args.poll_time_utc else utc_now()

    active_path = Path(args.active_fires_csv)
    state_path = Path(args.state_json)
    registry_path = Path(args.prediction_registry_csv)
    log_path = Path(args.validation_log_csv)
    runs_root = Path(args.runs_root)

    active = read_active_fires(
        active_path,
        agency=args.agency,
        fire_id_column=args.fire_id_column,
        start_date_column=args.start_date_column,
        start_date_unit=args.start_date_unit,
    )
    runs = discover_prediction_runs(runs_root)
    write_prediction_registry(runs, registry_path)

    state = load_state(state_path)
    seen_fires: dict[str, Any] = state["seen_fires"]

    if not state_path.exists() and not args.initialize_state:
        raise RuntimeError(
            f"State file does not exist: {state_path}. "
            "Run once with --initialize-state so existing fires are not treated as new ignitions."
        )

    if args.initialize_state:
        for row in active.itertuples(index=False):
            seen_fires[str(row._fire_id)] = {
                "first_seen_utc": poll_time.isoformat(),
                "fire_start_utc": None if pd.isna(row._fire_start_utc) else row._fire_start_utc.isoformat(),
                "baseline": True,
            }
        state["last_poll_time_utc"] = poll_time.isoformat()
        if not args.dry_run:
            save_state(state_path, state)
        print("Prospective validation state initialized")
        print(f"Baseline Alberta fires recorded: {len(active)}")
        print(f"Prediction runs registered: {len(runs)}")
        print(f"State: {state_path}")
        print(f"Prediction registry: {registry_path}")
        return

    new_rows = active[~active["_fire_id"].isin(seen_fires)].copy()
    print(f"Poll time UTC: {poll_time.isoformat()}")
    print(f"Current Alberta fires: {len(active)}")
    print(f"Previously seen fires: {len(seen_fires)}")
    print(f"New fires detected: {len(new_rows)}")
    print(f"Eligible prediction runs discovered: {len(runs)}")

    validation_rows: list[dict[str, Any]] = []
    artifact_cache: dict[str, tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]] = {}

    for row in new_rows.itertuples(index=False):
        fire_id = str(row._fire_id)
        fire_start = row._fire_start_utc
        base_record: dict[str, Any] = {
            "fire_id": fire_id,
            "fire_start_utc": None if pd.isna(fire_start) else fire_start.isoformat(),
            "fire_first_seen_utc": poll_time.isoformat(),
            "latitude": float(row._latitude),
            "longitude": float(row._longitude),
            "agency": args.agency,
            "source_csv": str(active_path),
        }

        if pd.isna(fire_start):
            base_record.update({"status": "invalid_fire_start_time", "prediction_run_id": None})
            validation_rows.append(base_record)
        else:
            selected_run = select_latest_prior_run(runs, fire_start)
            if selected_run is None:
                base_record.update({"status": "no_prior_prediction_run", "prediction_run_id": None})
                validation_rows.append(base_record)
            else:
                if selected_run.run_id not in artifact_cache:
                    model_b = load_model_b_candidates(selected_run.model_b_candidates)
                    model_a_all, model_a_positive = load_model_a_candidates(selected_run.model_a_geojson)
                    artifact_cache[selected_run.run_id] = (model_b, model_a_all, model_a_positive)

                model_b, model_a_all, model_a_positive = artifact_cache[selected_run.run_id]
                point = Point(float(row._longitude), float(row._latitude))
                model_b_distance = nearest_distance_m(point, model_b)
                model_a_all_distance = nearest_distance_m(point, model_a_all)
                model_a_positive_distance = nearest_distance_m(point, model_a_positive)

                base_record.update(
                    {
                        "status": "validated",
                        "prediction_run_id": selected_run.run_id,
                        "prediction_created_utc": selected_run.created_at.isoformat(),
                        "lead_time_hours": float((fire_start - selected_run.created_at).total_seconds() / 3600.0),
                    }
                )
                base_record.update(distance_fields("model_b_candidate", model_b_distance, thresholds))
                base_record.update(distance_fields("model_a_candidate", model_a_all_distance, thresholds))
                base_record.update(distance_fields("model_a_positive", model_a_positive_distance, thresholds))
                validation_rows.append(base_record)

        seen_fires[fire_id] = {
            "first_seen_utc": poll_time.isoformat(),
            "fire_start_utc": None if pd.isna(fire_start) else fire_start.isoformat(),
            "baseline": False,
        }

    state["last_poll_time_utc"] = poll_time.isoformat()
    if not args.dry_run:
        append_rows(log_path, validation_rows)
        save_state(state_path, state)

    validated_count = sum(row.get("status") == "validated" for row in validation_rows)
    print(f"New fires validated against a prior prediction: {validated_count}")
    print(f"Validation rows written: {0 if args.dry_run else len(validation_rows)}")
    print(f"Validation log: {log_path}")
    print(f"State: {state_path}")
    print(f"Prediction registry: {registry_path}")


if __name__ == "__main__":
    main()
