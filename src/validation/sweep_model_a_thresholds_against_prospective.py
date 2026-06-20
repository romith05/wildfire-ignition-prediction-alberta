"""Sweep Model A thresholds against prospective validation fires.

This read-only diagnostic reuses completed prediction artifacts and the
prospective validation log. It answers a practical calibration question:

    Can a stricter Model A threshold reduce candidate cells while preserving
    prospective fire hits within the configured distance thresholds?

Example:
    python -m src.validation.sweep_model_a_thresholds_against_prospective \
      --thresholds 0.50,0.60,0.70,0.80,0.90
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src.validation.poll_active_fires_and_validate import (
    DEFAULT_LOG_PATH,
    DEFAULT_THRESHOLDS_M,
    distance_fields,
    nearest_distance_m,
    parse_thresholds,
)

DEFAULT_RUNS_ROOT = Path("results/runs")
DEFAULT_MODEL_A_THRESHOLDS = "0.50,0.60,0.70,0.80,0.90"


def parse_float_list(value: str) -> list[float]:
    values = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def load_validated_fires(validation_log_csv: Path) -> pd.DataFrame:
    if not validation_log_csv.exists():
        raise FileNotFoundError(f"Validation log not found: {validation_log_csv}")

    df = pd.read_csv(validation_log_csv)
    if df.empty:
        return df

    required = ["status", "prediction_run_id", "latitude", "longitude"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Validation log missing required columns: {missing}")

    validated = df[df["status"].astype(str).str.lower().eq("validated")].copy()
    validated["latitude"] = pd.to_numeric(validated["latitude"], errors="coerce")
    validated["longitude"] = pd.to_numeric(validated["longitude"], errors="coerce")
    validated = validated.dropna(subset=["latitude", "longitude", "prediction_run_id"])
    return validated.reset_index(drop=True)


def load_model_a_geojson(runs_root: Path, run_id: str) -> gpd.GeoDataFrame:
    path = runs_root / run_id / "model_a_candidate_cells.geojson"
    if not path.exists():
        raise FileNotFoundError(f"Model A GeoJSON not found for run {run_id}: {path}")

    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Model A GeoJSON is empty: {path}")
    if gdf.crs is None:
        raise ValueError(f"Model A GeoJSON has no CRS: {path}")
    if "cell_max_prob" not in gdf.columns:
        raise ValueError(f"Model A GeoJSON has no cell_max_prob column: {path}")
    if "cell_positive_pixels" not in gdf.columns:
        raise ValueError(f"Model A GeoJSON has no cell_positive_pixels column: {path}")

    gdf["cell_max_prob"] = pd.to_numeric(gdf["cell_max_prob"], errors="coerce")
    gdf["cell_positive_pixels"] = pd.to_numeric(
        gdf["cell_positive_pixels"], errors="coerce"
    ).fillna(0)
    return gdf


def evaluate_threshold(
    fires: pd.DataFrame,
    artifacts: dict[str, gpd.GeoDataFrame],
    threshold: float,
    min_positive_pixels: int,
    distance_thresholds_m: list[float],
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    rows: list[dict[str, object]] = []
    retained_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}

    for fire in fires.itertuples(index=False):
        run_id = str(fire.prediction_run_id)
        candidates = artifacts[run_id]
        total_counts[run_id] = len(candidates)

        retained = candidates[
            candidates["cell_max_prob"].ge(threshold)
            & candidates["cell_positive_pixels"].ge(min_positive_pixels)
        ].copy()
        retained_counts[run_id] = len(retained)

        point = Point(float(fire.longitude), float(fire.latitude))
        distance_m = nearest_distance_m(point, retained)

        row = {
            "fire_id": getattr(fire, "fire_id", None),
            "prediction_run_id": run_id,
            "threshold": threshold,
            "min_positive_pixels": min_positive_pixels,
            "retained_candidates_for_run": len(retained),
            "total_candidates_for_run": len(candidates),
        }
        row.update(distance_fields("model_a_swept_positive", distance_m, distance_thresholds_m))
        rows.append(row)

    detail = pd.DataFrame(rows)
    summary: dict[str, float | int | str] = {
        "threshold": threshold,
        "min_positive_pixels": min_positive_pixels,
        "validated_fires": int(len(detail)),
        "unique_prediction_runs": int(len(retained_counts)),
        "median_retained_candidates_per_used_run": float(
            pd.Series(retained_counts, dtype="float64").median()
        ),
        "median_total_candidates_per_used_run": float(
            pd.Series(total_counts, dtype="float64").median()
        ),
    }

    distances_km = (
        pd.to_numeric(detail["model_a_swept_positive_nearest_distance_m"], errors="coerce")
        / 1000.0
    )
    if distances_km.notna().any():
        summary["median_nearest_distance_km"] = float(distances_km.median())
    else:
        summary["median_nearest_distance_km"] = "n/a"

    for distance_m in distance_thresholds_m:
        col = f"model_a_swept_positive_hit_within_{int(distance_m)}m"
        hits = pd.to_numeric(detail[col], errors="coerce").fillna(0).astype(int)
        hit_count = int(hits.sum())
        total = int(len(hits))
        distance_km = int(distance_m / 1000)
        summary[f"hits_within_{distance_km}km"] = hit_count
        summary[f"hit_rate_within_{distance_km}km"] = (
            None if total == 0 else float(hit_count / total)
        )

    return detail, summary


def print_summary(summary_df: pd.DataFrame, distance_thresholds_m: list[float]) -> None:
    if summary_df.empty:
        print("No threshold results to summarize.")
        return

    display_columns = [
        "threshold",
        "min_positive_pixels",
        "validated_fires",
        "median_retained_candidates_per_used_run",
        "median_total_candidates_per_used_run",
        "median_nearest_distance_km",
    ]
    for distance_m in distance_thresholds_m:
        distance_km = int(distance_m / 1000)
        display_columns.append(f"hits_within_{distance_km}km")
        display_columns.append(f"hit_rate_within_{distance_km}km")

    available = [column for column in display_columns if column in summary_df.columns]
    printable = summary_df[available].copy()
    for column in printable.columns:
        if column.startswith("hit_rate_within_"):
            printable[column] = printable[column].map(
                lambda value: "n/a" if pd.isna(value) else f"{100.0 * float(value):.1f}%"
            )
        elif column in {
            "median_retained_candidates_per_used_run",
            "median_total_candidates_per_used_run",
            "median_nearest_distance_km",
        }:
            printable[column] = printable[column].map(
                lambda value: value if isinstance(value, str) else f"{float(value):.2f}"
            )

    print(printable.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep Model A thresholds against prospective validation fires."
    )
    parser.add_argument("--validation-log-csv", default=DEFAULT_LOG_PATH)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--thresholds", default=DEFAULT_MODEL_A_THRESHOLDS)
    parser.add_argument("--min-positive-pixels", type=int, default=1)
    parser.add_argument("--distance-thresholds-m", default=DEFAULT_THRESHOLDS_M)
    parser.add_argument(
        "--output-summary-csv",
        default=None,
        help="Optional path to write threshold-level summary rows.",
    )
    parser.add_argument(
        "--output-detail-csv",
        default=None,
        help="Optional path to write fire-level threshold sweep rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation_log_csv = Path(args.validation_log_csv)
    runs_root = Path(args.runs_root)
    thresholds = parse_float_list(args.thresholds)
    distance_thresholds_m = parse_thresholds(args.distance_thresholds_m)

    fires = load_validated_fires(validation_log_csv)
    print("Model A prospective threshold sweep")
    print("=" * 40)
    print(f"Validation log: {validation_log_csv}")
    print(f"Runs root: {runs_root}")
    print(f"Validated fires: {len(fires)}")

    if fires.empty:
        print("No validated fires available for threshold sweeping.")
        return

    artifacts = {
        str(run_id): load_model_a_geojson(runs_root, str(run_id))
        for run_id in sorted(fires["prediction_run_id"].astype(str).unique())
    }

    detail_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float | int | str]] = []
    for threshold in thresholds:
        detail, summary = evaluate_threshold(
            fires=fires,
            artifacts=artifacts,
            threshold=threshold,
            min_positive_pixels=args.min_positive_pixels,
            distance_thresholds_m=distance_thresholds_m,
        )
        detail_frames.append(detail)
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()

    print_summary(summary_df, distance_thresholds_m)

    if args.output_summary_csv:
        output = Path(args.output_summary_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output, index=False)
        print(f"Summary CSV: {output}")

    if args.output_detail_csv:
        output = Path(args.output_detail_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        detail_df.to_csv(output, index=False)
        print(f"Detail CSV: {output}")


if __name__ == "__main__":
    main()
