
"""Compare top-ranked Model B candidates with nearest-to-fire candidates.

This read-only diagnostic samples Model B input feature values from saved NPZ
patches for two candidate groups:

1. top-ranked candidate cells by Model B probability
2. nearest candidate cell to each prospectively validated fire

It helps identify why far-away cells are ranked higher than cells near future
fires.

Example:
    python -m src.validation.compare_ranked_candidate_features \
      --top-n 25 \
      --output-detail-csv results/validation/ranked_candidate_feature_comparison_detail.csv \
      --output-summary-csv results/validation/ranked_candidate_feature_comparison_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from src.validation.poll_active_fires_and_validate import (
    DEFAULT_LOG_PATH,
    load_model_b_candidates,
)


DEFAULT_RUNS_ROOT = Path("results/runs")
DEFAULT_TOP_N = 25


def load_validated_fires(validation_log_csv: Path) -> pd.DataFrame:
    if not validation_log_csv.exists():
        raise FileNotFoundError(f"Validation log not found: {validation_log_csv}")

    df = pd.read_csv(validation_log_csv)
    if df.empty:
        return df

    required = ["fire_id", "status", "prediction_run_id", "latitude", "longitude"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Validation log missing required columns: {missing}")

    validated = df[df["status"].astype(str).str.lower().eq("validated")].copy()
    validated["latitude"] = pd.to_numeric(validated["latitude"], errors="coerce")
    validated["longitude"] = pd.to_numeric(validated["longitude"], errors="coerce")
    validated = validated.dropna(subset=["latitude", "longitude", "prediction_run_id"])
    return validated.reset_index(drop=True)


def distances_to_fire(candidates: gpd.GeoDataFrame, fire_row: Any) -> pd.Series:
    point_wgs84 = Point(float(fire_row.longitude), float(fire_row.latitude))
    point = gpd.GeoSeries([point_wgs84], crs="EPSG:4326").to_crs(candidates.crs).iloc[0]
    return candidates.geometry.distance(point)


def load_npz_arrays(npz_path: str | Path, cache: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    path = str(npz_path)
    if path not in cache:
        with np.load(path) as arrs:
            cache[path] = {
                key: arrs[key].copy()
                for key in arrs.files
                if key != "class" and np.asarray(arrs[key]).ndim >= 2
            }
    return cache[path]


def sample_features(candidate_row: Any, npz_cache: dict[str, dict[str, np.ndarray]]) -> dict[str, float | None]:
    npz_path = str(candidate_row.npz_path)
    row = int(candidate_row.row)
    col = int(candidate_row.col)

    features: dict[str, float | None] = {}
    arrays = load_npz_arrays(npz_path, npz_cache)

    for key, array in arrays.items():
        try:
            value = float(array[row, col])
        except Exception:
            value = np.nan
        features[key] = None if not np.isfinite(value) else value

    return features


def candidate_records(
    *,
    candidates: gpd.GeoDataFrame,
    fire_row: Any,
    top_n: int,
    npz_cache: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    candidates = candidates.copy()
    candidates["probability"] = pd.to_numeric(candidates["probability"], errors="coerce")
    candidates = candidates.dropna(subset=["probability"]).reset_index(drop=True)

    if candidates.empty:
        return []

    distances = distances_to_fire(candidates, fire_row)
    candidates["distance_to_fire_m"] = distances.astype(float)

    ranked = candidates.sort_values("probability", ascending=False).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)

    nearest_idx = int(candidates["distance_to_fire_m"].idxmin())
    nearest_candidate_id = str(candidates.loc[nearest_idx, "candidate_id"])

    selected = []
    top = ranked.head(top_n).copy()
    top["candidate_role"] = "top_ranked"
    selected.append(top)

    nearest = ranked[ranked["candidate_id"].astype(str).eq(nearest_candidate_id)].copy()
    nearest["candidate_role"] = "nearest_to_fire"
    selected.append(nearest)

    out_rows: list[dict[str, Any]] = []
    for row in pd.concat(selected, ignore_index=True).itertuples(index=False):
        record = {
            "fire_id": str(fire_row.fire_id),
            "prediction_run_id": str(fire_row.prediction_run_id),
            "candidate_role": str(row.candidate_role),
            "candidate_id": str(row.candidate_id),
            "rank": int(row.rank),
            "probability": float(row.probability),
            "distance_to_fire_m": float(row.distance_to_fire_m),
            "row": int(row.row),
            "col": int(row.col),
            "npz_path": str(row.npz_path),
        }
        if hasattr(fire_row, "lead_time_hours"):
            record["lead_time_hours"] = fire_row.lead_time_hours

        features = sample_features(row, npz_cache)
        for key, value in features.items():
            record[f"feature__{key}"] = value

        out_rows.append(record)

    return out_rows


def build_detail(validation_log_csv: Path, runs_root: Path, top_n: int) -> pd.DataFrame:
    fires = load_validated_fires(validation_log_csv)
    if fires.empty:
        return pd.DataFrame()

    run_cache: dict[str, gpd.GeoDataFrame] = {}
    npz_cache: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []

    for fire in fires.itertuples(index=False):
        run_id = str(fire.prediction_run_id)
        if run_id not in run_cache:
            candidates_path = runs_root / run_id / "model_b_candidates.csv"
            run_cache[run_id] = load_model_b_candidates(candidates_path)

        rows.extend(
            candidate_records(
                candidates=run_cache[run_id],
                fire_row=fire,
                top_n=top_n,
                npz_cache=npz_cache,
            )
        )

    return pd.DataFrame(rows)


def build_summary(detail: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [column for column in detail.columns if column.startswith("feature__")]
    rows: list[dict[str, Any]] = []

    for col in feature_cols:
        values = detail[["candidate_role", col]].copy()
        values[col] = pd.to_numeric(values[col], errors="coerce")

        nearest = values[values["candidate_role"].eq("nearest_to_fire")][col].dropna()
        top = values[values["candidate_role"].eq("top_ranked")][col].dropna()

        if nearest.empty or top.empty:
            continue

        nearest_median = float(nearest.median())
        top_median = float(top.median())

        rows.append(
            {
                "feature": col.replace("feature__", ""),
                "nearest_to_fire_median": nearest_median,
                "top_ranked_median": top_median,
                "top_minus_nearest": top_median - nearest_median,
                "nearest_to_fire_mean": float(nearest.mean()),
                "top_ranked_mean": float(top.mean()),
                "nearest_to_fire_count": int(len(nearest)),
                "top_ranked_count": int(len(top)),
            }
        )

    return pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare top-ranked Model B features against nearest-to-fire features."
    )
    parser.add_argument("--validation-log-csv", default=DEFAULT_LOG_PATH)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--output-detail-csv", default=None)
    parser.add_argument("--output-summary-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive.")

    detail = build_detail(
        validation_log_csv=Path(args.validation_log_csv),
        runs_root=Path(args.runs_root),
        top_n=args.top_n,
    )
    summary = build_summary(detail) if not detail.empty else pd.DataFrame()

    print("Ranked candidate feature comparison")
    print("=" * 40)
    print(f"Detail rows: {len(detail)}")
    print(f"Summary features: {len(summary)}")

    if not summary.empty:
        print("")
        print(summary.to_string(index=False))

    if args.output_detail_csv:
        output = Path(args.output_detail_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        detail.to_csv(output, index=False)
        print(f"Detail CSV: {output}")

    if args.output_summary_csv:
        output = Path(args.output_summary_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output, index=False)
        print(f"Summary CSV: {output}")


if __name__ == "__main__":
    main()
