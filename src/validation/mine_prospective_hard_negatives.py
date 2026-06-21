"""Mine prospective hard-negative candidates for Model B retraining.

This read-only diagnostic uses completed prospective validation artifacts to build a
candidate table for the next Model B ranking-improvement dataset.

It exports two sample groups for each validated fire:

1. near_fire_candidate
   Model B candidates closest to the future fire. These are useful positive or
   soft-positive examples depending on distance.

2. hard_negative_candidate
   High-probability Model B candidates far from the future fire. These are not
   guaranteed true negatives, but they are useful hard-negative candidates for
   reducing overconfident ranking of wrong regions.

Example:
    python -m src.validation.mine_prospective_hard_negatives \
      --output-csv results/validation/prospective_hard_negative_candidates.csv \
      --summary-csv results/validation/prospective_hard_negative_summary.csv
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
DEFAULT_OUTPUT_CSV = Path("results/validation/prospective_hard_negative_candidates.csv")
DEFAULT_SUMMARY_CSV = Path("results/validation/prospective_hard_negative_summary.csv")

LANDCOVER_LABELS = {
    0: "no_change",
    20: "water",
    31: "snow_ice",
    32: "rock_rubble",
    33: "exposed_barren_land",
    40: "bryoids",
    50: "shrubland",
    80: "wetland",
    81: "wetland_treed",
    100: "herbs",
    210: "coniferous",
    220: "broadleaf",
    230: "mixedwood",
}
NON_BURNABLE_LANDCOVER = {20, 31, 32, 33}


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
    arrays = load_npz_arrays(candidate_row.npz_path, npz_cache)
    row = int(candidate_row.row)
    col = int(candidate_row.col)

    features: dict[str, float | None] = {}
    for key, array in arrays.items():
        try:
            value = float(array[row, col])
        except Exception:  # noqa: BLE001 - diagnostic should continue across malformed arrays.
            value = np.nan
        features[f"feature__{key}"] = None if not np.isfinite(value) else value
    return features


def landcover_code(features: dict[str, float | None]) -> int | None:
    value = features.get("feature__landcover_1km")
    if value is None or not np.isfinite(float(value)):
        return None
    return int(round(float(value)))


def reason_flags(
    *,
    features: dict[str, float | None],
    distance_to_fire_m: float,
    probability: float,
    high_humidity_threshold: float,
    low_wind_threshold: float,
    remote_road_distance_m: float,
) -> list[str]:
    flags: list[str] = []
    code = landcover_code(features)
    if code in NON_BURNABLE_LANDCOVER:
        flags.append("non_burnable_landcover")

    humidity = features.get("feature__relative_humidity")
    wind = features.get("feature__wind_speed")
    road = features.get("feature__distance_to_road_1km")

    if humidity is not None and float(humidity) >= high_humidity_threshold:
        flags.append("high_humidity")
    if wind is not None and float(wind) <= low_wind_threshold:
        flags.append("low_wind")
    if road is not None and float(road) >= remote_road_distance_m:
        flags.append("remote_from_road")
    if distance_to_fire_m >= 50_000 and probability >= 0.90:
        flags.append("very_far_high_probability")

    return flags


def suggested_target_and_weight(sample_role: str, distance_to_fire_m: float, flags: list[str]) -> tuple[float, float]:
    if sample_role == "hard_negative_candidate":
        weight = 1.0
        if "non_burnable_landcover" in flags:
            weight += 1.0
        if "very_far_high_probability" in flags:
            weight += 0.5
        if "high_humidity" in flags and "low_wind" in flags:
            weight += 0.5
        return 0.0, weight

    if distance_to_fire_m <= 5_000:
        return 1.0, 1.5
    if distance_to_fire_m <= 10_000:
        return 0.75, 1.25
    if distance_to_fire_m <= 25_000:
        return 0.50, 1.0
    return 0.25, 0.5


def row_to_record(
    *,
    row: Any,
    fire_row: Any,
    sample_role: str,
    npz_cache: dict[str, dict[str, np.ndarray]],
    high_humidity_threshold: float,
    low_wind_threshold: float,
    remote_road_distance_m: float,
) -> dict[str, Any]:
    features = sample_features(row, npz_cache)
    code = landcover_code(features)
    flags = reason_flags(
        features=features,
        distance_to_fire_m=float(row.distance_to_fire_m),
        probability=float(row.probability),
        high_humidity_threshold=high_humidity_threshold,
        low_wind_threshold=low_wind_threshold,
        remote_road_distance_m=remote_road_distance_m,
    )
    target, weight = suggested_target_and_weight(sample_role, float(row.distance_to_fire_m), flags)

    record: dict[str, Any] = {
        "fire_id": str(fire_row.fire_id),
        "prediction_run_id": str(fire_row.prediction_run_id),
        "fire_latitude": float(fire_row.latitude),
        "fire_longitude": float(fire_row.longitude),
        "sample_role": sample_role,
        "candidate_id": str(row.candidate_id),
        "patch_id": str(row.patch_id),
        "npz_path": str(row.npz_path),
        "row": int(row.row),
        "col": int(row.col),
        "probability": float(row.probability),
        "rank": int(row.rank),
        "distance_to_fire_m": float(row.distance_to_fire_m),
        "landcover_code": code,
        "landcover_label": LANDCOVER_LABELS.get(code, "unknown"),
        "reason_flags": ";".join(flags),
        "suggested_target": target,
        "suggested_weight": weight,
    }
    if hasattr(fire_row, "fire_start_utc"):
        record["fire_start_utc"] = getattr(fire_row, "fire_start_utc")
    if hasattr(fire_row, "lead_time_hours"):
        record["lead_time_hours"] = getattr(fire_row, "lead_time_hours")

    record.update(features)
    return record


def mine_for_fire(
    *,
    candidates: gpd.GeoDataFrame,
    fire_row: Any,
    npz_cache: dict[str, dict[str, np.ndarray]],
    near_distance_km: float,
    far_distance_km: float,
    top_n: int,
    max_near_per_fire: int,
    max_hard_negatives_per_fire: int,
    high_humidity_threshold: float,
    low_wind_threshold: float,
    remote_road_distance_m: float,
) -> list[dict[str, Any]]:
    candidates = candidates.copy()
    candidates["probability"] = pd.to_numeric(candidates["probability"], errors="coerce")
    candidates = candidates.dropna(subset=["probability"]).reset_index(drop=True)
    if candidates.empty:
        return []

    candidates["distance_to_fire_m"] = distances_to_fire(candidates, fire_row).astype(float)
    ranked = candidates.sort_values("probability", ascending=False).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)

    near_distance_m = near_distance_km * 1000.0
    far_distance_m = far_distance_km * 1000.0

    near = ranked[ranked["distance_to_fire_m"].le(near_distance_m)].sort_values(
        ["distance_to_fire_m", "probability"], ascending=[True, False]
    )
    if near.empty:
        near = ranked.sort_values(["distance_to_fire_m", "probability"], ascending=[True, False]).head(1)
    else:
        near = near.head(max_near_per_fire)

    top_far = ranked[
        ranked["distance_to_fire_m"].ge(far_distance_m) & ranked["rank"].le(top_n)
    ].copy()
    if len(top_far) < max_hard_negatives_per_fire:
        extra_far = ranked[ranked["distance_to_fire_m"].ge(far_distance_m)].head(max_hard_negatives_per_fire)
        top_far = pd.concat([top_far, extra_far], ignore_index=True).drop_duplicates("candidate_id")
    hard = top_far.sort_values("probability", ascending=False).head(max_hard_negatives_per_fire)

    records: list[dict[str, Any]] = []
    for row in near.itertuples(index=False):
        records.append(
            row_to_record(
                row=row,
                fire_row=fire_row,
                sample_role="near_fire_candidate",
                npz_cache=npz_cache,
                high_humidity_threshold=high_humidity_threshold,
                low_wind_threshold=low_wind_threshold,
                remote_road_distance_m=remote_road_distance_m,
            )
        )
    for row in hard.itertuples(index=False):
        records.append(
            row_to_record(
                row=row,
                fire_row=fire_row,
                sample_role="hard_negative_candidate",
                npz_cache=npz_cache,
                high_humidity_threshold=high_humidity_threshold,
                low_wind_threshold=low_wind_threshold,
                remote_road_distance_m=remote_road_distance_m,
            )
        )
    return records


def mine_samples(args: argparse.Namespace) -> pd.DataFrame:
    fires = load_validated_fires(Path(args.validation_log_csv))
    if fires.empty:
        return pd.DataFrame()

    run_cache: dict[str, gpd.GeoDataFrame] = {}
    npz_cache: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []

    for fire in fires.itertuples(index=False):
        run_id = str(fire.prediction_run_id)
        if run_id not in run_cache:
            candidates_path = Path(args.runs_root) / run_id / "model_b_candidates.csv"
            run_cache[run_id] = load_model_b_candidates(candidates_path)

        rows.extend(
            mine_for_fire(
                candidates=run_cache[run_id],
                fire_row=fire,
                npz_cache=npz_cache,
                near_distance_km=args.near_distance_km,
                far_distance_km=args.far_distance_km,
                top_n=args.top_n,
                max_near_per_fire=args.max_near_per_fire,
                max_hard_negatives_per_fire=args.max_hard_negatives_per_fire,
                high_humidity_threshold=args.high_humidity_threshold,
                low_wind_threshold=args.low_wind_threshold,
                remote_road_distance_m=args.remote_road_distance_m,
            )
        )

    return pd.DataFrame(rows)


def build_summary(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return samples

    rows: list[dict[str, Any]] = []
    for role, group in samples.groupby("sample_role"):
        row: dict[str, Any] = {
            "sample_role": role,
            "sample_count": int(len(group)),
            "median_probability": float(pd.to_numeric(group["probability"], errors="coerce").median()),
            "median_distance_to_fire_km": float(
                pd.to_numeric(group["distance_to_fire_m"], errors="coerce").median() / 1000.0
            ),
            "median_rank": float(pd.to_numeric(group["rank"], errors="coerce").median()),
            "mean_suggested_weight": float(pd.to_numeric(group["suggested_weight"], errors="coerce").mean()),
        }
        for feature in [
            "feature__relative_humidity",
            "feature__wind_speed",
            "feature__temperature",
            "feature__distance_to_road_1km",
            "feature__DEM_1km",
        ]:
            if feature in group.columns:
                row[f"median_{feature.replace('feature__', '')}"] = float(
                    pd.to_numeric(group[feature], errors="coerce").median()
                )
        rows.append(row)

    landcover = (
        samples.groupby(["sample_role", "landcover_code", "landcover_label"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    landcover["percent_within_role"] = landcover.groupby("sample_role")["count"].transform(
        lambda values: 100.0 * values / values.sum()
    )

    summary = pd.DataFrame(rows)
    landcover_summary = landcover.sort_values(["sample_role", "count"], ascending=[True, False])
    return pd.concat(
        [
            summary.assign(summary_type="numeric"),
            landcover_summary.assign(summary_type="landcover"),
        ],
        ignore_index=True,
        sort=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine prospective hard-negative candidates for Model B.")
    parser.add_argument("--validation-log-csv", default=DEFAULT_LOG_PATH)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--near-distance-km", type=float, default=25.0)
    parser.add_argument("--far-distance-km", type=float, default=50.0)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--max-near-per-fire", type=int, default=10)
    parser.add_argument("--max-hard-negatives-per-fire", type=int, default=25)
    parser.add_argument("--high-humidity-threshold", type=float, default=75.0)
    parser.add_argument("--low-wind-threshold", type=float, default=10.0)
    parser.add_argument("--remote-road-distance-m", type=float, default=5000.0)
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive.")
    if args.max_near_per_fire <= 0 or args.max_hard_negatives_per_fire <= 0:
        raise ValueError("Sample limits must be positive.")

    samples = mine_samples(args)
    summary = build_summary(samples)

    print("Prospective hard-negative mining")
    print("=" * 40)
    print(f"Samples: {len(samples)}")
    if not samples.empty:
        print(samples["sample_role"].value_counts().to_string())

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output, index=False)
    print(f"Output CSV: {output}")

    summary_output = Path(args.summary_csv)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output, index=False)
    print(f"Summary CSV: {summary_output}")

    if not summary.empty:
        print("")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
