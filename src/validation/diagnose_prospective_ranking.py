"""Diagnose ranking quality around prospective validation fires.

This read-only diagnostic helps decide whether the next model improvement should
focus on recall or ranking. For each prospectively validated fire, it reports:

- nearest Model B / Model A candidate distance
- rank of that nearest candidate by model probability
- nearest distance among top-k ranked cells

Example:
    python -m src.validation.diagnose_prospective_ranking \
      --output-csv results/validation/prospective_ranking_diagnostic.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from src.validation.poll_active_fires_and_validate import (
    DEFAULT_LOG_PATH,
    load_model_b_candidates,
)

DEFAULT_RUNS_ROOT = Path("results/runs")
DEFAULT_TOP_K = "1,5,10,25,50,100"


def parse_int_list(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("Expected one or more positive integer top-k values.")
    return values


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


def load_model_a_candidates(path: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Model A GeoJSON not found: {path}")

    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Model A GeoJSON is empty: {path}")
    if gdf.crs is None:
        raise ValueError(f"Model A GeoJSON has no CRS: {path}")
    if "cell_max_prob" not in gdf.columns:
        raise ValueError(f"Model A GeoJSON has no cell_max_prob column: {path}")

    gdf = gdf.copy()
    gdf["cell_max_prob"] = pd.to_numeric(gdf["cell_max_prob"], errors="coerce")

    if "final_positive" in gdf.columns:
        positive_mask = (
            pd.to_numeric(gdf["final_positive"], errors="coerce").fillna(0).astype(int).eq(1)
        )
    else:
        positive_mask = pd.Series(False, index=gdf.index)

    return gdf, gdf[positive_mask].copy()


def rank_layer(layer: gpd.GeoDataFrame, score_column: str) -> gpd.GeoDataFrame:
    if score_column not in layer.columns:
        raise ValueError(f"Layer has no score column: {score_column}")

    ranked = layer.copy()
    ranked[score_column] = pd.to_numeric(ranked[score_column], errors="coerce")
    ranked = ranked.dropna(subset=[score_column]).sort_values(score_column, ascending=False).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def point_for_fire(row: object) -> Point:
    return Point(float(getattr(row, "longitude")), float(getattr(row, "latitude")))


def distances_to_point(layer: gpd.GeoDataFrame, point_wgs84: Point) -> pd.Series:
    if layer.empty:
        return pd.Series(dtype="float64")
    point = gpd.GeoSeries([point_wgs84], crs="EPSG:4326").to_crs(layer.crs).iloc[0]
    return layer.geometry.distance(point)


def summarize_layer(
    *,
    layer: gpd.GeoDataFrame,
    point_wgs84: Point,
    prefix: str,
    score_column: str,
    top_k_values: Iterable[int],
) -> dict[str, object]:
    fields: dict[str, object] = {
        f"{prefix}_candidate_count": int(len(layer)),
        f"{prefix}_score_column": score_column,
    }

    if layer.empty:
        fields.update(
            {
                f"{prefix}_nearest_distance_m": None,
                f"{prefix}_nearest_rank": None,
                f"{prefix}_nearest_score": None,
                f"{prefix}_nearest_id": None,
            }
        )
        for top_k in top_k_values:
            fields[f"{prefix}_top{top_k}_nearest_distance_m"] = None
        return fields

    ranked = rank_layer(layer, score_column)
    distances = distances_to_point(ranked, point_wgs84)
    nearest_idx = int(distances.idxmin())
    nearest = ranked.loc[nearest_idx]

    fields.update(
        {
            f"{prefix}_nearest_distance_m": float(distances.loc[nearest_idx]),
            f"{prefix}_nearest_rank": int(nearest["rank"]),
            f"{prefix}_nearest_score": float(nearest[score_column]),
            f"{prefix}_nearest_id": nearest.get("candidate_id", nearest.get("patch_id", "")),
        }
    )

    for top_k in top_k_values:
        top = ranked.head(top_k)
        top_distances = distances_to_point(top, point_wgs84)
        fields[f"{prefix}_top{top_k}_nearest_distance_m"] = (
            None if top_distances.empty else float(top_distances.min())
        )

    return fields


def load_run_artifacts(runs_root: Path, run_id: str) -> dict[str, gpd.GeoDataFrame]:
    run_dir = runs_root / run_id
    model_b_path = run_dir / "model_b_candidates.csv"
    model_a_path = run_dir / "model_a_candidate_cells.geojson"

    model_b = load_model_b_candidates(model_b_path)
    model_b["probability"] = pd.to_numeric(model_b["probability"], errors="coerce")

    model_a_all, model_a_positive = load_model_a_candidates(model_a_path)
    return {
        "model_b": model_b,
        "model_a_all": model_a_all,
        "model_a_positive": model_a_positive,
    }


def diagnose(
    fires: pd.DataFrame,
    runs_root: Path,
    top_k_values: list[int],
) -> pd.DataFrame:
    artifact_cache: dict[str, dict[str, gpd.GeoDataFrame]] = {}
    rows: list[dict[str, object]] = []

    for fire in fires.itertuples(index=False):
        run_id = str(fire.prediction_run_id)
        if run_id not in artifact_cache:
            artifact_cache[run_id] = load_run_artifacts(runs_root, run_id)

        artifacts = artifact_cache[run_id]
        point = point_for_fire(fire)

        row: dict[str, object] = {
            "fire_id": str(fire.fire_id),
            "fire_start_utc": getattr(fire, "fire_start_utc", None),
            "prediction_run_id": run_id,
            "latitude": float(fire.latitude),
            "longitude": float(fire.longitude),
        }
        if hasattr(fire, "lead_time_hours"):
            row["lead_time_hours"] = getattr(fire, "lead_time_hours")

        row.update(
            summarize_layer(
                layer=artifacts["model_b"],
                point_wgs84=point,
                prefix="model_b",
                score_column="probability",
                top_k_values=top_k_values,
            )
        )
        row.update(
            summarize_layer(
                layer=artifacts["model_a_all"],
                point_wgs84=point,
                prefix="model_a_all",
                score_column="cell_max_prob",
                top_k_values=top_k_values,
            )
        )
        row.update(
            summarize_layer(
                layer=artifacts["model_a_positive"],
                point_wgs84=point,
                prefix="model_a_positive",
                score_column="cell_max_prob",
                top_k_values=top_k_values,
            )
        )
        rows.append(row)

    return pd.DataFrame(rows)


def km(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "n/a"
    return f"{float(numeric) / 1000.0:.2f}"


def print_compact_summary(df: pd.DataFrame, top_k_values: list[int]) -> None:
    print("Prospective ranking diagnostic")
    print("=" * 40)
    print(f"Validated fires: {len(df)}")

    if df.empty:
        return

    print("")
    print("Per-fire ranking summary")
    print("-" * 40)

    top_k_cols = [1, 5, 10, 25]
    top_k_cols = [value for value in top_k_cols if value in top_k_values]

    rows: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        out: dict[str, object] = {
            "fire_id": row.fire_id,
            "run_id": row.prediction_run_id,
            "B_nearest_km": km(row.model_b_nearest_distance_m),
            "B_nearest_rank": row.model_b_nearest_rank,
            "A_nearest_km": km(row.model_a_positive_nearest_distance_m),
            "A_nearest_rank": row.model_a_positive_nearest_rank,
        }
        for top_k in top_k_cols:
            out[f"B_top{top_k}_km"] = km(getattr(row, f"model_b_top{top_k}_nearest_distance_m"))
            out[f"A_top{top_k}_km"] = km(getattr(row, f"model_a_positive_top{top_k}_nearest_distance_m"))
        rows.append(out)

    print(pd.DataFrame(rows).to_string(index=False))

    print("")
    print("Median diagnostics")
    print("-" * 40)
    for prefix, label in [
        ("model_b", "Model B"),
        ("model_a_all", "Model A all"),
        ("model_a_positive", "Model A positive"),
    ]:
        nearest = pd.to_numeric(df[f"{prefix}_nearest_distance_m"], errors="coerce") / 1000.0
        rank = pd.to_numeric(df[f"{prefix}_nearest_rank"], errors="coerce")
        print(f"{label} nearest distance median: {nearest.median():.2f} km")
        print(f"{label} nearest rank median: {rank.median():.1f}")
        for top_k in top_k_cols:
            col = f"{prefix}_top{top_k}_nearest_distance_m"
            top_distance = pd.to_numeric(df[col], errors="coerce") / 1000.0
            print(f"{label} top-{top_k} nearest distance median: {top_distance.median():.2f} km")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose candidate ranking around prospective fires.")
    parser.add_argument("--validation-log-csv", default=DEFAULT_LOG_PATH)
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--top-k", default=DEFAULT_TOP_K)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation_log_csv = Path(args.validation_log_csv)
    runs_root = Path(args.runs_root)
    top_k_values = parse_int_list(args.top_k)

    fires = load_validated_fires(validation_log_csv)
    diagnostic = diagnose(fires, runs_root, top_k_values) if not fires.empty else pd.DataFrame()
    print_compact_summary(diagnostic, top_k_values)

    if args.output_csv:
        output = Path(args.output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        diagnostic.to_csv(output, index=False)
        print(f"Output CSV: {output}")


if __name__ == "__main__":
    main()
