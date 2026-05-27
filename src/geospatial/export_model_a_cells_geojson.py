"""Export Model A candidate-cell predictions as GeoJSON polygons.

This utility reads the CSV produced by ``src.inference.run_model_a_geospatial``
and writes one polygon feature per completed candidate cell.

The output is intentionally lightweight compared with a full 25 m heatmap
GeoTIFF. It is useful for dashboard overview maps, where each feature represents
one 1 km x 1 km Model B candidate cell with Model A summary properties.

Example:
    python -m src.geospatial.export_model_a_cells_geojson \
      --predictions-csv results/geospatial/model_a_geospatial_predictions_100.csv \
      --output-geojson results/geospatial/model_a_candidate_cells_100.geojson
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


REQUIRED_COLUMNS = [
    "candidate_id",
    "status",
    "cell_xmin",
    "cell_ymin",
    "cell_xmax",
    "cell_ymax",
    "crs",
]

DEFAULT_PROPERTY_COLUMNS = [
    "candidate_id",
    "coarse_patch_id",
    "cell_max_prob",
    "cell_mean_prob",
    "cell_positive_pixels",
    "final_positive",
    "threshold",
    "min_positive_pixels",
    "cell_probability_tif_path",
    "cell_binary_tif_path",
]


NUMERIC_COLUMNS = {
    "cell_max_prob": float,
    "cell_mean_prob": float,
    "cell_positive_pixels": int,
    "final_positive": int,
    "threshold": float,
    "min_positive_pixels": int,
}


def read_predictions(csv_path: str | Path, include_failed: bool) -> pd.DataFrame:
    """Load and validate Model A geospatial prediction rows."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Prediction CSV missing required columns: {missing}")

    if not include_failed:
        df = df[df["status"] == "completed"].copy()

    if df.empty:
        raise ValueError(f"No rows available for GeoJSON export after filtering: {csv_path}")

    for column in ["cell_xmin", "cell_ymin", "cell_xmax", "cell_ymax"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_bounds = df[["cell_xmin", "cell_ymin", "cell_xmax", "cell_ymax"]].isna().any(axis=1)
    invalid_bounds |= df["cell_xmax"] <= df["cell_xmin"]
    invalid_bounds |= df["cell_ymax"] <= df["cell_ymin"]
    if invalid_bounds.any():
        bad_ids = df.loc[invalid_bounds, "candidate_id"].astype(str).head(10).tolist()
        raise ValueError(f"Invalid candidate-cell bounds for rows: {bad_ids}")

    return df


def infer_single_crs(df: pd.DataFrame) -> str:
    """Return the single CRS used by all exported rows."""
    crs_values = sorted({str(value) for value in df["crs"].dropna().unique() if str(value)})
    if not crs_values:
        raise ValueError("No CRS values found in prediction CSV.")
    if len(crs_values) != 1:
        raise ValueError(f"Expected one CRS for GeoJSON export, found: {crs_values}")
    return crs_values[0]


def normalize_properties(df: pd.DataFrame, property_columns: list[str]) -> pd.DataFrame:
    """Keep requested properties and coerce common numeric fields."""
    available_columns = [column for column in property_columns if column in df.columns]
    out = df[available_columns].copy()

    for column, caster in NUMERIC_COLUMNS.items():
        if column not in out.columns:
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        if caster is int:
            out[column] = values.fillna(0).astype(int)
        else:
            out[column] = values.astype(float)

    out = out.where(pd.notnull(out), None)
    return out


def build_geojson_dataframe(df: pd.DataFrame, property_columns: list[str]) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of 1 km candidate-cell polygons."""
    crs = infer_single_crs(df)
    properties = normalize_properties(df, property_columns)
    geometries = [
        box(float(row.cell_xmin), float(row.cell_ymin), float(row.cell_xmax), float(row.cell_ymax))
        for row in df.itertuples(index=False)
    ]

    gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs=crs)
    return gdf


def write_summary_json(gdf: gpd.GeoDataFrame, output_geojson: str | Path, summary_json: str | Path | None) -> None:
    """Optionally write a compact summary next to the GeoJSON."""
    if not summary_json:
        return

    summary_json = Path(summary_json)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    final_positive_count = None
    if "final_positive" in gdf.columns:
        final_positive_count = int(pd.to_numeric(gdf["final_positive"], errors="coerce").fillna(0).sum())

    max_prob = None
    mean_prob = None
    if "cell_max_prob" in gdf.columns:
        probs = pd.to_numeric(gdf["cell_max_prob"], errors="coerce")
        max_prob = float(probs.max()) if probs.notna().any() else None
        mean_prob = float(probs.mean()) if probs.notna().any() else None

    bounds = [float(value) for value in gdf.total_bounds]
    summary: dict[str, Any] = {
        "output_geojson": str(output_geojson),
        "feature_count": int(len(gdf)),
        "crs": str(gdf.crs),
        "bounds": {
            "left": bounds[0],
            "bottom": bounds[1],
            "right": bounds[2],
            "top": bounds[3],
        },
        "final_positive_count": final_positive_count,
        "cell_max_prob_max": max_prob,
        "cell_max_prob_mean": mean_prob,
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Model A candidate-cell predictions as GeoJSON polygons.")
    parser.add_argument("--predictions-csv", required=True, help="Model A geospatial prediction CSV.")
    parser.add_argument("--output-geojson", required=True, help="Output GeoJSON path.")
    parser.add_argument("--summary-json", default=None, help="Optional JSON summary output path.")
    parser.add_argument("--include-failed", action="store_true", help="Include failed prediction rows when they have valid bounds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = read_predictions(args.predictions_csv, include_failed=args.include_failed)
    gdf = build_geojson_dataframe(df, DEFAULT_PROPERTY_COLUMNS)

    output_geojson = Path(args.output_geojson)
    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_geojson, driver="GeoJSON")
    write_summary_json(gdf, output_geojson, args.summary_json)

    final_positive_count = 0
    if "final_positive" in gdf.columns:
        final_positive_count = int(pd.to_numeric(gdf["final_positive"], errors="coerce").fillna(0).sum())

    print("Model A candidate-cell GeoJSON export complete")
    print(f"Input CSV: {args.predictions_csv}")
    print(f"Output GeoJSON: {output_geojson}")
    if args.summary_json:
        print(f"Summary JSON: {args.summary_json}")
    print(f"Features: {len(gdf)}")
    print(f"CRS: {gdf.crs}")
    print(f"Final positive cells: {final_positive_count}")
    bounds = gdf.total_bounds
    print(
        "Bounds: "
        f"left={bounds[0]:.3f}, bottom={bounds[1]:.3f}, "
        f"right={bounds[2]:.3f}, top={bounds[3]:.3f}"
    )


if __name__ == "__main__":
    main()
