"""Extract Model B 1 km inference patches from aligned raster features.

This module converts coarse grid footprints into ``.npz`` tensors for the
Model B gatekeeper. It is intentionally config-driven so local raster paths are
not hardcoded in the repository.

Expected feature config JSON:

{
  "features": [
    {
      "key": "elevation",
      "path": "/path/to/aligned/elevation_1km.tif",
      "band": 1,
      "resampling": "bilinear",
      "fill_value": 0.0
    },
    {
      "key": "landcover",
      "path": "/path/to/aligned/landcover_1km.tif",
      "band": 1,
      "resampling": "nearest",
      "fill_value": 0.0
    }
  ]
}

Example:
    python -m src.geospatial.extract_model_b_patch \
      --grid data/grids/alberta_coarse_grid.geojson \
      --feature-config configs/model_b_1km_features.json \
      --output-dir data/patches/1km \
      --patch-id ab_coarse_000001

To extract multiple patches:
    python -m src.geospatial.extract_model_b_patch \
      --grid data/grids/alberta_coarse_grid.geojson \
      --feature-config configs/model_b_1km_features.json \
      --output-dir data/patches/1km \
      --all \
      --max-patches 100
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject


DEFAULT_PATCH_SIZE = 32
DEFAULT_RESOLUTION_M = 1000.0
DEFAULT_FILL_VALUE = 0.0


@dataclass(frozen=True)
class FeatureSpec:
    """Configuration for one raster feature layer."""

    key: str
    path: Path
    band: int = 1
    resampling: str = "bilinear"
    fill_value: float = DEFAULT_FILL_VALUE


RESAMPLING_MAP: dict[str, Resampling] = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
    "mode": Resampling.mode,
}


def load_feature_config(path: str | Path) -> list[FeatureSpec]:
    """Load feature raster configuration from JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Feature config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    features = config.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("Feature config must contain a non-empty 'features' list.")

    specs: list[FeatureSpec] = []
    seen_keys: set[str] = set()

    for i, item in enumerate(features):
        if not isinstance(item, dict):
            raise ValueError(f"Feature config entry {i} must be an object.")

        key = item.get("key")
        raster_path = item.get("path")
        if not key or not isinstance(key, str):
            raise ValueError(f"Feature config entry {i} is missing string key.")
        if not raster_path or not isinstance(raster_path, str):
            raise ValueError(f"Feature config entry {i} is missing string path.")
        if key in seen_keys:
            raise ValueError(f"Duplicate feature key in config: {key}")

        resampling = item.get("resampling", "bilinear")
        if resampling not in RESAMPLING_MAP:
            valid = ", ".join(sorted(RESAMPLING_MAP))
            raise ValueError(f"Invalid resampling '{resampling}' for {key}. Valid: {valid}")

        spec = FeatureSpec(
            key=key,
            path=Path(raster_path),
            band=int(item.get("band", 1)),
            resampling=resampling,
            fill_value=float(item.get("fill_value", DEFAULT_FILL_VALUE)),
        )
        if not spec.path.exists():
            raise FileNotFoundError(f"Raster for feature '{key}' not found: {spec.path}")
        if spec.band < 1:
            raise ValueError(f"Band index must be >= 1 for feature '{key}'")

        specs.append(spec)
        seen_keys.add(key)

    return specs


def load_grid(path: str | Path) -> gpd.GeoDataFrame:
    """Load coarse grid from GeoJSON, shapefile, or GeoParquet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Grid file not found: {path}")

    if path.suffix.lower() in {".parquet", ".geoparquet"}:
        grid = gpd.read_parquet(path)
    else:
        grid = gpd.read_file(path)

    required_columns = {"patch_id", "xmin", "ymin", "xmax", "ymax"}
    missing = required_columns.difference(grid.columns)
    if missing:
        raise ValueError(f"Grid is missing required columns: {sorted(missing)}")
    if grid.crs is None:
        raise ValueError("Grid has no CRS. Recreate it with create_alberta_coarse_grid.py.")
    if grid.empty:
        raise ValueError(f"Grid file is empty: {path}")

    return grid


def select_grid_rows(
    grid: gpd.GeoDataFrame,
    patch_id: str | None,
    extract_all: bool,
    max_patches: int | None,
) -> gpd.GeoDataFrame:
    """Select rows for extraction."""
    if patch_id and extract_all:
        raise ValueError("Use either --patch-id or --all, not both.")
    if not patch_id and not extract_all:
        raise ValueError("Provide --patch-id for one patch or --all for multiple patches.")

    if patch_id:
        selected = grid[grid["patch_id"] == patch_id]
        if selected.empty:
            raise ValueError(f"Patch ID not found in grid: {patch_id}")
    else:
        selected = grid.sort_values("patch_id")
        if max_patches is not None:
            selected = selected.head(max_patches)

    return selected.reset_index(drop=True)


def extract_feature_array(
    spec: FeatureSpec,
    bounds: tuple[float, float, float, float],
    dst_crs: Any,
    patch_size: int,
    resolution_m: float,
) -> np.ndarray:
    """Sample one raster feature into a fixed-size Model B patch array."""
    xmin, ymin, xmax, ymax = bounds
    expected_width = patch_size * resolution_m
    expected_height = patch_size * resolution_m

    actual_width = xmax - xmin
    actual_height = ymax - ymin
    if abs(actual_width - expected_width) > 1e-3 or abs(actual_height - expected_height) > 1e-3:
        raise ValueError(
            f"Grid bounds size {actual_width} x {actual_height} does not match "
            f"patch_size * resolution_m = {expected_width} x {expected_height}."
        )

    dst_transform = from_origin(xmin, ymax, resolution_m, resolution_m)
    dst = np.full((patch_size, patch_size), spec.fill_value, dtype=np.float32)

    with rasterio.open(spec.path) as src:
        if spec.band > src.count:
            raise ValueError(
                f"Feature '{spec.key}' requested band {spec.band}, "
                f"but raster only has {src.count} band(s): {spec.path}"
            )

        reproject(
            source=rasterio.band(src, spec.band),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=spec.fill_value,
            resampling=RESAMPLING_MAP[spec.resampling],
        )

    dst = np.nan_to_num(dst, nan=spec.fill_value, posinf=spec.fill_value, neginf=spec.fill_value)
    return dst.astype(np.float32)


def build_patch_metadata(
    row: Any,
    feature_specs: list[FeatureSpec],
    patch_size: int,
    resolution_m: float,
    grid_crs: Any,
) -> dict[str, Any]:
    """Create lightweight JSON-serializable metadata for an output patch."""
    return {
        "patch_id": str(row.patch_id),
        "xmin": float(row.xmin),
        "ymin": float(row.ymin),
        "xmax": float(row.xmax),
        "ymax": float(row.ymax),
        "patch_size": int(patch_size),
        "resolution_m": float(resolution_m),
        "crs": str(grid_crs),
        "feature_keys": [spec.key for spec in feature_specs],
        "feature_paths": [str(spec.path) for spec in feature_specs],
    }


def extract_patch(
    row: Any,
    feature_specs: list[FeatureSpec],
    output_dir: Path,
    patch_size: int,
    resolution_m: float,
    grid_crs: Any,
    overwrite: bool,
) -> Path:
    """Extract all configured features for one coarse grid row and save NPZ."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{row.patch_id}.npz"
    if output_path.exists() and not overwrite:
        return output_path

    bounds = (float(row.xmin), float(row.ymin), float(row.xmax), float(row.ymax))
    arrays: dict[str, np.ndarray] = {}

    for spec in feature_specs:
        arrays[spec.key] = extract_feature_array(
            spec=spec,
            bounds=bounds,
            dst_crs=grid_crs,
            patch_size=patch_size,
            resolution_m=resolution_m,
        )

    metadata = build_patch_metadata(
        row=row,
        feature_specs=feature_specs,
        patch_size=patch_size,
        resolution_m=resolution_m,
        grid_crs=grid_crs,
    )
    arrays["_metadata_json"] = np.array(json.dumps(metadata), dtype=object)

    np.savez_compressed(output_path, **arrays)
    return output_path


def write_manifest(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write extraction manifest CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["patch_id", "status", "npz_path", "message"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved manifest: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Model B 1 km NPZ patches from Alberta coarse grid footprints."
    )
    parser.add_argument("--grid", required=True, help="Coarse grid GeoJSON/Parquet path.")
    parser.add_argument("--feature-config", required=True, help="JSON config listing aligned raster features.")
    parser.add_argument("--output-dir", default="data/patches/1km", help="Output folder for NPZ patches.")
    parser.add_argument("--patch-id", default=None, help="Extract one patch by patch_id.")
    parser.add_argument("--all", action="store_true", help="Extract all grid patches.")
    parser.add_argument("--max-patches", type=int, default=None, help="Optional cap when using --all.")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE, help="Model B patch size in pixels.")
    parser.add_argument("--resolution-m", type=float, default=DEFAULT_RESOLUTION_M, help="Model B resolution in metres per pixel.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing NPZ files.")
    parser.add_argument(
        "--manifest",
        default="results/geospatial/model_b_patch_extraction_manifest.csv",
        help="Output CSV manifest path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    feature_specs = load_feature_config(args.feature_config)
    grid = load_grid(args.grid)
    selected = select_grid_rows(
        grid=grid,
        patch_id=args.patch_id,
        extract_all=args.all,
        max_patches=args.max_patches,
    )

    print("Extracting Model B geospatial patches")
    print(f"Grid: {args.grid}")
    print(f"Feature count: {len(feature_specs)}")
    print(f"Selected patches: {len(selected)}")
    print(f"Patch shape: {args.patch_size} x {args.patch_size}")
    print(f"Resolution: {args.resolution_m} m/px")
    print(f"Output dir: {args.output_dir}")

    manifest_rows: list[dict[str, Any]] = []
    output_dir = Path(args.output_dir)

    for row in selected.itertuples(index=False):
        patch_id = str(row.patch_id)
        try:
            npz_path = extract_patch(
                row=row,
                feature_specs=feature_specs,
                output_dir=output_dir,
                patch_size=args.patch_size,
                resolution_m=args.resolution_m,
                grid_crs=grid.crs,
                overwrite=args.overwrite,
            )
            manifest_rows.append(
                {
                    "patch_id": patch_id,
                    "status": "completed",
                    "npz_path": str(npz_path),
                    "message": "",
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep extraction robust over large grids.
            manifest_rows.append(
                {
                    "patch_id": patch_id,
                    "status": "failed",
                    "npz_path": "",
                    "message": str(exc),
                }
            )
            print(f"Failed {patch_id}: {exc}")

    write_manifest(manifest_rows, Path(args.manifest))

    completed = sum(row["status"] == "completed" for row in manifest_rows)
    failed = sum(row["status"] == "failed" for row in manifest_rows)
    print("Extraction summary")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
