"""Create Model-B-sized coarse inference grid over Alberta.

This is the first geospatial foundation step for the coarse-to-fine
wildfire ignition pipeline.

Terminology used here:
- coarse patch: one Model B inference footprint
- Model B patch tensor: patch_size x patch_size pixels at resolution_m metres
- default footprint: 32 x 32 pixels at 1000 m = 32 km x 32 km

Example:
    python -m src.geospatial.create_alberta_coarse_grid \
      --boundary /home/bondada.romith/wildfire/NFBD/Alberta_boundary.shp \
      --output-geojson data/grids/alberta_coarse_grid.geojson \
      --output-parquet data/grids/alberta_coarse_grid.parquet \
      --patch-size 32 \
      --resolution-m 1000
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box


DEFAULT_TARGET_CRS = "EPSG:3400"
DEFAULT_PATCH_SIZE = 32
DEFAULT_RESOLUTION_M = 1000


def load_boundary(boundary_path: str | Path, target_crs: str) -> gpd.GeoDataFrame:
    """Load and dissolve the Alberta boundary into one geometry."""
    boundary_path = Path(boundary_path)
    if not boundary_path.exists():
        raise FileNotFoundError(f"Boundary file not found: {boundary_path}")

    boundary = gpd.read_file(boundary_path)
    if boundary.empty:
        raise ValueError(f"Boundary file is empty: {boundary_path}")

    if boundary.crs is None:
        raise ValueError(
            f"Boundary file has no CRS: {boundary_path}. "
            "Ensure the .prj sidecar exists and is readable."
        )

    boundary = boundary.to_crs(target_crs)
    dissolved_geom = boundary.geometry.union_all()

    return gpd.GeoDataFrame(
        {"name": ["alberta_boundary"]},
        geometry=[dissolved_geom],
        crs=target_crs,
    )


def make_grid(boundary: gpd.GeoDataFrame, cell_size_m: float) -> gpd.GeoDataFrame:
    """Create square grid cells intersecting the boundary geometry."""
    if cell_size_m <= 0:
        raise ValueError(f"cell_size_m must be positive, got {cell_size_m}")

    boundary_geom = boundary.geometry.iloc[0]
    minx, miny, maxx, maxy = boundary.total_bounds

    start_x = math.floor(minx / cell_size_m) * cell_size_m
    start_y = math.floor(miny / cell_size_m) * cell_size_m
    end_x = math.ceil(maxx / cell_size_m) * cell_size_m
    end_y = math.ceil(maxy / cell_size_m) * cell_size_m

    records: list[dict] = []
    patch_counter = 0

    y = start_y
    while y < end_y:
        x = start_x
        while x < end_x:
            cell = box(x, y, x + cell_size_m, y + cell_size_m)
            if cell.intersects(boundary_geom):
                patch_counter += 1
                clipped = cell.intersection(boundary_geom)
                centroid = cell.centroid
                records.append(
                    {
                        "patch_id": f"ab_coarse_{patch_counter:06d}",
                        "xmin": float(x),
                        "ymin": float(y),
                        "xmax": float(x + cell_size_m),
                        "ymax": float(y + cell_size_m),
                        "centroid_x": float(centroid.x),
                        "centroid_y": float(centroid.y),
                        "cell_size_m": float(cell_size_m),
                        "geometry": clipped,
                    }
                )
            x += cell_size_m
        y += cell_size_m

    if not records:
        raise ValueError("No grid cells intersected the boundary. Check CRS and cell size.")

    return gpd.GeoDataFrame(records, geometry="geometry", crs=boundary.crs)


def save_grid(grid: gpd.GeoDataFrame, output_geojson: Path, output_parquet: Path | None) -> None:
    """Save grid to GeoJSON and optionally GeoParquet."""
    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    grid.to_file(output_geojson, driver="GeoJSON")
    print(f"Saved GeoJSON grid: {output_geojson}")

    if output_parquet is not None:
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        grid.to_parquet(output_parquet, index=False)
        print(f"Saved Parquet grid: {output_parquet}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Model-B-sized coarse inference grid over Alberta."
    )
    parser.add_argument(
        "--boundary",
        required=True,
        help="Path to Alberta boundary shapefile or vector file.",
    )
    parser.add_argument(
        "--output-geojson",
        default="data/grids/alberta_coarse_grid.geojson",
        help="Output GeoJSON grid path.",
    )
    parser.add_argument(
        "--output-parquet",
        default="data/grids/alberta_coarse_grid.parquet",
        help="Optional output GeoParquet grid path. Use empty string to skip.",
    )
    parser.add_argument(
        "--target-crs",
        default=DEFAULT_TARGET_CRS,
        help="Projected CRS for metric grid creation. Default: EPSG:3400.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help="Model B patch width/height in pixels. Default: 32.",
    )
    parser.add_argument(
        "--resolution-m",
        type=float,
        default=DEFAULT_RESOLUTION_M,
        help="Model B pixel resolution in metres. Default: 1000.",
    )
    parser.add_argument(
        "--cell-size-m",
        type=float,
        default=None,
        help="Override coarse grid cell size in metres. Defaults to patch_size * resolution_m.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.cell_size_m is None:
        cell_size_m = args.patch_size * args.resolution_m
    else:
        cell_size_m = args.cell_size_m

    print("Creating Alberta coarse grid")
    print(f"Boundary: {args.boundary}")
    print(f"Target CRS: {args.target_crs}")
    print(f"Patch size: {args.patch_size} px")
    print(f"Resolution: {args.resolution_m} m/px")
    print(f"Coarse footprint: {cell_size_m} m x {cell_size_m} m")

    boundary = load_boundary(args.boundary, args.target_crs)
    grid = make_grid(boundary, cell_size_m)

    output_parquet = Path(args.output_parquet) if args.output_parquet else None
    save_grid(grid, Path(args.output_geojson), output_parquet)

    print("Grid summary")
    print(f"Cells: {len(grid)}")
    print(f"CRS: {grid.crs}")
    print(f"Bounds: {grid.total_bounds.tolist()}")


if __name__ == "__main__":
    main()
