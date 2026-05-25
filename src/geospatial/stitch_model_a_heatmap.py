"""Stitch Model A probability GeoTIFFs into one heatmap.

Reads the CSV from src.inference.run_model_a_geospatial and merges completed
probability_tif_path rasters. Overlap is resolved by maximum probability.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import from_origin


DEFAULT_OUTPUT_TIF = "results/geospatial/alberta_model_a_heatmap.tif"
DEFAULT_THRESHOLD = 0.50
TOL = 1e-6


def read_probability_paths(csv_path: str | Path, max_files: int | None) -> list[Path]:
    """Return completed probability GeoTIFF paths from prediction CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {csv_path}")

    paths: list[Path] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"status", "probability_tif_path"}
        missing = required.difference(fields)
        if missing:
            raise ValueError(f"Prediction CSV missing columns: {sorted(missing)}")

        for row in reader:
            if row.get("status") != "completed":
                continue
            value = row.get("probability_tif_path", "")
            if not value:
                continue
            path = Path(value)
            if not path.exists():
                raise FileNotFoundError(f"Probability GeoTIFF not found: {path}")
            paths.append(path)

    if max_files is not None:
        paths = paths[: max(0, max_files)]
    if not paths:
        raise ValueError(f"No completed probability GeoTIFFs found in {csv_path}")
    return paths


def build_output_metadata(paths: list[Path]) -> dict[str, Any]:
    """Validate rasters and compute output mosaic transform/shape."""
    first_crs = None
    first_xres = None
    first_yres = None
    bounds = []

    for path in paths:
        with rasterio.open(path) as src:
            if src.count != 1:
                raise ValueError(f"Expected one-band raster: {path}")
            if src.crs is None:
                raise ValueError(f"Raster has no CRS: {path}")

            xres = abs(float(src.transform.a))
            yres = abs(float(src.transform.e))
            if first_crs is None:
                first_crs = src.crs
                first_xres = xres
                first_yres = yres
            else:
                if src.crs != first_crs:
                    raise ValueError(f"CRS mismatch for {path}: {src.crs} != {first_crs}")
                if abs(xres - float(first_xres)) > TOL or abs(yres - float(first_yres)) > TOL:
                    raise ValueError(f"Resolution mismatch for {path}")
            bounds.append(src.bounds)

    left = min(b.left for b in bounds)
    bottom = min(b.bottom for b in bounds)
    right = max(b.right for b in bounds)
    top = max(b.top for b in bounds)
    xres = float(first_xres)
    yres = float(first_yres)
    width = int(math.ceil((right - left) / xres))
    height = int(math.ceil((top - bottom) / yres))

    return {
        "crs": first_crs,
        "left": left,
        "bottom": bottom,
        "right": right,
        "top": top,
        "xres": xres,
        "yres": yres,
        "width": width,
        "height": height,
        "transform": from_origin(left, top, xres, yres),
    }


def paste_max(mosaic: np.ndarray, path: Path, meta: dict[str, Any]) -> None:
    """Paste one raster into mosaic using max probability."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)

        col0 = int(round((src.bounds.left - meta["left"]) / meta["xres"]))
        row0 = int(round((meta["top"] - src.bounds.top) / meta["yres"]))
        row1 = row0 + src.height
        col1 = col0 + src.width

        if row0 < 0 or col0 < 0 or row1 > mosaic.shape[0] or col1 > mosaic.shape[1]:
            raise ValueError(f"Raster outside output mosaic: {path}")

        window = mosaic[row0:row1, col0:col1]
        np.maximum(window, arr, out=window)


def write_tif(path: str | Path, arr: np.ndarray, meta: dict[str, Any], dtype: str) -> None:
    """Write one-band GeoTIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype=dtype,
        crs=meta["crs"],
        transform=meta["transform"],
        compress="deflate",
    ) as dst:
        dst.write(arr.astype(dtype), 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitch Model A probability GeoTIFFs into one heatmap.")
    parser.add_argument("--predictions-csv", required=True, help="Model A geospatial prediction CSV.")
    parser.add_argument("--output-tif", default=DEFAULT_OUTPUT_TIF, help="Output probability heatmap GeoTIFF.")
    parser.add_argument("--binary-output-tif", default=None, help="Optional binary heatmap GeoTIFF.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Binary threshold.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional smoke-test cap.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    paths = read_probability_paths(args.predictions_csv, args.max_files)
    meta = build_output_metadata(paths)
    mosaic = np.zeros((meta["height"], meta["width"]), dtype=np.float32)

    for path in paths:
        paste_max(mosaic, path, meta)

    write_tif(args.output_tif, mosaic, meta, dtype="float32")

    positive_pixels = int(np.count_nonzero(mosaic >= args.threshold))
    if args.binary_output_tif:
        binary = (mosaic >= args.threshold).astype(np.uint8)
        write_tif(args.binary_output_tif, binary, meta, dtype="uint8")

    print("Model A heatmap stitching complete")
    print(f"Input rasters: {len(paths)}")
    print(f"Output probability heatmap: {args.output_tif}")
    if args.binary_output_tif:
        print(f"Output binary heatmap: {args.binary_output_tif}")
    print(f"CRS: {meta['crs']}")
    print(f"Shape: {meta['height']} x {meta['width']}")
    print(
        "Bounds: "
        f"left={meta['left']:.3f}, bottom={meta['bottom']:.3f}, "
        f"right={meta['right']:.3f}, top={meta['top']:.3f}"
    )
    print(f"Probability min/max/mean: {float(mosaic.min()):.8f} / {float(mosaic.max()):.8f} / {float(mosaic.mean()):.8f}")
    print(f"Positive pixels at threshold {args.threshold:.3f}: {positive_pixels}")


if __name__ == "__main__":
    main()
