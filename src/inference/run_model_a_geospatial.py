"""Run Model A geospatial inference on generated 25 m NPZ patches.

This script reads the manifest produced by
``src.geospatial.create_model_a_25m_patches_from_candidates``, runs the frozen
Model A spatial refiner, and writes:

1. A CSV summary of Model A predictions per fine patch.
2. Full 64 x 64 context-patch probability and binary GeoTIFFs.
3. Cropped candidate-cell probability and binary GeoTIFFs.

Important geometry:
- Model A input patch: 64 x 64 pixels at 25 m = 1.6 km x 1.6 km.
- Model B candidate cell: 1 km x 1 km = 40 x 40 pixels at 25 m.
- The final heatmap should use the cropped candidate-cell rasters, not the full
  1.6 km context rasters.

Default operational setting:
- Model A: models/model_A_25m_spatial_unet.keras
- threshold: 0.50
- minimum positive pixels: 1

Example:
    python -m src.inference.run_model_a_geospatial \
      --model models/model_A_25m_spatial_unet.keras \
      --channel-stats /mnt/work/wildfire/25m/patches_25m_balanced/channel_stats.json \
      --manifest results/geospatial/model_a_25m_patch_manifest_one_patch.csv \
      --threshold 0.50 \
      --min-positive-pixels 1 \
      --output-csv results/geospatial/model_a_geospatial_predictions_one_patch.csv \
      --probability-dir results/geospatial/model_a_probability_tifs \
      --binary-dir results/geospatial/model_a_binary_tifs \
      --cell-probability-dir results/geospatial/model_a_cell_probability_tifs \
      --cell-binary-dir results/geospatial/model_a_cell_binary_tifs
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import from_origin
import tensorflow as tf


DEFAULT_THRESHOLD = 0.50
DEFAULT_MIN_POSITIVE_PIXELS = 1
DEFAULT_OUTPUT_CSV = "results/geospatial/model_a_geospatial_predictions.csv"
DEFAULT_PROBABILITY_DIR = "results/geospatial/model_a_probability_tifs"
DEFAULT_BINARY_DIR = "results/geospatial/model_a_binary_tifs"
DEFAULT_CELL_PROBABILITY_DIR = "results/geospatial/model_a_cell_probability_tifs"
DEFAULT_CELL_BINARY_DIR = "results/geospatial/model_a_cell_binary_tifs"
EPS = 1e-6


def load_channel_stats(path: str | Path) -> dict[str, Any]:
    """Load channel normalization stats and require explicit feature order."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Channel stats not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    for key in ["mean", "std", "feature_keys"]:
        if key not in stats:
            raise ValueError(f"Channel stats must contain '{key}': {path}")

    mean = stats["mean"]
    std = stats["std"]
    feature_keys = stats["feature_keys"]
    if not isinstance(mean, list) or not isinstance(std, list) or not isinstance(feature_keys, list):
        raise ValueError("channel_stats fields 'mean', 'std', and 'feature_keys' must be lists.")
    if not (len(mean) == len(std) == len(feature_keys)):
        raise ValueError(
            "channel_stats length mismatch: "
            f"mean={len(mean)}, std={len(std)}, feature_keys={len(feature_keys)}"
        )

    return stats


def parse_metadata_json(value: str | None) -> dict[str, Any]:
    """Parse manifest metadata_json safely."""
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def read_manifest_records(manifest_path: str | Path, max_files: int | None) -> list[dict[str, Any]]:
    """Read completed Model A patch records from a manifest CSV."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records: list[dict[str, Any]] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"candidate_id", "status", "npz_path", "patch_xmin", "patch_ymin", "patch_xmax", "patch_ymax", "crs"}
        missing = required.difference(fieldnames)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

        for row in reader:
            if row.get("status") != "completed" or not row.get("npz_path"):
                continue

            metadata = parse_metadata_json(row.get("metadata_json"))
            candidate_id = row.get("candidate_id") or metadata.get("candidate_id") or Path(row["npz_path"]).stem
            coarse_patch_id = row.get("coarse_patch_id") or metadata.get("coarse_patch_id", "")

            records.append(
                {
                    "candidate_id": str(candidate_id),
                    "coarse_patch_id": str(coarse_patch_id),
                    "npz_path": Path(row["npz_path"]),
                    "patch_xmin": float(row["patch_xmin"]),
                    "patch_ymin": float(row["patch_ymin"]),
                    "patch_xmax": float(row["patch_xmax"]),
                    "patch_ymax": float(row["patch_ymax"]),
                    "crs": row.get("crs", ""),
                    "metadata": metadata,
                }
            )

    if max_files is not None:
        records = records[: max(0, max_files)]
    if not records:
        raise ValueError(f"No completed Model A NPZ records found in manifest: {manifest_path}")
    return records


def load_npz_feature_stack(
    npz_path: str | Path,
    feature_keys: list[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Load one Model A NPZ patch as normalized (H, W, C) float32."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ patch not found: {npz_path}")

    with np.load(npz_path) as arrs:
        missing = [key for key in feature_keys if key not in arrs.files]
        extra = [key for key in arrs.files if key not in feature_keys and key != "class"]
        if missing:
            raise ValueError(f"Missing feature key(s) in {npz_path}: {missing}")
        if extra:
            raise ValueError(f"Unexpected feature key(s) in {npz_path}: {extra}")

        arrays = [arrs[key].astype(np.float32) for key in feature_keys]

    first_shape = arrays[0].shape
    if any(arr.shape != first_shape for arr in arrays):
        raise ValueError(f"Feature arrays do not share the same shape in {npz_path}")

    X = np.stack(arrays, axis=-1).astype(np.float32)
    invalid_count = int(np.count_nonzero(~np.isfinite(X)))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = (X - mean) / (std + EPS)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.astype(np.float32), invalid_count


def prediction_to_2d_map(prediction: np.ndarray) -> np.ndarray:
    """Convert one Model A output into a 2D probability map."""
    pred = np.squeeze(np.asarray(prediction, dtype=np.float32))
    if pred.ndim != 2:
        raise ValueError(f"Expected 2D Model A output after squeeze, got shape {pred.shape}")
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    return pred.astype(np.float32)


def patch_bounds(record: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return full Model A context-patch bounds."""
    return (
        float(record["patch_xmin"]),
        float(record["patch_ymin"]),
        float(record["patch_xmax"]),
        float(record["patch_ymax"]),
    )


def cell_bounds(record: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return the original 1 km Model B candidate-cell bounds from metadata."""
    metadata = record.get("metadata", {})
    required = ["cell_xmin", "cell_ymin", "cell_xmax", "cell_ymax"]
    missing = [key for key in required if metadata.get(key, "") == ""]
    if missing:
        raise ValueError(
            f"Candidate {record.get('candidate_id', '')} is missing cell bounds in metadata_json: {missing}"
        )
    return (
        float(metadata["cell_xmin"]),
        float(metadata["cell_ymin"]),
        float(metadata["cell_xmax"]),
        float(metadata["cell_ymax"]),
    )


def make_transform(bounds: tuple[float, float, float, float], width: int, height: int) -> tuple[Any, float, float]:
    """Create raster transform from bounds and output shape."""
    xmin, ymin, xmax, ymax = bounds
    pixel_width = (xmax - xmin) / float(width)
    pixel_height = (ymax - ymin) / float(height)
    if pixel_width <= 0 or pixel_height <= 0:
        raise ValueError(f"Invalid raster bounds: {bounds}")
    return from_origin(xmin, ymax, pixel_width, pixel_height), pixel_width, pixel_height


def crop_prediction_to_cell(
    probability_map: np.ndarray,
    record: dict[str, Any],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Crop a 64 x 64 context prediction to the selected 1 km candidate cell.

    For the standard setup, this returns a 40 x 40 crop because the cell is
    1,000 m wide and Model A resolution is 25 m.
    """
    patch_xmin, patch_ymin, patch_xmax, patch_ymax = patch_bounds(record)
    cell_xmin, cell_ymin, cell_xmax, cell_ymax = cell_bounds(record)

    height, width = probability_map.shape
    xres = (patch_xmax - patch_xmin) / float(width)
    yres = (patch_ymax - patch_ymin) / float(height)
    if xres <= 0 or yres <= 0:
        raise ValueError(f"Invalid patch resolution for {record.get('candidate_id', '')}")

    col0 = int(round((cell_xmin - patch_xmin) / xres))
    col1 = int(round((cell_xmax - patch_xmin) / xres))
    row0 = int(round((patch_ymax - cell_ymax) / yres))
    row1 = int(round((patch_ymax - cell_ymin) / yres))

    if row0 < 0 or col0 < 0 or row1 > height or col1 > width or row1 <= row0 or col1 <= col0:
        raise ValueError(
            f"Candidate-cell crop is outside patch bounds for {record.get('candidate_id', '')}: "
            f"rows {row0}:{row1}, cols {col0}:{col1}, patch shape {height}x{width}"
        )

    crop = probability_map[row0:row1, col0:col1]
    return crop.astype(np.float32), (cell_xmin, cell_ymin, cell_xmax, cell_ymax)


def write_geotiff(
    path: str | Path,
    array: np.ndarray,
    bounds: tuple[float, float, float, float],
    crs: str,
    dtype: str,
) -> None:
    """Write a single-band georeferenced GeoTIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not crs:
        raise ValueError("Cannot write GeoTIFF without CRS.")

    height, width = array.shape
    transform, _, _ = make_transform(bounds, width=width, height=height)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(array.astype(dtype), 1)


def predict_batch(model: tf.keras.Model, batch: np.ndarray) -> np.ndarray:
    """Run Model A prediction."""
    preds = model.predict(batch, verbose=0)
    return np.asarray(preds, dtype=np.float32)


def score_records(
    model: tf.keras.Model,
    records: list[dict[str, Any]],
    stats: dict[str, Any],
    threshold: float,
    min_positive_pixels: int,
    batch_size: int,
    probability_dir: str | Path,
    binary_dir: str | Path,
    cell_probability_dir: str | Path,
    cell_binary_dir: str | Path,
    write_rasters: bool,
) -> list[dict[str, Any]]:
    """Run Model A inference and return CSV-ready rows."""
    feature_keys = [str(key) for key in stats["feature_keys"]]
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(1, 1, -1)

    output_rows: list[dict[str, Any]] = []
    pending_x: list[np.ndarray] = []
    pending_meta: list[dict[str, Any]] = []
    probability_dir = Path(probability_dir)
    binary_dir = Path(binary_dir)
    cell_probability_dir = Path(cell_probability_dir)
    cell_binary_dir = Path(cell_binary_dir)

    def flush_pending() -> None:
        if not pending_x:
            return

        batch = np.stack(pending_x, axis=0)
        preds = predict_batch(model, batch)

        for meta, pred in zip(pending_meta, preds):
            candidate_id = meta["candidate_id"]
            try:
                prob_map = prediction_to_2d_map(pred)
                full_binary_map = (prob_map >= threshold).astype(np.uint8)
                patch_positive_pixels = int(np.count_nonzero(full_binary_map))
                patch_max_prob = float(np.max(prob_map))
                patch_mean_prob = float(np.mean(prob_map))
                patch_final_positive = int(patch_positive_pixels >= min_positive_pixels)

                cell_prob_map, selected_cell_bounds = crop_prediction_to_cell(prob_map, meta)
                cell_binary_map = (cell_prob_map >= threshold).astype(np.uint8)
                cell_positive_pixels = int(np.count_nonzero(cell_binary_map))
                cell_max_prob = float(np.max(cell_prob_map))
                cell_mean_prob = float(np.mean(cell_prob_map))
                final_positive = int(cell_positive_pixels >= min_positive_pixels)

                prob_path = probability_dir / f"{candidate_id}_prob.tif"
                binary_path = binary_dir / f"{candidate_id}_binary.tif"
                cell_prob_path = cell_probability_dir / f"{candidate_id}_cell_prob.tif"
                cell_binary_path = cell_binary_dir / f"{candidate_id}_cell_binary.tif"

                if write_rasters:
                    crs = str(meta.get("crs", ""))
                    full_bounds = patch_bounds(meta)
                    write_geotiff(prob_path, prob_map, full_bounds, crs=crs, dtype="float32")
                    write_geotiff(binary_path, full_binary_map, full_bounds, crs=crs, dtype="uint8")
                    write_geotiff(cell_prob_path, cell_prob_map, selected_cell_bounds, crs=crs, dtype="float32")
                    write_geotiff(cell_binary_path, cell_binary_map, selected_cell_bounds, crs=crs, dtype="uint8")

                output_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "coarse_patch_id": meta.get("coarse_patch_id", ""),
                        "npz_path": meta["npz_path"],
                        "status": "completed",
                        "model_a_max_prob": f"{patch_max_prob:.8f}",
                        "model_a_mean_prob": f"{patch_mean_prob:.8f}",
                        "positive_pixels": cell_positive_pixels,
                        "final_positive": final_positive,
                        "patch_positive_pixels": patch_positive_pixels,
                        "patch_final_positive": patch_final_positive,
                        "cell_max_prob": f"{cell_max_prob:.8f}",
                        "cell_mean_prob": f"{cell_mean_prob:.8f}",
                        "cell_positive_pixels": cell_positive_pixels,
                        "threshold": f"{threshold:.6f}",
                        "min_positive_pixels": min_positive_pixels,
                        "probability_tif_path": str(prob_path) if write_rasters else "",
                        "binary_tif_path": str(binary_path) if write_rasters else "",
                        "cell_probability_tif_path": str(cell_prob_path) if write_rasters else "",
                        "cell_binary_tif_path": str(cell_binary_path) if write_rasters else "",
                        "patch_xmin": f"{float(meta['patch_xmin']):.3f}",
                        "patch_ymin": f"{float(meta['patch_ymin']):.3f}",
                        "patch_xmax": f"{float(meta['patch_xmax']):.3f}",
                        "patch_ymax": f"{float(meta['patch_ymax']):.3f}",
                        "cell_xmin": f"{selected_cell_bounds[0]:.3f}",
                        "cell_ymin": f"{selected_cell_bounds[1]:.3f}",
                        "cell_xmax": f"{selected_cell_bounds[2]:.3f}",
                        "cell_ymax": f"{selected_cell_bounds[3]:.3f}",
                        "crs": meta.get("crs", ""),
                        "invalid_values_before_cleaning": meta["invalid_values_before_cleaning"],
                        "message": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                output_rows.append(failed_row(meta, threshold, min_positive_pixels, str(exc)))

        pending_x.clear()
        pending_meta.clear()

    for record in records:
        try:
            X, invalid_count = load_npz_feature_stack(record["npz_path"], feature_keys, mean, std)
            pending_x.append(X)
            pending_meta.append(
                {
                    **record,
                    "npz_path": str(record["npz_path"]),
                    "invalid_values_before_cleaning": invalid_count,
                }
            )
            if len(pending_x) >= batch_size:
                flush_pending()
        except Exception as exc:  # noqa: BLE001
            output_rows.append(failed_row(record, threshold, min_positive_pixels, str(exc)))

    flush_pending()
    return output_rows


def failed_row(record: dict[str, Any], threshold: float, min_positive_pixels: int, message: str) -> dict[str, Any]:
    """Create a failed output row for robust large jobs."""
    return {
        "candidate_id": record.get("candidate_id", ""),
        "coarse_patch_id": record.get("coarse_patch_id", ""),
        "npz_path": str(record.get("npz_path", "")),
        "status": "failed",
        "model_a_max_prob": "",
        "model_a_mean_prob": "",
        "positive_pixels": "",
        "final_positive": "",
        "patch_positive_pixels": "",
        "patch_final_positive": "",
        "cell_max_prob": "",
        "cell_mean_prob": "",
        "cell_positive_pixels": "",
        "threshold": f"{threshold:.6f}",
        "min_positive_pixels": min_positive_pixels,
        "probability_tif_path": "",
        "binary_tif_path": "",
        "cell_probability_tif_path": "",
        "cell_binary_tif_path": "",
        "patch_xmin": record.get("patch_xmin", ""),
        "patch_ymin": record.get("patch_ymin", ""),
        "patch_xmax": record.get("patch_xmax", ""),
        "patch_ymax": record.get("patch_ymax", ""),
        "cell_xmin": "",
        "cell_ymin": "",
        "cell_xmax": "",
        "cell_ymax": "",
        "crs": record.get("crs", ""),
        "invalid_values_before_cleaning": "",
        "message": message,
    }


def write_output_csv(rows: list[dict[str, Any]], output_csv: str | Path) -> None:
    """Write Model A geospatial prediction summary."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_id",
        "coarse_patch_id",
        "npz_path",
        "status",
        "model_a_max_prob",
        "model_a_mean_prob",
        "positive_pixels",
        "final_positive",
        "patch_positive_pixels",
        "patch_final_positive",
        "cell_max_prob",
        "cell_mean_prob",
        "cell_positive_pixels",
        "threshold",
        "min_positive_pixels",
        "probability_tif_path",
        "binary_tif_path",
        "cell_probability_tif_path",
        "cell_binary_tif_path",
        "patch_xmin",
        "patch_ymin",
        "patch_xmax",
        "patch_ymax",
        "cell_xmin",
        "cell_ymin",
        "cell_xmax",
        "cell_ymax",
        "crs",
        "invalid_values_before_cleaning",
        "message",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]], output_csv: str | Path) -> None:
    """Print compact Model A inference summary."""
    completed = [row for row in rows if row["status"] == "completed"]
    failed = [row for row in rows if row["status"] != "completed"]
    final_positive = [row for row in completed if int(row["final_positive"]) == 1]

    print("Model A geospatial inference complete")
    print(f"Rows: {len(rows)}")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")
    print(f"Final positive candidate cells: {len(final_positive)}")
    if completed:
        print(f"Final positive candidate-cell rate: {len(final_positive) / len(completed):.4f}")
    if failed:
        print("First failure messages:")
        for row in failed[:5]:
            candidate_id = row.get("candidate_id", "")
            message = row.get("message", "")
            print(f"  {candidate_id}: {message}")
    print(f"Output CSV: {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Model A geospatial inference on generated 25 m NPZ patches.")
    parser.add_argument("--model", required=True, help="Path to trained Model A .keras file.")
    parser.add_argument("--channel-stats", required=True, help="25 m channel_stats.json path.")
    parser.add_argument("--manifest", required=True, help="Model A 25 m patch manifest CSV.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Model A probability threshold.")
    parser.add_argument(
        "--min-positive-pixels",
        type=int,
        default=DEFAULT_MIN_POSITIVE_PIXELS,
        help="Minimum positive pixels required for final_positive.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="Output prediction summary CSV.")
    parser.add_argument("--probability-dir", default=DEFAULT_PROBABILITY_DIR, help="Output directory for full-patch probability GeoTIFFs.")
    parser.add_argument("--binary-dir", default=DEFAULT_BINARY_DIR, help="Output directory for full-patch binary GeoTIFFs.")
    parser.add_argument(
        "--cell-probability-dir",
        default=DEFAULT_CELL_PROBABILITY_DIR,
        help="Output directory for cropped 1 km candidate-cell probability GeoTIFFs.",
    )
    parser.add_argument(
        "--cell-binary-dir",
        default=DEFAULT_CELL_BINARY_DIR,
        help="Output directory for cropped 1 km candidate-cell binary GeoTIFFs.",
    )
    parser.add_argument("--no-rasters", action="store_true", help="Skip writing GeoTIFF rasters and write CSV only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.min_positive_pixels < 1:
        raise ValueError("--min-positive-pixels must be >= 1.")

    stats = load_channel_stats(args.channel_stats)
    records = read_manifest_records(args.manifest, max_files=args.max_files)

    print("Loading Model A")
    print(f"Model: {args.model}")
    model = tf.keras.models.load_model(args.model, compile=False)

    print("Running Model A geospatial inference")
    print(f"Patches selected: {len(records)}")
    print(f"Channels: {len(stats['feature_keys'])}")
    print(f"Threshold: {args.threshold}")
    print(f"Min positive pixels: {args.min_positive_pixels}")
    print(f"Write rasters: {not args.no_rasters}")

    rows = score_records(
        model=model,
        records=records,
        stats=stats,
        threshold=args.threshold,
        min_positive_pixels=args.min_positive_pixels,
        batch_size=args.batch_size,
        probability_dir=args.probability_dir,
        binary_dir=args.binary_dir,
        cell_probability_dir=args.cell_probability_dir,
        cell_binary_dir=args.cell_binary_dir,
        write_rasters=not args.no_rasters,
    )
    write_output_csv(rows, args.output_csv)
    print_summary(rows, args.output_csv)


if __name__ == "__main__":
    main()
