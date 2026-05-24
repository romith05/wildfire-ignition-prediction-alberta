"""Run Model B gatekeeper inference on geospatial 1 km NPZ patches.

This script scores coarse Alberta inference patches created by
``src.geospatial.extract_model_b_patch``. It also writes a second CSV containing
Model-B-selected 1 km candidate cells inside each 32 km coarse patch.

This is the bridge to the fine Model A stage:
- coarse patch: 32 x 32 pixels at 1 km resolution = 32 km x 32 km
- Model B output map: one probability per 1 km pixel
- candidate CSV: one row per 1 km pixel whose probability passes threshold

Default operational setting:
- Model B: models/model_B_1km_gatekeeper_hardneg_phase2.keras
- threshold: 0.30

Example:
    python -m src.inference.run_model_b_geospatial \
      --model models/model_B_1km_gatekeeper_hardneg_phase2.keras \
      --channel-stats /mnt/work/wildfire/1km/patches_1km_balanced/channel_stats.json \
      --manifest results/geospatial/model_b_patch_extraction_manifest.csv \
      --threshold 0.30 \
      --output-csv results/geospatial/model_b_geospatial_scores.csv \
      --candidate-csv results/geospatial/model_b_candidate_1km_cells.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


DEFAULT_THRESHOLD = 0.30
DEFAULT_PATTERN = "*.npz"
DEFAULT_CANDIDATE_CSV = "results/geospatial/model_b_candidate_1km_cells.csv"
EPS = 1e-6


def load_channel_stats(path: str | Path) -> dict[str, Any]:
    """Load normalization stats and require explicit feature-key order."""
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
    """Parse manifest metadata_json field if present."""
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def read_manifest_records(manifest_path: str | Path) -> list[dict[str, Any]]:
    """Read completed NPZ records from extraction manifest CSV."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    records: list[dict[str, Any]] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "npz_path" not in (reader.fieldnames or []):
            raise ValueError(f"Manifest must contain 'npz_path' column: {manifest_path}")
        for row in reader:
            status = row.get("status", "completed")
            npz_path = row.get("npz_path", "")
            if status != "completed" or not npz_path:
                continue

            metadata = parse_metadata_json(row.get("metadata_json"))
            patch_id = row.get("patch_id") or metadata.get("patch_id") or Path(npz_path).stem
            records.append(
                {
                    "patch_id": str(patch_id),
                    "npz_path": Path(npz_path),
                    "metadata": metadata,
                }
            )

    if not records:
        raise ValueError(f"No completed NPZ paths found in manifest: {manifest_path}")
    return records


def collect_npz_records(
    patch_dir: str | Path | None,
    manifest: str | Path | None,
    pattern: str,
    max_files: int | None,
) -> list[dict[str, Any]]:
    """Collect NPZ records from either a manifest or a folder."""
    if manifest and patch_dir:
        raise ValueError("Use either --manifest or --patch-dir, not both.")
    if not manifest and not patch_dir:
        raise ValueError("Provide either --manifest or --patch-dir.")

    if manifest:
        records = read_manifest_records(manifest)
    else:
        patch_dir = Path(patch_dir)  # type: ignore[arg-type]
        if not patch_dir.exists():
            raise FileNotFoundError(f"Patch dir not found: {patch_dir}")
        paths = sorted(patch_dir.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No NPZ files found in {patch_dir} using pattern {pattern}")
        records = [
            {
                "patch_id": path.stem,
                "npz_path": path,
                "metadata": {},
            }
            for path in paths
        ]

    if max_files is not None:
        records = records[: max(0, max_files)]
    if not records:
        raise ValueError("No NPZ files selected for inference.")
    return records


def load_npz_feature_stack(
    npz_path: str | Path,
    feature_keys: list[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Load one NPZ patch as normalized ``(H, W, C)`` float32 array."""
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


def predict_batch(model: tf.keras.Model, batch: np.ndarray) -> np.ndarray:
    """Run model prediction and return raw probabilities as numpy array."""
    preds = model.predict(batch, verbose=0)
    return np.asarray(preds, dtype=np.float32)


def prediction_to_2d_map(prediction: np.ndarray) -> np.ndarray:
    """Convert one model output into a 2D probability map.

    Model B is expected to output a spatial map for coarse-to-fine inference.
    A scalar output is tolerated for robustness and treated as a 1x1 map over
    the full coarse patch.
    """
    pred = np.squeeze(np.asarray(prediction, dtype=np.float32))
    if pred.ndim == 0:
        return pred.reshape(1, 1)
    if pred.ndim == 2:
        return pred
    raise ValueError(f"Expected scalar or 2D Model B output after squeeze, got shape {pred.shape}")


def infer_cell_bounds(
    metadata: dict[str, Any],
    row: int,
    col: int,
    rows: int,
    cols: int,
) -> dict[str, str]:
    """Infer geospatial bounds for a candidate Model B output cell."""
    required = ["xmin", "ymin", "xmax", "ymax"]
    if any(key not in metadata for key in required):
        return {
            "cell_xmin": "",
            "cell_ymin": "",
            "cell_xmax": "",
            "cell_ymax": "",
            "coarse_xmin": "",
            "coarse_ymin": "",
            "coarse_xmax": "",
            "coarse_ymax": "",
            "crs": str(metadata.get("crs", "")),
        }

    xmin = float(metadata["xmin"])
    ymin = float(metadata["ymin"])
    xmax = float(metadata["xmax"])
    ymax = float(metadata["ymax"])

    cell_width = (xmax - xmin) / float(cols)
    cell_height = (ymax - ymin) / float(rows)

    cell_xmin = xmin + col * cell_width
    cell_xmax = xmin + (col + 1) * cell_width
    cell_ymax = ymax - row * cell_height
    cell_ymin = ymax - (row + 1) * cell_height

    return {
        "cell_xmin": f"{cell_xmin:.3f}",
        "cell_ymin": f"{cell_ymin:.3f}",
        "cell_xmax": f"{cell_xmax:.3f}",
        "cell_ymax": f"{cell_ymax:.3f}",
        "coarse_xmin": f"{xmin:.3f}",
        "coarse_ymin": f"{ymin:.3f}",
        "coarse_xmax": f"{xmax:.3f}",
        "coarse_ymax": f"{ymax:.3f}",
        "crs": str(metadata.get("crs", "")),
    }


def build_candidate_rows(
    patch_id: str,
    npz_path: str,
    prediction_map: np.ndarray,
    metadata: dict[str, Any],
    candidate_threshold: float,
) -> list[dict[str, Any]]:
    """Create one row per Model-B-selected 1 km candidate cell."""
    rows, cols = prediction_map.shape
    selected = np.argwhere(prediction_map >= candidate_threshold)
    candidate_rows: list[dict[str, Any]] = []

    for row, col in selected:
        prob = float(prediction_map[row, col])
        bounds = infer_cell_bounds(metadata, int(row), int(col), rows, cols)
        candidate_id = f"{patch_id}_r{int(row):02d}_c{int(col):02d}"
        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "patch_id": patch_id,
                "npz_path": npz_path,
                "row": int(row),
                "col": int(col),
                "probability": f"{prob:.8f}",
                "candidate_threshold": f"{candidate_threshold:.6f}",
                **bounds,
            }
        )

    return candidate_rows


def score_patches(
    model: tf.keras.Model,
    records: list[dict[str, Any]],
    stats: dict[str, Any],
    threshold: float,
    candidate_threshold: float,
    batch_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score NPZ records and return score rows plus candidate-cell rows."""
    feature_keys = [str(key) for key in stats["feature_keys"]]
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(1, 1, -1)

    score_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    pending_x: list[np.ndarray] = []
    pending_meta: list[dict[str, Any]] = []

    def flush_pending() -> None:
        if not pending_x:
            return
        batch = np.stack(pending_x, axis=0)
        preds = predict_batch(model, batch)
        for meta, pred in zip(pending_meta, preds):
            pred_map = prediction_to_2d_map(pred)
            pred_map = np.nan_to_num(pred_map, nan=0.0, posinf=0.0, neginf=0.0)

            max_prob = float(np.max(pred_map))
            mean_prob = float(np.mean(pred_map))
            argmax_flat = int(np.argmax(pred_map))
            argmax_row, argmax_col = np.unravel_index(argmax_flat, pred_map.shape)

            patch_candidate_rows = build_candidate_rows(
                patch_id=meta["patch_id"],
                npz_path=meta["npz_path"],
                prediction_map=pred_map,
                metadata=meta["metadata"],
                candidate_threshold=candidate_threshold,
            )
            candidate_rows.extend(patch_candidate_rows)

            score_rows.append(
                {
                    "patch_id": meta["patch_id"],
                    "npz_path": meta["npz_path"],
                    "status": "completed",
                    "model_b_max_prob": f"{max_prob:.8f}",
                    "model_b_mean_prob": f"{mean_prob:.8f}",
                    "argmax_row": int(argmax_row),
                    "argmax_col": int(argmax_col),
                    "candidate_cell_count": len(patch_candidate_rows),
                    "passed_gate": int(max_prob >= threshold),
                    "threshold": f"{threshold:.6f}",
                    "candidate_threshold": f"{candidate_threshold:.6f}",
                    "invalid_values_before_cleaning": meta["invalid_values_before_cleaning"],
                    "message": "",
                }
            )
        pending_x.clear()
        pending_meta.clear()

    for record in records:
        patch_id = str(record["patch_id"])
        npz_path = Path(record["npz_path"])
        try:
            X, invalid_count = load_npz_feature_stack(npz_path, feature_keys, mean, std)
            pending_x.append(X)
            pending_meta.append(
                {
                    "patch_id": patch_id,
                    "npz_path": str(npz_path),
                    "metadata": record.get("metadata", {}),
                    "invalid_values_before_cleaning": invalid_count,
                }
            )
            if len(pending_x) >= batch_size:
                flush_pending()
        except Exception as exc:  # noqa: BLE001 - continue over large inference jobs.
            score_rows.append(
                {
                    "patch_id": patch_id,
                    "npz_path": str(npz_path),
                    "status": "failed",
                    "model_b_max_prob": "",
                    "model_b_mean_prob": "",
                    "argmax_row": "",
                    "argmax_col": "",
                    "candidate_cell_count": "",
                    "passed_gate": "",
                    "threshold": f"{threshold:.6f}",
                    "candidate_threshold": f"{candidate_threshold:.6f}",
                    "invalid_values_before_cleaning": "",
                    "message": str(exc),
                }
            )

    flush_pending()
    return score_rows, candidate_rows


def write_score_csv(rows: list[dict[str, Any]], output_csv: str | Path) -> None:
    """Write coarse-patch inference result rows to CSV."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patch_id",
        "npz_path",
        "status",
        "model_b_max_prob",
        "model_b_mean_prob",
        "argmax_row",
        "argmax_col",
        "candidate_cell_count",
        "passed_gate",
        "threshold",
        "candidate_threshold",
        "invalid_values_before_cleaning",
        "message",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_candidate_csv(rows: list[dict[str, Any]], output_csv: str | Path) -> None:
    """Write selected 1 km candidate cells to CSV."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_id",
        "patch_id",
        "npz_path",
        "row",
        "col",
        "probability",
        "candidate_threshold",
        "cell_xmin",
        "cell_ymin",
        "cell_xmax",
        "cell_ymax",
        "coarse_xmin",
        "coarse_ymin",
        "coarse_xmax",
        "coarse_ymax",
        "crs",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(
    score_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    output_csv: str | Path,
    candidate_csv: str | Path,
) -> None:
    """Print compact inference summary."""
    completed = [row for row in score_rows if row["status"] == "completed"]
    failed = [row for row in score_rows if row["status"] != "completed"]
    passed = [row for row in completed if int(row["passed_gate"]) == 1]

    print("Model B geospatial inference complete")
    print(f"Coarse patch rows: {len(score_rows)}")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")
    print(f"Passed coarse patches: {len(passed)}")
    if completed:
        print(f"Coarse pass rate: {len(passed) / len(completed):.4f}")
    print(f"Candidate 1 km cells: {len(candidate_rows)}")
    print(f"Output CSV: {output_csv}")
    print(f"Candidate CSV: {candidate_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Model B geospatial inference on 1 km NPZ patches.")
    parser.add_argument("--model", required=True, help="Path to trained Model B .keras file.")
    parser.add_argument("--channel-stats", required=True, help="1 km channel_stats.json path.")
    parser.add_argument("--patch-dir", default=None, help="Folder containing extracted geospatial NPZ patches.")
    parser.add_argument("--manifest", default=None, help="Extraction manifest CSV with completed npz_path rows.")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Glob pattern when using --patch-dir.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Model B coarse-patch gate threshold.")
    parser.add_argument(
        "--candidate-threshold",
        type=float,
        default=None,
        help="Threshold for writing 1 km candidate cells. Defaults to --threshold.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument(
        "--output-csv",
        default="results/geospatial/model_b_geospatial_scores.csv",
        help="Output CSV path for coarse Model B scores.",
    )
    parser.add_argument(
        "--candidate-csv",
        default=DEFAULT_CANDIDATE_CSV,
        help="Output CSV path for selected 1 km candidate cells.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    candidate_threshold = args.threshold if args.candidate_threshold is None else args.candidate_threshold

    stats = load_channel_stats(args.channel_stats)
    records = collect_npz_records(
        patch_dir=args.patch_dir,
        manifest=args.manifest,
        pattern=args.pattern,
        max_files=args.max_files,
    )

    print("Loading Model B")
    print(f"Model: {args.model}")
    model = tf.keras.models.load_model(args.model, compile=False)

    print("Running geospatial Model B inference")
    print(f"Patches selected: {len(records)}")
    print(f"Channels: {len(stats['feature_keys'])}")
    print(f"Coarse threshold: {args.threshold}")
    print(f"Candidate threshold: {candidate_threshold}")

    score_rows, candidate_rows = score_patches(
        model=model,
        records=records,
        stats=stats,
        threshold=args.threshold,
        candidate_threshold=candidate_threshold,
        batch_size=args.batch_size,
    )
    write_score_csv(score_rows, args.output_csv)
    write_candidate_csv(candidate_rows, args.candidate_csv)
    print_summary(score_rows, candidate_rows, args.output_csv, args.candidate_csv)


if __name__ == "__main__":
    main()
