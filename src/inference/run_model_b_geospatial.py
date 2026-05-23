"""Run Model B gatekeeper inference on geospatial 1 km NPZ patches.

This script scores coarse Alberta inference patches created by
``src.geospatial.extract_model_b_patch``. It is designed for unlabeled
geospatial inference patches, not training/evaluation folders.

Default operational setting:
- Model B: models/model_B_1km_gatekeeper_hardneg_phase2.keras
- threshold: 0.30

Examples:
    # Score patches listed in an extraction manifest.
    python -m src.inference.run_model_b_geospatial \
      --model models/model_B_1km_gatekeeper_hardneg_phase2.keras \
      --channel-stats /mnt/work/wildfire/1km/patches_1km_balanced/channel_stats.json \
      --manifest results/geospatial/model_b_patch_extraction_manifest.csv \
      --threshold 0.30 \
      --output-csv results/geospatial/model_b_geospatial_scores.csv

    # Score all NPZ files in a folder.
    python -m src.inference.run_model_b_geospatial \
      --model models/model_B_1km_gatekeeper_hardneg_phase2.keras \
      --channel-stats /mnt/work/wildfire/1km/patches_1km_balanced/channel_stats.json \
      --patch-dir data/patches/1km \
      --threshold 0.30 \
      --output-csv results/geospatial/model_b_geospatial_scores.csv
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


def read_manifest_npz_paths(manifest_path: str | Path) -> list[Path]:
    """Read completed NPZ paths from extraction manifest CSV."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    paths: list[Path] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "npz_path" not in (reader.fieldnames or []):
            raise ValueError(f"Manifest must contain 'npz_path' column: {manifest_path}")
        for row in reader:
            status = row.get("status", "completed")
            npz_path = row.get("npz_path", "")
            if status == "completed" and npz_path:
                paths.append(Path(npz_path))

    if not paths:
        raise ValueError(f"No completed NPZ paths found in manifest: {manifest_path}")
    return paths


def collect_npz_paths(
    patch_dir: str | Path | None,
    manifest: str | Path | None,
    pattern: str,
    max_files: int | None,
) -> list[Path]:
    """Collect NPZ paths from either a manifest or a folder."""
    if manifest and patch_dir:
        raise ValueError("Use either --manifest or --patch-dir, not both.")
    if not manifest and not patch_dir:
        raise ValueError("Provide either --manifest or --patch-dir.")

    if manifest:
        paths = read_manifest_npz_paths(manifest)
    else:
        patch_dir = Path(patch_dir)  # type: ignore[arg-type]
        if not patch_dir.exists():
            raise FileNotFoundError(f"Patch dir not found: {patch_dir}")
        paths = sorted(patch_dir.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No NPZ files found in {patch_dir} using pattern {pattern}")

    if max_files is not None:
        paths = paths[: max(0, max_files)]
    if not paths:
        raise ValueError("No NPZ files selected for inference.")
    return paths


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


def score_patches(
    model: tf.keras.Model,
    npz_paths: list[Path],
    stats: dict[str, Any],
    threshold: float,
    batch_size: int,
) -> list[dict[str, Any]]:
    """Score NPZ paths and return CSV-ready result rows."""
    feature_keys = [str(key) for key in stats["feature_keys"]]
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(1, 1, -1)

    rows: list[dict[str, Any]] = []
    pending_x: list[np.ndarray] = []
    pending_meta: list[dict[str, Any]] = []

    def flush_pending() -> None:
        if not pending_x:
            return
        batch = np.stack(pending_x, axis=0)
        preds = predict_batch(model, batch)
        for meta, pred in zip(pending_meta, preds):
            pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            max_prob = float(np.max(pred))
            mean_prob = float(np.mean(pred))
            rows.append(
                {
                    "patch_id": meta["patch_id"],
                    "npz_path": meta["npz_path"],
                    "status": "completed",
                    "model_b_max_prob": f"{max_prob:.8f}",
                    "model_b_mean_prob": f"{mean_prob:.8f}",
                    "passed_gate": int(max_prob >= threshold),
                    "threshold": f"{threshold:.6f}",
                    "invalid_values_before_cleaning": meta["invalid_values_before_cleaning"],
                    "message": "",
                }
            )
        pending_x.clear()
        pending_meta.clear()

    for npz_path in npz_paths:
        patch_id = npz_path.stem
        try:
            X, invalid_count = load_npz_feature_stack(npz_path, feature_keys, mean, std)
            pending_x.append(X)
            pending_meta.append(
                {
                    "patch_id": patch_id,
                    "npz_path": str(npz_path),
                    "invalid_values_before_cleaning": invalid_count,
                }
            )
            if len(pending_x) >= batch_size:
                flush_pending()
        except Exception as exc:  # noqa: BLE001 - continue over large inference jobs.
            rows.append(
                {
                    "patch_id": patch_id,
                    "npz_path": str(npz_path),
                    "status": "failed",
                    "model_b_max_prob": "",
                    "model_b_mean_prob": "",
                    "passed_gate": "",
                    "threshold": f"{threshold:.6f}",
                    "invalid_values_before_cleaning": "",
                    "message": str(exc),
                }
            )

    flush_pending()
    return rows


def write_csv(rows: list[dict[str, Any]], output_csv: str | Path) -> None:
    """Write inference result rows to CSV."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patch_id",
        "npz_path",
        "status",
        "model_b_max_prob",
        "model_b_mean_prob",
        "passed_gate",
        "threshold",
        "invalid_values_before_cleaning",
        "message",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]], output_csv: str | Path) -> None:
    """Print compact inference summary."""
    completed = [row for row in rows if row["status"] == "completed"]
    failed = [row for row in rows if row["status"] != "completed"]
    passed = [row for row in completed if int(row["passed_gate"]) == 1]

    print("Model B geospatial inference complete")
    print(f"Rows: {len(rows)}")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")
    print(f"Passed gate: {len(passed)}")
    if completed:
        print(f"Pass rate: {len(passed) / len(completed):.4f}")
    print(f"Output CSV: {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Model B geospatial inference on 1 km NPZ patches.")
    parser.add_argument("--model", required=True, help="Path to trained Model B .keras file.")
    parser.add_argument("--channel-stats", required=True, help="1 km channel_stats.json path.")
    parser.add_argument("--patch-dir", default=None, help="Folder containing extracted geospatial NPZ patches.")
    parser.add_argument("--manifest", default=None, help="Extraction manifest CSV with completed npz_path rows.")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Glob pattern when using --patch-dir.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Model B gate threshold.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument(
        "--output-csv",
        default="results/geospatial/model_b_geospatial_scores.csv",
        help="Output CSV path for Model B scores.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    stats = load_channel_stats(args.channel_stats)
    npz_paths = collect_npz_paths(
        patch_dir=args.patch_dir,
        manifest=args.manifest,
        pattern=args.pattern,
        max_files=args.max_files,
    )

    print("Loading Model B")
    print(f"Model: {args.model}")
    model = tf.keras.models.load_model(args.model, compile=False)

    print("Running geospatial Model B inference")
    print(f"Patches selected: {len(npz_paths)}")
    print(f"Channels: {len(stats['feature_keys'])}")
    print(f"Threshold: {args.threshold}")

    rows = score_patches(
        model=model,
        npz_paths=npz_paths,
        stats=stats,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )
    write_csv(rows, args.output_csv)
    print_summary(rows, args.output_csv)


if __name__ == "__main__":
    main()
