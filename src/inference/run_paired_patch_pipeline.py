"""Run a test-only paired 1 km -> 25 m inference pipeline.

This script is for controlled validation only.

It assumes that a 1 km patch and its corresponding 25 m patch have the same
filename and live in separate folders. That is useful for testing model handoff,
but it is not the final geospatial prototype.

Final prototype requirement:
- use the 1 km patch geospatial footprint,
- generate or retrieve all corresponding 25 m patches inside that footprint,
- run Model A on those fine patches,
- stitch predictions back into a geospatial output.

Current test flow:
1. scan 1 km patches with Model B,
2. if Model B max probability >= threshold, find same filename in 25 m folder,
3. run Model A on the matched 25 m patch,
4. call the patch final-positive only if Model A predicts enough positive pixels,
5. write per-patch CSV results.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from src.data.npz_loader import load_npz_patch
from src.training.losses import combined_loss, focal_loss, patch_level_fp_loss
from src.training.metrics import patch_fp_rate, pixel_fp_rate, pixel_precision, pixel_recall


CUSTOM_OBJECTS = {
    "focal_loss": focal_loss,
    "combined_loss": combined_loss,
    "patch_level_fp_loss": patch_level_fp_loss,
    "pixel_precision": pixel_precision,
    "pixel_recall": pixel_recall,
    "pixel_fp_rate": pixel_fp_rate,
    "patch_fp_rate": patch_fp_rate,
}


def load_channel_stats(path: str | None):
    """Load optional channel stats JSON."""
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_patch_files(data_dir: str | Path, pattern: str) -> list[Path]:
    """Find sorted patch files."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir} matching {pattern}")
    return files


def build_filename_index(data_dir: str | Path, pattern: str) -> dict[str, Path]:
    """Build filename -> path index for paired 25 m patches."""
    files = find_patch_files(data_dir, pattern)
    return {path.name: path for path in files}


def predict_max(model, patch: np.ndarray) -> tuple[float, np.ndarray]:
    """Return max probability and full probability mask for one patch."""
    patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    pred = model.predict(patch[np.newaxis, ...], verbose=0)[0]
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    return float(np.max(pred)), pred


def label_has_positive(label: np.ndarray | None) -> bool | None:
    """Return patch-level label truth if label exists."""
    if label is None:
        return None
    return bool(np.max(label > 0.5))


def count_positive_pixels(mask: np.ndarray, threshold: float) -> int:
    """Count positive pixels at a threshold."""
    return int((mask >= threshold).sum())


def run_pipeline(args: argparse.Namespace) -> None:
    """Run paired patch inference."""
    model_b = load_model(args.model_b, compile=False, custom_objects=CUSTOM_OBJECTS)
    model_a = load_model(args.model_a, compile=False, custom_objects=CUSTOM_OBJECTS)

    stats_1km = load_channel_stats(args.channel_stats_1km)
    stats_25m = load_channel_stats(args.channel_stats_25m)

    one_km_files = find_patch_files(args.patches_1km_dir, args.pattern)
    paired_25m_index = build_filename_index(args.patches_25m_dir, args.pattern)

    if args.max_files is not None and args.max_files > 0:
        one_km_files = one_km_files[: args.max_files]

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scanned = 0
    passed_model_b = 0
    missing_25m = 0
    ran_model_a = 0
    final_positive = 0

    fieldnames = [
        "filename",
        "path_1km",
        "path_25m",
        "model_b_max_prob",
        "model_b_passed",
        "model_a_max_prob",
        "model_a_positive_pixels",
        "model_a_min_positive_pixels",
        "final_positive",
        "label_1km_positive",
        "label_25m_positive",
        "status",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, path_1km in enumerate(one_km_files, start=1):
            scanned += 1
            row = {
                "filename": path_1km.name,
                "path_1km": str(path_1km),
                "path_25m": "",
                "model_b_max_prob": "",
                "model_b_passed": 0,
                "model_a_max_prob": "",
                "model_a_positive_pixels": "",
                "model_a_min_positive_pixels": args.model_a_min_positive_pixels,
                "final_positive": 0,
                "label_1km_positive": "",
                "label_25m_positive": "",
                "status": "",
            }

            try:
                patch_1km, label_1km, _ = load_npz_patch(
                    path_1km,
                    label_key=args.label_key,
                    channel_stats=stats_1km,
                )
                row["label_1km_positive"] = label_has_positive(label_1km)

                model_b_max, _ = predict_max(model_b, patch_1km)
                model_b_passed = model_b_max >= args.model_b_threshold
                row["model_b_max_prob"] = f"{model_b_max:.8f}"
                row["model_b_passed"] = int(model_b_passed)

                if not model_b_passed:
                    row["status"] = "blocked_by_model_b"
                    writer.writerow(row)
                    continue

                passed_model_b += 1
                path_25m = paired_25m_index.get(path_1km.name)
                if path_25m is None:
                    missing_25m += 1
                    row["status"] = "missing_matching_25m_patch"
                    writer.writerow(row)
                    continue

                row["path_25m"] = str(path_25m)

                patch_25m, label_25m, _ = load_npz_patch(
                    path_25m,
                    label_key=args.label_key,
                    channel_stats=stats_25m,
                )
                row["label_25m_positive"] = label_has_positive(label_25m)

                model_a_max, pred_25m = predict_max(model_a, patch_25m)
                model_a_positive_pixels = count_positive_pixels(pred_25m, args.model_a_threshold)
                is_final_positive = model_a_positive_pixels >= args.model_a_min_positive_pixels

                row["model_a_max_prob"] = f"{model_a_max:.8f}"
                row["model_a_positive_pixels"] = model_a_positive_pixels
                row["final_positive"] = int(is_final_positive)
                row["status"] = "completed"

                ran_model_a += 1
                if is_final_positive:
                    final_positive += 1

                writer.writerow(row)

            except Exception as exc:
                row["status"] = f"error: {exc}"
                writer.writerow(row)

            if idx % args.log_every == 0:
                print(
                    f"Processed {idx}/{len(one_km_files)} | "
                    f"B_passed={passed_model_b} | A_ran={ran_model_a} | "
                    f"final_positive={final_positive} | missing_25m={missing_25m}"
                )

    print("\nPaired patch pipeline complete")
    print(f"1 km patches scanned: {scanned}")
    print(f"passed Model B: {passed_model_b}")
    print(f"missing paired 25 m patches: {missing_25m}")
    print(f"ran Model A: {ran_model_a}")
    print(f"Model A threshold: {args.model_a_threshold}")
    print(f"Model A min positive pixels: {args.model_a_min_positive_pixels}")
    print(f"final positive patches: {final_positive}")
    print(f"output CSV: {output_path}")
    print("\nReminder: this is a test-only filename-paired pipeline, not the final geospatial prototype.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run test-only paired 1 km -> 25 m patch inference.")
    parser.add_argument("--model-b", required=True, help="Saved Model B gatekeeper .keras file.")
    parser.add_argument("--model-a", required=True, help="Saved Model A spatial-refiner .keras file.")
    parser.add_argument("--patches-1km-dir", required=True, help="Folder containing 1 km patches.")
    parser.add_argument("--patches-25m-dir", required=True, help="Folder containing paired 25 m patches with same filenames.")
    parser.add_argument("--channel-stats-1km", default=None, help="1 km channel stats JSON.")
    parser.add_argument("--channel-stats-25m", default=None, help="25 m channel stats JSON.")
    parser.add_argument("--output-csv", default="results/paired_patch_pipeline_results.csv")
    parser.add_argument("--model-b-threshold", type=float, default=0.40)
    parser.add_argument("--model-a-threshold", type=float, default=0.50)
    parser.add_argument(
        "--model-a-min-positive-pixels",
        type=int,
        default=1,
        help="Minimum number of Model A pixels >= threshold required for final_positive.",
    )
    parser.add_argument("--pattern", default="patch_*.npz", help="Patch file glob pattern.")
    parser.add_argument("--label-key", default="class", help="NPZ label/mask key.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional max 1 km files to scan.")
    parser.add_argument("--log-every", type=int, default=250, help="Progress logging interval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_a_min_positive_pixels < 1:
        raise ValueError("--model-a-min-positive-pixels must be >= 1")
    run_pipeline(args)


if __name__ == "__main__":
    main()
