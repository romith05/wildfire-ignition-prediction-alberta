"""Diagnose 25 m Model A validation behavior.

This script is intended for debugging the spatial-refiner validation issue where
training metrics improve but validation precision/recall stay near 0.5.

It reports:
- patch counts and positive/negative split,
- label mask statistics,
- feature invalid-value counts,
- prediction max/mean probability distributions,
- pixel-level threshold sweep metrics.
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
    """Find patch files."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found in {data_dir} matching {pattern}")
    return files


def parse_thresholds(raw: str) -> list[float]:
    """Parse comma-separated threshold string."""
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def update_confusion(stats: dict, y_true: np.ndarray, y_pred_prob: np.ndarray, thresholds: list[float]) -> None:
    """Accumulate pixel-level confusion counts for each threshold."""
    true_bin = y_true > 0.5

    for threshold in thresholds:
        pred_bin = y_pred_prob >= threshold
        key = f"{threshold:.4f}"

        stats[key]["tp"] += int(np.logical_and(true_bin, pred_bin).sum())
        stats[key]["fp"] += int(np.logical_and(~true_bin, pred_bin).sum())
        stats[key]["tn"] += int(np.logical_and(~true_bin, ~pred_bin).sum())
        stats[key]["fn"] += int(np.logical_and(true_bin, ~pred_bin).sum())


def summarize_confusion(counts: dict) -> dict:
    """Convert confusion counts into metrics."""
    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "fp_rate": fp_rate,
        "iou": iou,
        "dice": dice,
    }


def percentile_summary(values: list[float]) -> dict:
    """Summarize a list of scalar values."""
    if not values:
        return {"min": 0, "p25": 0, "median": 0, "p75": 0, "max": 0, "mean": 0}

    arr = np.asarray(values, dtype="float64")
    return {
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def diagnose(args: argparse.Namespace) -> None:
    """Run full validation diagnosis."""
    model = load_model(args.model, compile=False, custom_objects=CUSTOM_OBJECTS)
    channel_stats = load_channel_stats(args.channel_stats)
    files = find_patch_files(args.data_dir, args.pattern)

    if args.max_files is not None and args.max_files > 0:
        files = files[: args.max_files]

    thresholds = parse_thresholds(args.thresholds)
    confusion = {
        f"{threshold:.4f}": {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        for threshold in thresholds
    }

    patch_rows = []
    positive_patches = 0
    negative_patches = 0
    total_positive_pixels = 0
    total_pixels = 0
    invalid_feature_values = 0
    invalid_label_values = 0
    pred_max_values = []
    pred_mean_values = []
    label_positive_pixel_counts = []
    skipped = 0

    for idx, file_path in enumerate(files, start=1):
        try:
            patch, label, feature_keys = load_npz_patch(
                file_path,
                label_key=args.label_key,
                channel_stats=channel_stats,
            )

            invalid_feature_values += int((~np.isfinite(patch)).sum())
            if label is None:
                raise ValueError("Missing label array")
            invalid_label_values += int((~np.isfinite(label)).sum())

            patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
            label = np.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

            pred = model.predict(patch[np.newaxis, ...], verbose=0)[0]
            pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

            positive_pixels = int((label > 0.5).sum())
            patch_pixels = int(label.size)
            is_positive_patch = positive_pixels > 0

            if is_positive_patch:
                positive_patches += 1
            else:
                negative_patches += 1

            total_positive_pixels += positive_pixels
            total_pixels += patch_pixels

            pred_max = float(np.max(pred))
            pred_mean = float(np.mean(pred))
            pred_max_values.append(pred_max)
            pred_mean_values.append(pred_mean)
            label_positive_pixel_counts.append(float(positive_pixels))

            update_confusion(confusion, label, pred, thresholds)

            patch_rows.append(
                {
                    "path": str(file_path),
                    "positive_pixels": positive_pixels,
                    "is_positive_patch": int(is_positive_patch),
                    "pred_max": f"{pred_max:.8f}",
                    "pred_mean": f"{pred_mean:.8f}",
                }
            )

        except Exception as exc:
            skipped += 1
            print(f"Skipping {file_path}: {exc}")

        if idx % args.log_every == 0:
            print(f"Processed {idx}/{len(files)} | skipped={skipped}")

    print("\nValidation dataset summary")
    print(f"files_seen: {len(files)}")
    print(f"usable_patches: {len(patch_rows)}")
    print(f"skipped: {skipped}")
    print(f"positive_patches: {positive_patches}")
    print(f"negative_patches: {negative_patches}")
    print(f"total_pixels: {total_pixels}")
    print(f"total_positive_pixels: {total_positive_pixels}")
    print(f"positive_pixel_fraction: {total_positive_pixels / total_pixels if total_pixels else 0:.8f}")
    print(f"invalid_feature_values_before_cleaning: {invalid_feature_values}")
    print(f"invalid_label_values_before_cleaning: {invalid_label_values}")

    print("\nPrediction max probability summary")
    for key, value in percentile_summary(pred_max_values).items():
        print(f"pred_max_{key}: {value:.8f}")

    print("\nPrediction mean probability summary")
    for key, value in percentile_summary(pred_mean_values).items():
        print(f"pred_mean_{key}: {value:.8f}")

    print("\nPositive pixels per patch summary")
    for key, value in percentile_summary(label_positive_pixel_counts).items():
        print(f"positive_pixels_{key}: {value:.4f}")

    print("\nPixel threshold sweep")
    print("threshold | precision | recall | fp_rate | dice | iou | tp | fp | fn")
    metric_rows = []
    for threshold in thresholds:
        key = f"{threshold:.4f}"
        metrics = summarize_confusion(confusion[key])
        metric_row = {"threshold": threshold, **metrics}
        metric_rows.append(metric_row)
        print(
            f"{threshold:.2f} | "
            f"{metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | "
            f"{metrics['fp_rate']:.8f} | "
            f"{metrics['dice']:.4f} | "
            f"{metrics['iou']:.4f} | "
            f"{metrics['tp']} | {metrics['fp']} | {metrics['fn']}"
        )

    if args.output_patch_csv:
        output_patch_csv = Path(args.output_patch_csv)
        output_patch_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_patch_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["path", "positive_pixels", "is_positive_patch", "pred_max", "pred_mean"],
            )
            writer.writeheader()
            writer.writerows(patch_rows)
        print(f"\nSaved patch diagnostics to: {output_patch_csv}")

    if args.output_threshold_csv:
        output_threshold_csv = Path(args.output_threshold_csv)
        output_threshold_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_threshold_csv.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "threshold",
                "precision",
                "recall",
                "fp_rate",
                "iou",
                "dice",
                "tp",
                "fp",
                "tn",
                "fn",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metric_rows)
        print(f"Saved threshold diagnostics to: {output_threshold_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose 25 m Model A validation behavior.")
    parser.add_argument("--model", required=True, help="Saved Model A .keras file.")
    parser.add_argument("--data-dir", required=True, help="25 m validation patch folder.")
    parser.add_argument("--channel-stats", default=None, help="25 m channel stats JSON.")
    parser.add_argument("--pattern", default="patch_*.npz", help="Patch file glob pattern.")
    parser.add_argument("--label-key", default="class", help="NPZ label/mask key.")
    parser.add_argument(
        "--thresholds",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50",
        help="Comma-separated pixel thresholds.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Optional max files for quick diagnostics.")
    parser.add_argument("--log-every", type=int, default=250, help="Progress logging interval.")
    parser.add_argument("--output-patch-csv", default=None, help="Optional per-patch diagnostics CSV.")
    parser.add_argument("--output-threshold-csv", default=None, help="Optional threshold metrics CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnose(args)


if __name__ == "__main__":
    main()
