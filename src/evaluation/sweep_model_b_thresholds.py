"""Sweep Model B gate thresholds on validation patches.

This evaluator compares saved Model B phase checkpoints by threshold. It scores
patch-level gate behavior, not pixel segmentation quality.

Expected filename convention:
- positive ignition patches: *_ignition_* but not *_no_ignition_*
- negative patches: *_no_ignition_*
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


def is_negative_patch(path: str | Path, negative_token: str = "_no_ignition_") -> bool:
    """Return True for filename-marked no-ignition patches."""
    return negative_token.lower() in Path(path).name.lower()


def is_positive_patch(path: str | Path) -> bool:
    """Return True for positive ignition patches using the project filename convention."""
    name = Path(path).name.lower()
    return "_ignition_" in name and "_no_ignition_" not in name


def find_patch_files(data_dir: str | Path, pattern: str) -> list[Path]:
    """Find patch files to evaluate."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No patch files found in {data_dir} matching {pattern}")
    return files


def load_channel_stats(path: str | None):
    """Load optional channel stats."""
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_patch(model, patch: np.ndarray) -> float:
    """Return max predicted probability for one patch."""
    patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    pred = model.predict(patch[np.newaxis, ...], verbose=0)[0]
    return float(np.nanmax(pred))


def score_dataset(args: argparse.Namespace) -> list[dict]:
    """Score every validation patch once."""
    model = load_model(args.model, compile=False, custom_objects=CUSTOM_OBJECTS)
    channel_stats = load_channel_stats(args.channel_stats)
    files = find_patch_files(args.data_dir, args.pattern)

    if args.max_files is not None and args.max_files > 0:
        files = files[: args.max_files]

    rows = []
    skipped = 0

    for idx, file_path in enumerate(files, start=1):
        positive = is_positive_patch(file_path)
        negative = is_negative_patch(file_path, args.negative_token)

        if args.filename_labels and not positive and not negative:
            skipped += 1
            continue

        try:
            patch, label, _ = load_npz_patch(
                file_path,
                label_key=args.label_key,
                channel_stats=channel_stats,
            )
            score = score_patch(model, patch)

            if args.filename_labels:
                y_true = int(positive)
            else:
                if label is None:
                    raise ValueError("Patch has no label and --filename-labels is disabled")
                y_true = int(np.nanmax(label) > 0)

            rows.append({"path": str(file_path), "y_true": y_true, "score": score})

        except Exception as exc:
            skipped += 1
            print(f"Skipping {file_path}: {exc}")

        if idx % args.log_every == 0:
            print(f"Scored {idx}/{len(files)} files | usable={len(rows)} | skipped={skipped}")

    print(f"Finished scoring. Usable={len(rows)} | skipped={skipped}")
    return rows


def compute_threshold_metrics(rows: list[dict], threshold: float) -> dict:
    """Compute patch-level metrics at one threshold."""
    tp = fp = tn = fn = 0

    for row in rows:
        y_true = int(row["y_true"])
        y_pred = int(float(row["score"]) >= threshold)

        if y_true == 1 and y_pred == 1:
            tp += 1
        elif y_true == 1 and y_pred == 0:
            fn += 1
        elif y_true == 0 and y_pred == 1:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    passed = tp + fp

    patch_recall = tp / (tp + fn) if (tp + fn) else 0.0
    patch_precision = tp / (tp + fp) if (tp + fp) else 0.0
    patch_fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
    pass_rate = passed / total if total else 0.0

    return {
        "threshold": threshold,
        "total_patches": total,
        "positive_patches": tp + fn,
        "negative_patches": fp + tn,
        "passed_patches": passed,
        "pass_rate": pass_rate,
        "patch_recall": patch_recall,
        "patch_precision": patch_precision,
        "patch_fp_rate": patch_fp_rate,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def parse_thresholds(raw: str) -> list[float]:
    """Parse comma-separated thresholds."""
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def write_results(output_path: str | Path, metrics: list[dict]) -> None:
    """Write threshold sweep CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "threshold",
        "total_patches",
        "positive_patches",
        "negative_patches",
        "passed_patches",
        "pass_rate",
        "patch_recall",
        "patch_precision",
        "patch_fp_rate",
        "tp",
        "fp",
        "tn",
        "fn",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def print_summary(metrics: list[dict]) -> None:
    """Print compact threshold summary."""
    print("\nThreshold sweep summary")
    print("threshold | recall | fp_rate | precision | pass_rate | passed")
    for row in metrics:
        print(
            f"{row['threshold']:.2f} | "
            f"{row['patch_recall']:.4f} | "
            f"{row['patch_fp_rate']:.4f} | "
            f"{row['patch_precision']:.4f} | "
            f"{row['pass_rate']:.4f} | "
            f"{row['passed_patches']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Model B patch-gate thresholds.")
    parser.add_argument("--model", required=True, help="Saved Model B .keras file.")
    parser.add_argument("--data-dir", required=True, help="Validation patch folder.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--channel-stats", default=None, help="Channel stats JSON used during training.")
    parser.add_argument("--pattern", default="patch_*.npz", help="Patch filename glob pattern.")
    parser.add_argument("--label-key", default="class", help="NPZ label/mask key.")
    parser.add_argument("--negative-token", default="_no_ignition_", help="Filename token for negative patches.")
    parser.add_argument(
        "--thresholds",
        default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50",
        help="Comma-separated thresholds to evaluate.",
    )
    parser.add_argument(
        "--filename-labels",
        action="store_true",
        default=True,
        help="Use filename convention for patch labels. Default: enabled.",
    )
    parser.add_argument(
        "--mask-labels",
        dest="filename_labels",
        action="store_false",
        help="Use NPZ class mask to determine patch labels instead of filenames.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Optional max files for quick tests.")
    parser.add_argument("--log-every", type=int, default=500, help="Progress logging interval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = score_dataset(args)
    thresholds = parse_thresholds(args.thresholds)
    metrics = [compute_threshold_metrics(rows, threshold) for threshold in thresholds]
    write_results(args.output, metrics)
    print_summary(metrics)
    print(f"\nSaved threshold sweep to: {args.output}")


if __name__ == "__main__":
    main()
