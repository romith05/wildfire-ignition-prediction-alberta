"""Export visual prediction samples for the 25 m Model A spatial refiner.

The goal is to inspect qualitative behavior before integrating Model A into the
full coarse-to-fine pipeline.

Outputs PNG files grouped into:
- true_positive: label has ignition and prediction catches it
- false_negative: label has ignition and prediction misses it
- false_positive: label has no ignition and prediction activates
- clean_negative: label has no ignition and prediction stays quiet

Each PNG contains three panels:
1. selected feature channel image
2. ground-truth mask
3. predicted probability mask
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
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


CATEGORY_NAMES = ["true_positive", "false_negative", "false_positive", "clean_negative"]


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


def choose_display_channel(feature_keys: list[str], preferred_key: str | None) -> int:
    """Choose a feature channel index for the left visualization panel."""
    if preferred_key is None:
        return 0

    lowered = [key.lower() for key in feature_keys]
    preferred = preferred_key.lower()
    if preferred in lowered:
        return lowered.index(preferred)

    for idx, key in enumerate(lowered):
        if preferred in key:
            return idx

    print(f"Warning: display feature '{preferred_key}' not found. Using channel 0: {feature_keys[0]}")
    return 0


def classify_prediction(label: np.ndarray, pred: np.ndarray, threshold: float) -> str:
    """Classify patch-level prediction outcome."""
    has_label = bool(np.max(label > 0.5))
    has_pred = bool(np.max(pred >= threshold))

    if has_label and has_pred:
        return "true_positive"
    if has_label and not has_pred:
        return "false_negative"
    if not has_label and has_pred:
        return "false_positive"
    return "clean_negative"


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Robustly normalize one feature channel for display."""
    image = np.asarray(image, dtype="float32")
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    lo = float(np.percentile(image, 2))
    hi = float(np.percentile(image, 98))
    if hi <= lo:
        return np.zeros_like(image, dtype="float32")
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def save_sample_png(
    output_path: Path,
    feature_image: np.ndarray,
    label: np.ndarray,
    pred: np.ndarray,
    title: str,
    threshold: float,
) -> None:
    """Save one three-panel visual sample."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    label_2d = np.squeeze(label)
    pred_2d = np.squeeze(pred)
    pred_bin = pred_2d >= threshold

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    axes[0].imshow(normalize_for_display(feature_image))
    axes[0].set_title("feature")
    axes[0].axis("off")

    axes[1].imshow(label_2d, vmin=0, vmax=1)
    axes[1].set_title("ground truth")
    axes[1].axis("off")

    axes[2].imshow(pred_2d, vmin=0, vmax=1)
    axes[2].set_title("probability")
    axes[2].axis("off")

    axes[3].imshow(pred_bin.astype("float32"), vmin=0, vmax=1)
    axes[3].set_title(f"pred >= {threshold:.2f}")
    axes[3].axis("off")

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def export_samples(args: argparse.Namespace) -> None:
    """Export categorized visual samples."""
    model = load_model(args.model, compile=False, custom_objects=CUSTOM_OBJECTS)
    channel_stats = load_channel_stats(args.channel_stats)
    files = find_patch_files(args.data_dir, args.pattern)

    if args.max_files is not None and args.max_files > 0:
        files = files[: args.max_files]

    output_dir = Path(args.output_dir)
    for category in CATEGORY_NAMES:
        (output_dir / category).mkdir(parents=True, exist_ok=True)

    counts = {category: 0 for category in CATEGORY_NAMES}
    skipped = 0

    for idx, file_path in enumerate(files, start=1):
        if all(counts[category] >= args.samples_per_category for category in CATEGORY_NAMES):
            break

        try:
            patch, label, feature_keys = load_npz_patch(
                file_path,
                label_key=args.label_key,
                channel_stats=channel_stats,
            )
            if label is None:
                raise ValueError("Missing label array")

            patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
            label = np.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

            pred = model.predict(patch[np.newaxis, ...], verbose=0)[0]
            pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

            category = classify_prediction(label, pred, args.threshold)
            if counts[category] >= args.samples_per_category:
                continue

            channel_idx = choose_display_channel(feature_keys, args.display_feature)
            feature_image = patch[..., channel_idx]

            pred_max = float(np.max(pred))
            label_pixels = int((label > 0.5).sum())
            title = (
                f"{category} | {file_path.name} | "
                f"label_pixels={label_pixels} | pred_max={pred_max:.4f}"
            )

            output_path = output_dir / category / f"{counts[category]:03d}_{file_path.stem}.png"
            save_sample_png(
                output_path=output_path,
                feature_image=feature_image,
                label=label,
                pred=pred,
                title=title,
                threshold=args.threshold,
            )
            counts[category] += 1

        except Exception as exc:
            skipped += 1
            print(f"Skipping {file_path}: {exc}")

        if idx % args.log_every == 0:
            print(f"Processed {idx}/{len(files)} | counts={counts} | skipped={skipped}")

    print("\nExport complete")
    print(f"Output dir: {output_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Counts: {counts}")
    print(f"Skipped: {skipped}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export visual samples for 25 m Model A predictions.")
    parser.add_argument("--model", required=True, help="Saved Model A .keras file.")
    parser.add_argument("--data-dir", required=True, help="25 m validation/test patch folder.")
    parser.add_argument("--output-dir", default="results/model_a_samples", help="Output folder for PNG samples.")
    parser.add_argument("--channel-stats", default=None, help="25 m channel stats JSON.")
    parser.add_argument("--threshold", type=float, default=0.50, help="Prediction threshold for categorizing patches.")
    parser.add_argument("--pattern", default="patch_*.npz", help="Patch glob pattern.")
    parser.add_argument("--label-key", default="class", help="NPZ label/mask key.")
    parser.add_argument("--display-feature", default=None, help="Feature key or substring to display in the first panel.")
    parser.add_argument("--samples-per-category", type=int, default=12, help="Number of PNGs per outcome category.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional max files to scan.")
    parser.add_argument("--log-every", type=int, default=250, help="Progress logging interval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_samples(args)


if __name__ == "__main__":
    main()
