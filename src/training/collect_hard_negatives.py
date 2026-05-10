"""Collect hard-negative patches for Model B training.

Hard negatives are no-ignition patches that Model B scores as likely ignition.
This utility scans negative patch files, runs Model B, and copies high-confidence
false positives into a separate folder for later retraining/fine-tuning.

Expected filename convention:
- positive patches: *_ignition_*
- negative patches: *_no_ignition_*

Only *_no_ignition_* files are scanned by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from src.data.npz_loader import load_npz_patch
from src.training.losses import combined_loss, focal_loss, patch_level_fp_loss
from src.training.metrics import pixel_fp_rate, pixel_precision, pixel_recall, patch_fp_rate


CUSTOM_OBJECTS = {
    "focal_loss": focal_loss,
    "combined_loss": combined_loss,
    "patch_level_fp_loss": patch_level_fp_loss,
    "pixel_precision": pixel_precision,
    "pixel_recall": pixel_recall,
    "pixel_fp_rate": pixel_fp_rate,
    "patch_fp_rate": patch_fp_rate,
}


def is_no_ignition_patch(path: str | Path, negative_token: str = "_no_ignition_") -> bool:
    """Return True only for filename-marked no-ignition patches."""
    name = Path(path).name.lower()
    return negative_token.lower() in name


def find_negative_patches(data_dir: str | Path, pattern: str, negative_token: str) -> list[Path]:
    """Find sorted no-ignition patches under a folder."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob(pattern))
    negative_files = [path for path in files if is_no_ignition_patch(path, negative_token)]

    if not files:
        raise FileNotFoundError(f"No files matched {pattern} in {data_dir}")
    if not negative_files:
        raise FileNotFoundError(
            f"No no-ignition files found in {data_dir}. Expected filenames containing {negative_token}"
        )

    return negative_files


def load_channel_stats(path: str | None):
    """Load optional channel normalization stats."""
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def score_patch(model, patch: np.ndarray) -> float:
    """Return max Model B probability for one patch."""
    patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    pred = model.predict(patch[np.newaxis, ...], verbose=0)[0]
    return float(np.nanmax(pred))


def write_manifest_row(writer, source_path: Path, output_path: Path, score: float, copied: bool) -> None:
    """Write one hard-negative scoring record."""
    writer.writerow(
        {
            "source_path": str(source_path),
            "output_path": str(output_path) if copied else "",
            "score": f"{score:.8f}",
            "copied": int(copied),
        }
    )


def collect_hard_negatives(args: argparse.Namespace) -> None:
    """Score negative patches and copy high-confidence false positives."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    channel_stats = load_channel_stats(args.channel_stats)
    model = load_model(args.model, compile=False, custom_objects=CUSTOM_OBJECTS)

    negative_files = find_negative_patches(
        data_dir=args.data_dir,
        pattern=args.pattern,
        negative_token=args.negative_token,
    )

    if args.max_files is not None and args.max_files > 0:
        negative_files = negative_files[: args.max_files]

    print(f"Negative patches found: {len(negative_files)}")
    print(f"Threshold: {args.threshold}")
    print(f"Output dir: {output_dir}")
    print(f"Manifest: {manifest_path}")

    copied_count = 0
    scored_count = 0

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_path", "output_path", "score", "copied"],
        )
        writer.writeheader()

        for source_path in negative_files:
            try:
                patch, _, _ = load_npz_patch(
                    source_path,
                    label_key=args.label_key,
                    channel_stats=channel_stats,
                )
                score = score_patch(model, patch)
                scored_count += 1

                copied = score >= args.threshold
                output_path = output_dir / source_path.name
                if copied:
                    shutil.copy2(source_path, output_path)
                    copied_count += 1

                write_manifest_row(writer, source_path, output_path, score, copied)

                if scored_count % args.log_every == 0:
                    print(f"Scored {scored_count}/{len(negative_files)} | copied {copied_count}")

            except Exception as exc:
                print(f"Skipping {source_path}: {exc}")

    print("Done collecting hard negatives")
    print(f"Scored patches: {scored_count}")
    print(f"Hard negatives copied: {copied_count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect no-ignition patches that fool Model B.")
    parser.add_argument("--model", required=True, help="Trained Model B .keras file.")
    parser.add_argument("--data-dir", required=True, help="Folder containing balanced patch_*.npz files.")
    parser.add_argument("--output-dir", required=True, help="Folder where hard-negative .npz files will be copied.")
    parser.add_argument(
        "--manifest",
        default="results/hard_negatives_manifest.csv",
        help="CSV manifest path for all scanned negative patches and scores.",
    )
    parser.add_argument("--channel-stats", default=None, help="Channel stats JSON used during Model B training.")
    parser.add_argument("--threshold", type=float, default=0.25, help="Copy patches with max probability >= threshold.")
    parser.add_argument("--pattern", default="patch_*.npz", help="Patch file glob pattern.")
    parser.add_argument("--negative-token", default="_no_ignition_", help="Filename token identifying negative patches.")
    parser.add_argument("--label-key", default="class", help="NPZ label/mask key.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional maximum number of negative files to scan.")
    parser.add_argument("--log-every", type=int, default=500, help="Progress logging interval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect_hard_negatives(args)


if __name__ == "__main__":
    main()
