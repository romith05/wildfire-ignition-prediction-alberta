"""Compute normalization statistics for 1 km wildfire NPZ patches.

This is a resolution-specific wrapper around ``src.data.compute_channel_stats``.
It exists to prevent mixing 1 km Model B/base-model normalization with 25 m
Model A/refiner normalization.

Use this for:
- 1 km ignition-only base Model A training
- 1 km mixed/balanced Model B gatekeeper training

Do not use this output for 25 m Model A training or 25 m inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.compute_channel_stats import compute_channel_stats, save_stats


DEFAULT_1KM_OUTPUT = "patches_1km_balanced/channel_stats_1km.json"
DEFAULT_1KM_RESOLUTION_M = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute channel mean/std JSON for 1 km wildfire NPZ patches."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Folder containing 1 km patch_*.npz files, usually patches_1km_balanced/train.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_1KM_OUTPUT,
        help=f"Output JSON path. Default: {DEFAULT_1KM_OUTPUT}",
    )
    parser.add_argument("--label-key", default="class", help="Label/mask key to exclude.")
    parser.add_argument("--pattern", default="patch_*.npz", help="Patch file glob pattern.")
    parser.add_argument(
        "--max-files",
        type=int,
        default=300,
        help="Maximum files to scan. Use -1 to scan all files.",
    )
    parser.add_argument(
        "--std-floor",
        type=float,
        default=0.1,
        help="Minimum std value used for stable normalization. Use 0 to disable.",
    )
    parser.add_argument(
        "--resolution-m",
        type=int,
        default=DEFAULT_1KM_RESOLUTION_M,
        help="Metadata only. Expected pixel resolution for this stats file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_files = None if args.max_files == -1 else args.max_files

    stats = compute_channel_stats(
        data_dir=args.data_dir,
        label_key=args.label_key,
        pattern=args.pattern,
        max_files=max_files,
        std_floor=args.std_floor,
    )

    stats["resolution_m"] = int(args.resolution_m)
    stats["stats_scope"] = "1km"
    stats["source_data_dir"] = str(Path(args.data_dir))
    stats["warning"] = "Use only with 1 km patches/models. Do not reuse for 25 m patches."

    save_stats(stats, args.output)

    print(f"Saved 1 km channel stats to: {args.output}")
    print(f"Resolution metadata: {stats['resolution_m']} m")
    print(f"Files used: {stats['num_files_used']}")
    print(f"Pixels used: {stats['num_pixels_used']}")
    print(f"Channels: {len(stats['mean'])}")
    if stats["skipped_files"]:
        print(f"Skipped files: {len(stats['skipped_files'])}")


if __name__ == "__main__":
    main()
