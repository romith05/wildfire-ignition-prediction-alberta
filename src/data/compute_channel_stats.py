"""Compute per-channel normalization statistics for wildfire NPZ patches.

This utility matches the patch format used by the wildfire training code:
- patch files are named like ``patch_*.npz``
- feature arrays are 2D rasters inside each NPZ
- the label/mask key defaults to ``class`` and is excluded from statistics

Output JSON contains ``mean`` and ``std`` lists that can be passed into
``NPZPatchDataset(..., channel_stats=stats)`` after loading the JSON.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


DEFAULT_LABEL_KEY = "class"
DEFAULT_PATTERN = "patch_*.npz"
DEFAULT_STD_FLOOR = 0.1
EPS = 1e-6


def find_npz_files(data_dir: str | Path, pattern: str = DEFAULT_PATTERN) -> list[Path]:
    """Return sorted NPZ patch paths from a directory."""
    data_dir = Path(data_dir)
    files = sorted(Path(p) for p in glob.glob(str(data_dir / pattern)))
    if not files:
        raise FileNotFoundError(f"No NPZ patch files found in: {data_dir} using pattern: {pattern}")
    return files


def get_feature_keys(npz_file: np.lib.npyio.NpzFile, label_key: str) -> list[str]:
    """Return feature keys, excluding the label key."""
    return [key for key in npz_file.files if key != label_key]


def load_feature_stack(
    file_path: str | Path,
    label_key: str = DEFAULT_LABEL_KEY,
    expected_feature_keys: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Load one patch as an ``(H, W, C)`` float64 array.

    Args:
        file_path: Path to a ``.npz`` patch file.
        label_key: Key for the target mask. This key is excluded.
        expected_feature_keys: Optional feature-key order from the first valid patch.
            If provided, every later patch must contain the same feature keys.

    Returns:
        Tuple of ``X`` and the feature-key order.
    """
    with np.load(file_path) as arrs:
        if label_key not in arrs.files:
            raise KeyError(f"Missing label key '{label_key}' in {file_path}")

        feature_keys = get_feature_keys(arrs, label_key)
        if not feature_keys:
            raise ValueError(f"No feature keys found in {file_path}")

        if expected_feature_keys is not None:
            missing = [key for key in expected_feature_keys if key not in arrs.files]
            extra = [key for key in feature_keys if key not in expected_feature_keys]
            if missing or extra:
                raise ValueError(
                    f"Feature key mismatch in {file_path}. "
                    f"Missing: {missing}. Extra: {extra}."
                )
            feature_keys = expected_feature_keys

        feature_arrays = [arrs[key] for key in feature_keys]

    first_shape = feature_arrays[0].shape
    if any(arr.shape != first_shape for arr in feature_arrays):
        raise ValueError(f"Feature arrays do not share the same shape in {file_path}")

    X = np.stack(feature_arrays, axis=-1).astype(np.float64)
    return X, feature_keys


def update_running_stats(
    mean_acc: np.ndarray | None,
    m2_acc: np.ndarray | None,
    count: int,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Update channel-wise running mean and variance accumulators.

    Uses a batch-wise version of Welford/parallel variance, matching the notebook
    approach but packaged for reuse as a CLI utility.
    """
    flat = X.reshape(-1, X.shape[-1])
    batch_count = flat.shape[0]
    batch_mean = flat.mean(axis=0)
    batch_var = flat.var(axis=0)
    batch_m2 = batch_var * batch_count

    if mean_acc is None or m2_acc is None:
        return batch_mean, batch_m2, batch_count

    total_count = count + batch_count
    delta = batch_mean - mean_acc
    new_mean = mean_acc + delta * batch_count / total_count
    new_m2 = m2_acc + batch_m2 + delta**2 * (count * batch_count / total_count)

    return new_mean, new_m2, total_count


def compute_channel_stats(
    data_dir: str | Path,
    label_key: str = DEFAULT_LABEL_KEY,
    pattern: str = DEFAULT_PATTERN,
    max_files: int | None = 300,
    std_floor: float = DEFAULT_STD_FLOOR,
) -> dict:
    """Compute per-channel mean/std for NPZ patch features.

    Args:
        data_dir: Folder containing ``patch_*.npz`` files.
        label_key: NPZ key used for the ignition mask/label.
        pattern: Glob pattern for patch files.
        max_files: Maximum number of files to scan. Use ``None`` to scan all files.
        std_floor: Minimum allowed std. This prevents unstable scaling for nearly
            constant channels.

    Returns:
        Dictionary with mean/std lists plus metadata.
    """
    files = find_npz_files(data_dir, pattern=pattern)
    if max_files is not None:
        files = files[: max(0, min(max_files, len(files)))]
    if not files:
        raise ValueError("max_files selected zero files; choose a positive value or omit it.")

    mean_acc: np.ndarray | None = None
    m2_acc: np.ndarray | None = None
    count = 0
    feature_keys: list[str] | None = None
    skipped_files: list[str] = []

    for file_path in files:
        try:
            X, current_keys = load_feature_stack(
                file_path=file_path,
                label_key=label_key,
                expected_feature_keys=feature_keys,
            )
            if feature_keys is None:
                feature_keys = current_keys

            mean_acc, m2_acc, count = update_running_stats(mean_acc, m2_acc, count, X)
        except Exception as exc:
            skipped_files.append(f"{file_path}: {exc}")

    if mean_acc is None or m2_acc is None or feature_keys is None or count == 0:
        raise ValueError(f"No valid NPZ patches found in {data_dir}")

    std = np.sqrt((m2_acc / count) + EPS)
    if std_floor > 0:
        std = np.where(std < std_floor, std_floor, std)

    return {
        "mean": mean_acc.astype(np.float32).tolist(),
        "std": std.astype(np.float32).tolist(),
        "feature_keys": feature_keys,
        "label_key": label_key,
        "num_files_used": len(files) - len(skipped_files),
        "num_pixels_used": int(count),
        "std_floor": float(std_floor),
        "skipped_files": skipped_files,
    }


def save_stats(stats: dict, output_path: str | Path) -> None:
    """Save statistics dictionary as pretty JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute channel mean/std JSON for wildfire NPZ patch normalization."
    )
    parser.add_argument("--data-dir", required=True, help="Folder containing patch_*.npz files.")
    parser.add_argument(
        "--output",
        default="channel_stats.json",
        help="Output JSON path. Example: data/sample_patches/channel_stats.json",
    )
    parser.add_argument("--label-key", default=DEFAULT_LABEL_KEY, help="Label/mask key to exclude.")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="Patch file glob pattern.")
    parser.add_argument(
        "--max-files",
        type=int,
        default=300,
        help="Maximum files to scan. Use -1 to scan all files.",
    )
    parser.add_argument(
        "--std-floor",
        type=float,
        default=DEFAULT_STD_FLOOR,
        help="Minimum std value used for stable normalization. Use 0 to disable.",
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
    save_stats(stats, args.output)

    print(f"Saved channel stats to: {args.output}")
    print(f"Files used: {stats['num_files_used']}")
    print(f"Pixels used: {stats['num_pixels_used']}")
    print(f"Channels: {len(stats['mean'])}")
    if stats["skipped_files"]:
        print(f"Skipped files: {len(stats['skipped_files'])}")


if __name__ == "__main__":
    main()
