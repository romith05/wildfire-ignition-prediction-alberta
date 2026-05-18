"""Collect Model B hard negatives from a paired-pipeline CSV.

A Model B hard negative is a no-ignition 1 km patch that the current Model B
incorrectly passed through the gate.

Expected CSV source:
    output from src.inference.run_paired_patch_pipeline

Default selection rule:
    model_b_passed == 1
    label_1km_positive == False

The script copies the selected 1 km NPZ files into a hard-negative folder and
writes a manifest CSV for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import pandas as pd


def parse_bool_series(series: pd.Series) -> pd.Series:
    """Parse bool-like CSV values into booleans."""
    parsed = series.astype("string").str.strip().str.lower().map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "1.0": True,
            "0.0": False,
        }
    )
    if parsed.isna().any():
        bad_values = sorted(series[parsed.isna()].astype(str).unique().tolist())
        raise ValueError(f"Could not parse boolean values: {bad_values}")
    return parsed.astype(bool)


def collect_hard_negatives(args: argparse.Namespace) -> None:
    """Copy Model B false-positive 1 km patches into a hard-negative folder."""
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"path_1km", "filename", "model_b_passed", "label_1km_positive", "model_b_max_prob"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["model_b_passed_bool"] = df["model_b_passed"].astype(int).astype(bool)
    df["label_1km_positive_bool"] = parse_bool_series(df["label_1km_positive"])

    selected = df[(df["model_b_passed_bool"] == True) & (df["label_1km_positive_bool"] == False)].copy()

    if args.min_model_b_prob is not None:
        selected = selected[selected["model_b_max_prob"].astype(float) >= args.min_model_b_prob].copy()

    if args.max_files is not None and args.max_files > 0:
        selected = selected.sort_values("model_b_max_prob", ascending=False).head(args.max_files).copy()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped_missing = 0
    skipped_existing = 0
    manifest_rows = []

    for _, row in selected.iterrows():
        src = Path(row["path_1km"])
        dst = output_dir / src.name

        if not src.exists():
            skipped_missing += 1
            manifest_rows.append(
                {
                    "filename": row["filename"],
                    "source_path": str(src),
                    "output_path": str(dst),
                    "model_b_max_prob": row["model_b_max_prob"],
                    "copied": 0,
                    "status": "missing_source",
                }
            )
            continue

        if dst.exists() and not args.overwrite:
            skipped_existing += 1
            manifest_rows.append(
                {
                    "filename": row["filename"],
                    "source_path": str(src),
                    "output_path": str(dst),
                    "model_b_max_prob": row["model_b_max_prob"],
                    "copied": 0,
                    "status": "already_exists",
                }
            )
            continue

        shutil.copy2(src, dst)
        copied += 1
        manifest_rows.append(
            {
                "filename": row["filename"],
                "source_path": str(src),
                "output_path": str(dst),
                "model_b_max_prob": row["model_b_max_prob"],
                "copied": 1,
                "status": "copied",
            }
        )

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "filename",
            "source_path",
            "output_path",
            "model_b_max_prob",
            "copied",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("Model B hard-negative collection complete")
    print(f"CSV: {csv_path}")
    print(f"Selected hard negatives: {len(selected)}")
    print(f"Copied: {copied}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Skipped missing source: {skipped_missing}")
    print(f"Output dir: {output_dir}")
    print(f"Manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Model B hard negatives from paired-pipeline results.")
    parser.add_argument("--csv", required=True, help="Paired-pipeline CSV containing Model B predictions and labels.")
    parser.add_argument("--output-dir", required=True, help="Folder to copy selected hard-negative 1 km NPZ files into.")
    parser.add_argument("--manifest", required=True, help="Manifest CSV path to write.")
    parser.add_argument("--min-model-b-prob", type=float, default=None, help="Optional minimum Model B max probability.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional maximum number of hard negatives to copy.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing copied files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect_hard_negatives(args)


if __name__ == "__main__":
    main()
