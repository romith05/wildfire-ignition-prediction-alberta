"""Create Model A 25 m patches from Model B candidate 1 km cells.

Reads the candidate-cell CSV from run_model_b_geospatial.py and creates one
64x64, 25 m patch centered on each selected 1 km cell.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.geospatial.extract_model_b_patch import extract_feature_array, load_feature_config


DEFAULT_PATCH_SIZE = 64
DEFAULT_RESOLUTION_M = 25.0
DEFAULT_OUTPUT_DIR = "data/cache/model_a_25m_patches"
DEFAULT_MANIFEST = "results/geospatial/model_a_25m_patch_manifest.csv"


def read_candidates(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {path}")

    required = {
        "candidate_id",
        "patch_id",
        "cell_xmin",
        "cell_ymin",
        "cell_xmax",
        "cell_ymax",
        "crs",
    }
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Candidate CSV missing columns: {sorted(missing)}")
        rows = [dict(row) for row in reader]

    if not rows:
        raise ValueError(f"Candidate CSV has no rows: {path}")
    return rows


def get_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"Missing numeric candidate field: {key}")
    return float(value)


def centered_bounds(row: dict[str, str], patch_size: int, resolution_m: float) -> tuple[float, float, float, float]:
    cell_xmin = get_float(row, "cell_xmin")
    cell_ymin = get_float(row, "cell_ymin")
    cell_xmax = get_float(row, "cell_xmax")
    cell_ymax = get_float(row, "cell_ymax")

    center_x = (cell_xmin + cell_xmax) / 2.0
    center_y = (cell_ymin + cell_ymax) / 2.0
    half = (patch_size * resolution_m) / 2.0
    return center_x - half, center_y - half, center_x + half, center_y + half


def extract_one(row: dict[str, str], feature_specs: list[Any], output_dir: Path, patch_size: int, resolution_m: float, overwrite: bool) -> tuple[Path, dict[str, Any]]:
    candidate_id = row["candidate_id"]
    crs = row.get("crs", "")
    if not crs:
        raise ValueError(f"Candidate {candidate_id} has no CRS")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{candidate_id}.npz"
    bounds = centered_bounds(row, patch_size, resolution_m)
    xmin, ymin, xmax, ymax = bounds

    metadata = {
        "candidate_id": candidate_id,
        "coarse_patch_id": row.get("patch_id", ""),
        "source_row": row.get("row", ""),
        "source_col": row.get("col", ""),
        "source_probability": row.get("probability", ""),
        "cell_xmin": row.get("cell_xmin", ""),
        "cell_ymin": row.get("cell_ymin", ""),
        "cell_xmax": row.get("cell_xmax", ""),
        "cell_ymax": row.get("cell_ymax", ""),
        "patch_xmin": float(xmin),
        "patch_ymin": float(ymin),
        "patch_xmax": float(xmax),
        "patch_ymax": float(ymax),
        "patch_size": int(patch_size),
        "resolution_m": float(resolution_m),
        "crs": crs,
        "feature_keys": [spec.key for spec in feature_specs],
    }

    if output_path.exists() and not overwrite:
        return output_path, metadata

    arrays: dict[str, np.ndarray] = {}
    for spec in feature_specs:
        arrays[spec.key] = extract_feature_array(
            spec=spec,
            bounds=bounds,
            dst_crs=crs,
            patch_size=patch_size,
            resolution_m=resolution_m,
        )

    np.savez_compressed(output_path, **arrays)
    return output_path, metadata


def write_manifest(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_id",
        "coarse_patch_id",
        "status",
        "npz_path",
        "message",
        "patch_xmin",
        "patch_ymin",
        "patch_xmax",
        "patch_ymax",
        "crs",
        "metadata_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved manifest: {path}")


def build_patches(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = read_candidates(args.candidates)
    if args.max_candidates is not None:
        candidates = candidates[: max(0, args.max_candidates)]
    if not candidates:
        raise ValueError("No candidate rows selected")

    feature_specs = load_feature_config(args.feature_config)
    output_dir = Path(args.output_dir)
    rows: list[dict[str, Any]] = []

    print("Creating Model A 25 m patches")
    print(f"Candidates: {len(candidates)}")
    print(f"Features: {len(feature_specs)}")
    print(f"Patch shape: {args.patch_size} x {args.patch_size}")
    print(f"Resolution: {args.resolution_m} m/px")

    for candidate in candidates:
        candidate_id = candidate.get("candidate_id", "")
        coarse_patch_id = candidate.get("patch_id", "")
        try:
            npz_path, metadata = extract_one(
                row=candidate,
                feature_specs=feature_specs,
                output_dir=output_dir,
                patch_size=args.patch_size,
                resolution_m=args.resolution_m,
                overwrite=args.overwrite,
            )
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "coarse_patch_id": coarse_patch_id,
                    "status": "completed",
                    "npz_path": str(npz_path),
                    "message": "",
                    "patch_xmin": f"{metadata['patch_xmin']:.3f}",
                    "patch_ymin": f"{metadata['patch_ymin']:.3f}",
                    "patch_xmax": f"{metadata['patch_xmax']:.3f}",
                    "patch_ymax": f"{metadata['patch_ymax']:.3f}",
                    "crs": metadata["crs"],
                    "metadata_json": json.dumps(metadata),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "coarse_patch_id": coarse_patch_id,
                    "status": "failed",
                    "npz_path": "",
                    "message": str(exc),
                    "patch_xmin": "",
                    "patch_ymin": "",
                    "patch_xmax": "",
                    "patch_ymax": "",
                    "crs": candidate.get("crs", ""),
                    "metadata_json": "",
                }
            )
            print(f"Failed {candidate_id}: {exc}")

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 25 m Model A patches from Model B candidate cells.")
    parser.add_argument("--candidates", required=True, help="Model B candidate 1 km cells CSV")
    parser.add_argument("--feature-config", required=True, help="Model A 25 m feature config JSON")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for NPZ patches")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Output manifest CSV")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE, help="Model A patch size in pixels")
    parser.add_argument("--resolution-m", type=float, default=DEFAULT_RESOLUTION_M, help="Model A resolution in metres per pixel")
    parser.add_argument("--max-candidates", type=int, default=None, help="Optional smoke-test cap")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing NPZ files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.patch_size <= 0:
        raise ValueError("--patch-size must be positive")
    if args.resolution_m <= 0:
        raise ValueError("--resolution-m must be positive")

    rows = build_patches(args)
    write_manifest(rows, args.manifest)
    completed = sum(row["status"] == "completed" for row in rows)
    failed = sum(row["status"] == "failed" for row in rows)
    print("Model A 25 m patch creation complete")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
