"""Build a Model B hard-negative NPZ folder from prospective mining output.

This script converts `results/validation/prospective_hard_negative_candidates.csv`
into a plain NPZ folder that can be passed to the existing Model B trainer with
`--hard-negative-dir`.

The generated patches are copied from the source live-run NPZs, but their label
mask is replaced with all zeros. This makes them compatible with the existing
Model B hard-negative mixing path, which expects full-patch no-ignition examples.

The script is intentionally conservative:
- it does not modify source NPZ files;
- it writes to a new output folder;
- it writes a manifest CSV documenting every generated patch;
- by default, it de-duplicates by source patch so repeated high-scoring cells from
  the same patch do not silently dominate the supplement.

Example:
    python -m src.training.build_model_b_phase3_hard_negative_dir \
      --input-csv results/validation/prospective_hard_negative_candidates.csv \
      --output-dir data/model_b_hard_negatives/prospective_phase3_ranking

Then train with the existing Model B trainer by adding:
    --hard-negative-dir data/model_b_hard_negatives/prospective_phase3_ranking
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_INPUT_CSV = Path("results/validation/prospective_hard_negative_candidates.csv")
DEFAULT_OUTPUT_DIR = Path("data/model_b_hard_negatives/prospective_phase3_ranking")
DEFAULT_MANIFEST_NAME = "manifest.csv"
DEFAULT_METADATA_NAME = "metadata.json"

NON_BURNABLE_LANDCOVER = {20, 31, 32, 33}


def resolve_path(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    joined = repo_root / path
    if joined.exists():
        return joined
    raise FileNotFoundError(f"Source NPZ not found: {path_value}")


def load_candidates(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    required = [
        "sample_role",
        "npz_path",
        "candidate_id",
        "probability",
        "distance_to_fire_m",
        "rank",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    hard = df[df["sample_role"].astype(str).eq("hard_negative_candidate")].copy()
    if hard.empty:
        raise ValueError("Input CSV contains no hard_negative_candidate rows.")

    hard["probability"] = pd.to_numeric(hard["probability"], errors="coerce")
    hard["distance_to_fire_m"] = pd.to_numeric(hard["distance_to_fire_m"], errors="coerce")
    hard["rank"] = pd.to_numeric(hard["rank"], errors="coerce")
    hard = hard.dropna(subset=["probability", "distance_to_fire_m", "rank"])
    return hard.reset_index(drop=True)


def apply_filters(
    df: pd.DataFrame,
    *,
    min_probability: float,
    min_distance_to_fire_km: float,
    max_rank: int | None,
    include_non_burnable_only: bool,
    exclude_non_burnable: bool,
) -> pd.DataFrame:
    out = df.copy()
    out = out[out["probability"].ge(min_probability)]
    out = out[out["distance_to_fire_m"].ge(min_distance_to_fire_km * 1000.0)]
    if max_rank is not None:
        out = out[out["rank"].le(max_rank)]

    if "landcover_code" in out.columns:
        landcover = pd.to_numeric(out["landcover_code"], errors="coerce").astype("Int64")
        if include_non_burnable_only and exclude_non_burnable:
            raise ValueError("Cannot use both --include-non-burnable-only and --exclude-non-burnable.")
        if include_non_burnable_only:
            out = out[landcover.isin(NON_BURNABLE_LANDCOVER)]
        if exclude_non_burnable:
            out = out[~landcover.isin(NON_BURNABLE_LANDCOVER)]

    return out.reset_index(drop=True)


def dedupe_candidates(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "none":
        return df.reset_index(drop=True)
    if mode == "source_patch":
        return (
            df.sort_values(["npz_path", "probability"], ascending=[True, False])
            .drop_duplicates(subset=["npz_path"], keep="first")
            .sort_values("probability", ascending=False)
            .reset_index(drop=True)
        )
    if mode == "candidate_cell":
        keys = [key for key in ["npz_path", "row", "col"] if key in df.columns]
        if not keys:
            return df.drop_duplicates(subset=["candidate_id"], keep="first").reset_index(drop=True)
        return df.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)
    raise ValueError(f"Unsupported dedupe mode: {mode}")


def npz_feature_payload(source_npz: Path, label_key: str) -> dict[str, Any]:
    with np.load(source_npz) as arrs:
        feature_keys = sorted([key for key in arrs.files if key != label_key])
        if not feature_keys:
            raise ValueError(f"No feature arrays found in {source_npz}")

        first = np.asarray(arrs[feature_keys[0]])
        if first.ndim != 2:
            raise ValueError(f"Expected 2D feature arrays in {source_npz}; got {first.shape}")

        payload: dict[str, Any] = {}
        for key in feature_keys:
            array = np.asarray(arrs[key])
            if array.shape != first.shape:
                raise ValueError(
                    f"Feature shape mismatch in {source_npz}: {key} has {array.shape}, expected {first.shape}"
                )
            payload[key] = array.astype(np.float32, copy=True)

    payload[label_key] = np.zeros(first.shape, dtype=np.float32)
    return payload


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite to replace generated files."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def build_hard_negative_dir(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    repo_root = Path(args.repo_root).resolve()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)

    candidates = load_candidates(input_csv)
    filtered = apply_filters(
        candidates,
        min_probability=args.min_probability,
        min_distance_to_fire_km=args.min_distance_to_fire_km,
        max_rank=args.max_rank,
        include_non_burnable_only=args.include_non_burnable_only,
        exclude_non_burnable=args.exclude_non_burnable,
    )
    selected = dedupe_candidates(filtered, args.dedupe_mode)
    if selected.empty:
        raise ValueError("No candidates remain after filtering and de-duplication.")

    prepare_output_dir(output_dir, overwrite=args.overwrite)

    manifest_rows: list[dict[str, Any]] = []
    for out_idx, row in enumerate(selected.itertuples(index=False), start=1):
        source_npz = resolve_path(row.npz_path, repo_root)
        payload = npz_feature_payload(source_npz, label_key=args.label_key)
        output_name = f"patch_prospective_hardneg_{out_idx:05d}.npz"
        output_path = output_dir / output_name
        np.savez_compressed(output_path, **payload)

        record = row._asdict()
        record.update(
            {
                "generated_npz_path": str(output_path),
                "generated_npz_name": output_name,
                "source_npz_resolved": str(source_npz),
                "label_key": args.label_key,
                "label_policy": "all_zero_full_patch_hard_negative",
            }
        )
        manifest_rows.append(record)

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / DEFAULT_MANIFEST_NAME
    manifest.to_csv(manifest_path, index=False)

    metadata = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "manifest_csv": str(manifest_path),
        "label_key": args.label_key,
        "label_policy": "all_zero_full_patch_hard_negative",
        "dedupe_mode": args.dedupe_mode,
        "source_rows": int(len(candidates)),
        "filtered_rows": int(len(filtered)),
        "generated_patches": int(len(manifest)),
        "min_probability": args.min_probability,
        "min_distance_to_fire_km": args.min_distance_to_fire_km,
        "max_rank": args.max_rank,
        "include_non_burnable_only": bool(args.include_non_burnable_only),
        "exclude_non_burnable": bool(args.exclude_non_burnable),
    }
    metadata_path = output_dir / DEFAULT_METADATA_NAME
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return manifest, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Model B hard-negative NPZ folder from prospective mining output."
    )
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--label-key", default="class")
    parser.add_argument("--min-probability", type=float, default=0.90)
    parser.add_argument("--min-distance-to-fire-km", type=float, default=50.0)
    parser.add_argument("--max-rank", type=int, default=None)
    parser.add_argument(
        "--dedupe-mode",
        choices=["source_patch", "candidate_cell", "none"],
        default="source_patch",
        help="source_patch is conservative for full-patch zero-label hard negatives.",
    )
    parser.add_argument("--include-non-burnable-only", action="store_true")
    parser.add_argument("--exclude-non-burnable", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, metadata = build_hard_negative_dir(args)

    print("Model B phase-3 hard-negative NPZ builder")
    print("=" * 48)
    print(f"Input CSV: {metadata['input_csv']}")
    print(f"Output dir: {metadata['output_dir']}")
    print(f"Source rows: {metadata['source_rows']}")
    print(f"Filtered rows: {metadata['filtered_rows']}")
    print(f"Generated patches: {metadata['generated_patches']}")
    print(f"Dedupe mode: {metadata['dedupe_mode']}")
    print(f"Manifest: {metadata['manifest_csv']}")

    if not manifest.empty:
        cols = [
            "generated_npz_name",
            "fire_id",
            "candidate_id",
            "probability",
            "rank",
            "distance_to_fire_m",
            "landcover_code",
            "landcover_label",
            "reason_flags",
        ]
        available = [col for col in cols if col in manifest.columns]
        print("")
        print(manifest[available].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
