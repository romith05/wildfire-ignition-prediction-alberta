#!/usr/bin/env python3
"""Export a public-safe dashboard data bundle.

This script is intended to run after a successful live inference / validation
cycle on the university machine. It copies only dashboard-ready outputs into a
small public bundle and removes internal machine paths such as ``npz_path``.

Default output:
    public_dashboard_bundle/

The exported bundle can be pushed to a public dashboard repository or served by
Streamlit Cloud / GitHub Pages / another public host. It should not include raw
rasters, NPZ patches, model weights, API keys, or university filesystem paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_DIR = Path("public_dashboard_bundle")
LATEST_RUN_FILE = Path("results/validation/latest_prediction_run.txt")
RUNS_ROOT = Path("results/runs")
VALIDATION_LOG = Path("results/validation/prospective_validation_log.csv")
ACTIVE_FIRES = Path("data/validation/alberta_activefires_current.csv")

# Columns that should not be published because they expose local filesystem
# structure, generated patch locations, or internal processing details.
SENSITIVE_EXACT_COLUMNS = {
    "npz_path",
    "patch_path",
    "model_path",
    "raster_path",
    "source_path",
    "input_path",
    "output_path",
    "local_path",
    "file_path",
    "filepath",
}
SENSITIVE_NAME_FRAGMENTS = (
    "/mnt/",
    "/home/",
    "\\\\",
    "npz_path",
    "path",
)

# Optional diagnostics produced by the validation scripts. These are useful for
# the conference dashboard but should still be cleaned before publishing.
OPTIONAL_VALIDATION_FILES = [
    Path("results/validation/model_a_threshold_sweep_summary.csv"),
    Path("results/validation/prospective_ranking_diagnostic_summary.csv"),
    Path("results/validation/prospective_ranking_diagnostic_detail.csv"),
    Path("results/validation/ranked_candidate_feature_comparison_summary.csv"),
    Path("results/validation/prospective_hard_negative_summary.csv"),
]

RUN_OUTPUT_FILES = [
    "model_b_candidates.csv",
    "model_b_scores.csv",
    "model_a_predictions.csv",
    "model_a_candidate_cells.geojson",
]


@dataclass(frozen=True)
class ExportedFile:
    source: str
    output: str
    bytes: int
    sha256: str
    rows: int | None = None
    columns: list[str] | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_safe_output_dir(path: Path) -> None:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    if resolved == cwd:
        raise ValueError("Refusing to use repository root as output directory")
    if cwd not in resolved.parents:
        raise ValueError(f"Output directory must be inside the repository: {path}")


def reset_output_dir(path: Path) -> None:
    ensure_safe_output_dir(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_latest_run_id(latest_run_file: Path = LATEST_RUN_FILE) -> str:
    if not latest_run_file.exists():
        raise FileNotFoundError(f"Latest run pointer not found: {latest_run_file}")
    run_id = latest_run_file.read_text(encoding="utf-8").strip()
    if not run_id:
        raise ValueError(f"Latest run pointer is empty: {latest_run_file}")
    return run_id


def is_sensitive_column(name: str) -> bool:
    lowered = name.strip().lower()
    if lowered in SENSITIVE_EXACT_COLUMNS:
        return True
    # Keep public geometry/bound columns such as cell_xmin and cell_ymax.
    if lowered.startswith("cell_") or lowered.startswith("coarse_"):
        return False
    return any(fragment == lowered or fragment in lowered for fragment in SENSITIVE_NAME_FRAGMENTS)


def value_looks_internal(value: Any) -> bool:
    if value is None:
        return False
    text = str(value)
    return any(fragment in text for fragment in ("/mnt/", "/home/", "\\\\"))


def clean_scalar(value: Any) -> Any:
    if value_looks_internal(value):
        return None
    return value


def copy_text(src: Path, dst: Path) -> ExportedFile:
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8").strip() + "\n"
    dst.write_text(text, encoding="utf-8")
    return ExportedFile(
        source=str(src),
        output=str(dst),
        bytes=dst.stat().st_size,
        sha256=sha256_file(dst),
    )


def clean_csv(src: Path, dst: Path) -> ExportedFile:
    dst.parent.mkdir(parents=True, exist_ok=True)

    with src.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            dst.write_text("", encoding="utf-8")
            return ExportedFile(str(src), str(dst), 0, sha256_file(dst), rows=0, columns=[])

        keep_columns = [col for col in reader.fieldnames if not is_sensitive_column(col)]
        rows_written = 0

        with dst.open("w", encoding="utf-8", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=keep_columns)
            writer.writeheader()
            for row in reader:
                cleaned = {col: clean_scalar(row.get(col)) for col in keep_columns}
                writer.writerow(cleaned)
                rows_written += 1

    return ExportedFile(
        source=str(src),
        output=str(dst),
        bytes=dst.stat().st_size,
        sha256=sha256_file(dst),
        rows=rows_written,
        columns=keep_columns,
    )


def clean_geojson_properties(properties: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in properties.items():
        if is_sensitive_column(key):
            continue
        cleaned[key] = clean_scalar(value)
    return cleaned


def clean_geojson(src: Path, dst: Path) -> ExportedFile:
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(src.read_text(encoding="utf-8"))

    feature_count = None
    property_keys: set[str] = set()

    if data.get("type") == "FeatureCollection":
        features = data.get("features", [])
        feature_count = len(features)
        for feature in features:
            props = feature.get("properties") or {}
            cleaned = clean_geojson_properties(props)
            feature["properties"] = cleaned
            property_keys.update(cleaned.keys())
    elif data.get("type") == "Feature":
        props = data.get("properties") or {}
        cleaned = clean_geojson_properties(props)
        data["properties"] = cleaned
        property_keys.update(cleaned.keys())
        feature_count = 1

    dst.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return ExportedFile(
        source=str(src),
        output=str(dst),
        bytes=dst.stat().st_size,
        sha256=sha256_file(dst),
        rows=feature_count,
        columns=sorted(property_keys),
    )


def export_file(src: Path, dst: Path) -> ExportedFile:
    suffix = src.suffix.lower()
    if suffix == ".csv":
        return clean_csv(src, dst)
    if suffix in {".geojson", ".json"}:
        if suffix == ".geojson":
            return clean_geojson(src, dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(src.read_text(encoding="utf-8"))
        dst.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return ExportedFile(str(src), str(dst), dst.stat().st_size, sha256_file(dst))
    return copy_text(src, dst)


def write_metadata(
    output_dir: Path,
    run_id: str,
    exported_files: Iterable[ExportedFile],
    missing_files: Iterable[str],
) -> None:
    metadata = {
        "bundle_schema_version": 1,
        "generated_at_utc": utc_now_iso(),
        "latest_run_id": run_id,
        "description": "Public-safe dashboard data bundle for wildfire ignition-risk dashboard.",
        "public_safety_notes": [
            "Raw rasters are not exported.",
            "NPZ patches are not exported.",
            "Model weight files are not exported.",
            "Internal filesystem path columns are removed where detected.",
        ],
        "files": [file.__dict__ for file in exported_files],
        "missing_optional_files": list(missing_files),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def export_bundle(output_dir: Path) -> tuple[str, list[ExportedFile], list[str]]:
    run_id = read_latest_run_id()
    run_dir = RUNS_ROOT / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Latest run directory not found: {run_dir}")

    reset_output_dir(output_dir)

    exported: list[ExportedFile] = []
    missing: list[str] = []

    exported.append(copy_text(LATEST_RUN_FILE, output_dir / "latest_prediction_run.txt"))

    core_files = [
        (VALIDATION_LOG, output_dir / "data" / "prospective_validation_log.csv"),
        (ACTIVE_FIRES, output_dir / "data" / "active_fires.csv"),
    ]

    for src, dst in core_files:
        if src.exists():
            exported.append(export_file(src, dst))
        else:
            missing.append(str(src))

    for src in OPTIONAL_VALIDATION_FILES:
        if src.exists():
            exported.append(export_file(src, output_dir / "diagnostics" / src.name))
        else:
            missing.append(str(src))

    latest_public_run_dir = output_dir / "runs" / "latest"
    named_public_run_dir = output_dir / "runs" / run_id

    for filename in RUN_OUTPUT_FILES:
        src = run_dir / filename
        if not src.exists():
            missing.append(str(src))
            continue

        exported_file = export_file(src, latest_public_run_dir / filename)
        exported.append(exported_file)

        # Also keep a run-id version for reproducibility. Copy the cleaned output
        # rather than cleaning twice so hashes remain predictable for latest/.
        named_dst = named_public_run_dir / filename
        named_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(latest_public_run_dir / filename, named_dst)
        exported.append(
            ExportedFile(
                source=str(src),
                output=str(named_dst),
                bytes=named_dst.stat().st_size,
                sha256=sha256_file(named_dst),
                rows=exported_file.rows,
                columns=exported_file.columns,
            )
        )

    write_metadata(output_dir, run_id, exported, missing)
    return run_id, exported, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the public-safe dashboard bundle will be written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id, exported, missing = export_bundle(args.output_dir)

    print(f"Exported public dashboard bundle for run: {run_id}")
    print(f"Output directory: {args.output_dir}")
    print(f"Files exported: {len(exported)}")
    if missing:
        print("Missing optional/input files:")
        for item in missing:
            print(f"  - {item}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
