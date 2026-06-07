"""Run Model A on Model-B candidate cells with live weather and saved NPZs.

This is the validation-mode live-weather Model A runner. It intentionally saves
64 x 64 Model A NPZ patches so we can inspect live-weather candidate patches
before switching to a no-save streaming setup.

By default, the runner can reuse the weather already fetched during the Model B
stage from ``model_b_manifest.csv``. This avoids extra Open-Meteo calls for
Model A and keeps the two stages weather-consistent for the same run.

Example using reused Model B weather:
    python -m src.inference.run_live_weather_model_a_candidates \
      --candidates results/runs/live_weather_model_b_100_current/model_b_candidates.csv \
      --model-b-manifest results/runs/live_weather_model_b_100_current/model_b_manifest.csv \
      --weather-mode model-b-manifest \
      --feature-config configs/model_a_25m_features.json \
      --model models/model_A_25m_spatial_unet.keras \
      --channel-stats /mnt/work/wildfire/25m/patches_25m_balanced/channel_stats.json \
      --run-id live_weather_model_b_100_current \
      --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from pyproj import Transformer

from src.geospatial.create_model_a_25m_patches_from_candidates import (
    centered_bounds,
    get_float,
    read_candidates,
)
from src.geospatial.extract_model_b_patch import extract_feature_array, load_feature_config
from src.inference.run_model_a_geospatial import (
    load_channel_stats,
    print_summary,
    read_manifest_records,
    score_records,
    write_output_csv,
)
from src.weather.open_meteo_client import DEFAULT_TIMEZONE, fetch_live_weather_features, parse_iso_time
from src.weather.weather_patch_generator import (
    DEFAULT_HUMIDITY_NOISE_SCALE,
    DEFAULT_PIXEL_NOISE_FRACTION,
    DEFAULT_TEMPERATURE_NOISE_SCALE,
    DEFAULT_WIND_NOISE_SCALE,
    WeatherPatchConfig,
    generate_weather_patch,
    summarize_weather_patch,
)


DEFAULT_PATCH_SIZE = 64
DEFAULT_RESOLUTION_M = 25.0
DEFAULT_THRESHOLD = 0.50
DEFAULT_MIN_POSITIVE_PIXELS = 1
WEATHER_KEYS = {"temperature", "relative_humidity", "wind_speed"}
WEATHER_MODE_API = "api"
WEATHER_MODE_MODEL_B_MANIFEST = "model-b-manifest"


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("live_weather_model_a_%Y%m%d_%H%M%S")


def apply_default_paths(args: argparse.Namespace) -> argparse.Namespace:
    if args.output_dir is None:
        args.output_dir = f"data/runs/{args.run_id}/model_a_npz"
    if args.manifest is None:
        args.manifest = f"results/runs/{args.run_id}/model_a_manifest.csv"
    if args.output_csv is None:
        args.output_csv = f"results/runs/{args.run_id}/model_a_predictions.csv"
    if args.probability_dir is None:
        args.probability_dir = f"results/runs/{args.run_id}/model_a_probability_tifs"
    if args.binary_dir is None:
        args.binary_dir = f"results/runs/{args.run_id}/model_a_binary_tifs"
    if args.cell_probability_dir is None:
        args.cell_probability_dir = f"results/runs/{args.run_id}/model_a_cell_probability_tifs"
    if args.cell_binary_dir is None:
        args.cell_binary_dir = f"results/runs/{args.run_id}/model_a_cell_binary_tifs"
    if args.run_metadata_json is None:
        args.run_metadata_json = f"results/runs/{args.run_id}/model_a_metadata.json"
    if args.model_b_manifest is None:
        args.model_b_manifest = f"results/runs/{args.run_id}/model_b_manifest.csv"
    return args


def candidate_center(row: dict[str, str]) -> tuple[float, float]:
    cell_xmin = get_float(row, "cell_xmin")
    cell_ymin = get_float(row, "cell_ymin")
    cell_xmax = get_float(row, "cell_xmax")
    cell_ymax = get_float(row, "cell_ymax")
    return (cell_xmin + cell_xmax) / 2.0, (cell_ymin + cell_ymax) / 2.0


def weather_config_from_args(args: argparse.Namespace, index: int) -> WeatherPatchConfig:
    seed = None if args.seed is None else int(args.seed) + int(index)
    return WeatherPatchConfig(
        height=args.patch_size,
        width=args.patch_size,
        temperature_noise_scale=args.temperature_noise_scale,
        humidity_noise_scale=args.humidity_noise_scale,
        wind_noise_scale=args.wind_noise_scale,
        pixel_noise_fraction=args.pixel_noise_fraction,
        seed=seed,
    )


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


def load_model_b_weather_by_patch(manifest_path: str | Path) -> dict[str, dict[str, Any]]:
    """Load Model B weather metadata keyed by coarse patch_id.

    The Model B manifest contains one row per coarse patch and stores the fetched
    Open-Meteo values inside metadata_json. Reusing this for Model A avoids one
    extra API call per candidate cell.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Model B manifest not found: {manifest_path}")

    weather_by_patch: dict[str, dict[str, Any]] = {}
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"patch_id", "status", "metadata_json"}
        missing = required.difference(fieldnames)
        if missing:
            raise ValueError(f"Model B manifest is missing required columns: {sorted(missing)}")

        for row in reader:
            if row.get("status") != "completed":
                continue
            patch_id = str(row.get("patch_id", ""))
            metadata_json = row.get("metadata_json", "")
            if not patch_id or not metadata_json:
                continue
            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError:
                continue
            weather = metadata.get("weather", {})
            if not isinstance(weather, dict):
                continue
            required_weather = ["temperature", "relative_humidity", "wind_speed"]
            if all(key in weather and weather[key] not in (None, "") for key in required_weather):
                weather_by_patch[patch_id] = weather

    if not weather_by_patch:
        raise ValueError(f"No reusable Model B weather records found in: {manifest_path}")
    return weather_by_patch


def generate_weather_patch_for_candidate(
    candidate: dict[str, str],
    lat: float,
    lon: float,
    args: argparse.Namespace,
    target_time: datetime | None,
    weather_config: WeatherPatchConfig,
    model_b_weather_by_patch: dict[str, dict[str, Any]] | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    """Create 64 x 64 weather layers for one Model A candidate."""
    if args.weather_mode == WEATHER_MODE_MODEL_B_MANIFEST:
        if model_b_weather_by_patch is None:
            raise ValueError("Model B weather lookup is required for weather-mode=model-b-manifest.")
        coarse_patch_id = str(candidate.get("patch_id", ""))
        if coarse_patch_id not in model_b_weather_by_patch:
            raise ValueError(f"No Model B weather found for coarse patch_id={coarse_patch_id}")
        weather = dict(model_b_weather_by_patch[coarse_patch_id])
        weather_patch = generate_weather_patch(
            temperature=float(weather["temperature"]),
            relative_humidity=float(weather["relative_humidity"]),
            wind_speed=float(weather["wind_speed"]),
            config=weather_config,
        )
        return weather_patch, weather, WEATHER_MODE_MODEL_B_MANIFEST

    weather_features = fetch_live_weather_features(
        lat=float(lat),
        lon=float(lon),
        target_time=target_time,
        timezone=args.weather_timezone,
        timeout_seconds=args.weather_timeout_seconds,
    )
    weather_patch = generate_weather_patch(
        temperature=weather_features.temperature,
        relative_humidity=weather_features.relative_humidity,
        wind_speed=weather_features.wind_speed,
        config=weather_config,
    )
    return weather_patch, asdict(weather_features), WEATHER_MODE_API


def save_one_live_weather_model_a_npz(
    candidate: dict[str, str],
    candidate_index: int,
    feature_specs: list[Any],
    output_dir: Path,
    args: argparse.Namespace,
    target_time: datetime | None,
    model_b_weather_by_patch: dict[str, dict[str, Any]] | None,
) -> tuple[Path, dict[str, Any]]:
    candidate_id = candidate["candidate_id"]
    crs = candidate.get("crs", "")
    if not crs:
        raise ValueError(f"Candidate {candidate_id} has no CRS")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{candidate_id}.npz"
    bounds = centered_bounds(candidate, args.patch_size, args.resolution_m)
    xmin, ymin, xmax, ymax = bounds
    center_x, center_y = candidate_center(candidate)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(center_x, center_y)

    weather_config = weather_config_from_args(args, candidate_index)
    metadata: dict[str, Any] = {
        "candidate_id": candidate_id,
        "coarse_patch_id": candidate.get("patch_id", ""),
        "source_row": candidate.get("row", ""),
        "source_col": candidate.get("col", ""),
        "source_probability": candidate.get("probability", ""),
        "cell_xmin": candidate.get("cell_xmin", ""),
        "cell_ymin": candidate.get("cell_ymin", ""),
        "cell_xmax": candidate.get("cell_xmax", ""),
        "cell_ymax": candidate.get("cell_ymax", ""),
        "patch_xmin": float(xmin),
        "patch_ymin": float(ymin),
        "patch_xmax": float(xmax),
        "patch_ymax": float(ymax),
        "patch_size": int(args.patch_size),
        "resolution_m": float(args.resolution_m),
        "crs": crs,
        "feature_keys": [spec.key for spec in feature_specs],
        "centroid_x": float(center_x),
        "centroid_y": float(center_y),
        "centroid_lat": float(lat),
        "centroid_lon": float(lon),
        "weather_mode": args.weather_mode,
        "weather_target_time": args.target_time or "",
        "weather_patch_config": asdict(weather_config),
    }

    if output_path.exists() and not args.overwrite:
        return output_path, metadata

    weather_patch, weather, resolved_weather_source = generate_weather_patch_for_candidate(
        candidate=candidate,
        lat=float(lat),
        lon=float(lon),
        args=args,
        target_time=target_time,
        weather_config=weather_config,
        model_b_weather_by_patch=model_b_weather_by_patch,
    )
    metadata["weather_source"] = resolved_weather_source
    metadata["weather"] = weather
    metadata["weather_patch_summary"] = asdict(summarize_weather_patch(weather_patch, weather_config))

    arrays: dict[str, np.ndarray] = {}
    for spec in feature_specs:
        if spec.key in WEATHER_KEYS:
            arrays[spec.key] = weather_patch[spec.key].astype(np.float32)
        else:
            arrays[spec.key] = extract_feature_array(
                spec=spec,
                bounds=bounds,
                dst_crs=crs,
                patch_size=args.patch_size,
                resolution_m=args.resolution_m,
            )

    np.savez_compressed(output_path, **arrays)
    return output_path, metadata


def create_live_weather_model_a_npzs(args: argparse.Namespace, target_time: datetime | None) -> list[dict[str, Any]]:
    candidates = read_candidates(args.candidates)
    if args.max_candidates is not None:
        candidates = candidates[: max(0, args.max_candidates)]
    if not candidates:
        raise ValueError("No candidate rows selected")

    model_b_weather_by_patch = None
    if args.weather_mode == WEATHER_MODE_MODEL_B_MANIFEST:
        model_b_weather_by_patch = load_model_b_weather_by_patch(args.model_b_manifest)

    feature_specs = load_feature_config(args.feature_config)
    output_dir = Path(args.output_dir)
    rows: list[dict[str, Any]] = []

    print("Creating live-weather Model A 25 m patches")
    print(f"Candidates: {len(candidates)}")
    print(f"Features: {len(feature_specs)}")
    print(f"Patch shape: {args.patch_size} x {args.patch_size}")
    print(f"Resolution: {args.resolution_m} m/px")
    print(f"Weather mode: {args.weather_mode}")
    if args.weather_mode == WEATHER_MODE_MODEL_B_MANIFEST:
        print(f"Model B manifest weather source: {args.model_b_manifest}")
    print(f"Output dir: {args.output_dir}")

    for index, candidate in enumerate(candidates):
        candidate_id = candidate.get("candidate_id", "")
        coarse_patch_id = candidate.get("patch_id", "")
        try:
            npz_path, metadata = save_one_live_weather_model_a_npz(
                candidate=candidate,
                candidate_index=index,
                feature_specs=feature_specs,
                output_dir=output_dir,
                args=args,
                target_time=target_time,
                model_b_weather_by_patch=model_b_weather_by_patch,
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
        except Exception as exc:  # noqa: BLE001 - continue over candidate batches.
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


def write_run_metadata(
    path: str | Path,
    args: argparse.Namespace,
    manifest_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    completed_predictions = [row for row in prediction_rows if row.get("status") == "completed"]
    final_positive = [row for row in completed_predictions if int(row.get("final_positive", 0)) == 1]
    payload = {
        "run_id": args.run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "validation_saved_npz",
        "weather_mode": args.weather_mode,
        "weather_source": "model_b_manifest" if args.weather_mode == WEATHER_MODE_MODEL_B_MANIFEST else "open-meteo",
        "model_b_manifest": args.model_b_manifest if args.weather_mode == WEATHER_MODE_MODEL_B_MANIFEST else "",
        "weather_target_time": args.target_time or "",
        "weather_timezone": args.weather_timezone,
        "weather_noise": {
            "temperature_noise_scale": args.temperature_noise_scale,
            "humidity_noise_scale": args.humidity_noise_scale,
            "wind_noise_scale": args.wind_noise_scale,
            "pixel_noise_fraction": args.pixel_noise_fraction,
            "seed": args.seed,
        },
        "inputs": {
            "candidates": args.candidates,
            "model_b_manifest": args.model_b_manifest if args.weather_mode == WEATHER_MODE_MODEL_B_MANIFEST else "",
            "feature_config": args.feature_config,
            "model": args.model,
            "channel_stats": args.channel_stats,
        },
        "outputs": {
            "npz_dir": args.output_dir,
            "manifest": args.manifest,
            "prediction_csv": args.output_csv,
            "run_metadata_json": str(path),
        },
        "counts": {
            "candidate_rows": len(manifest_rows),
            "npz_completed": sum(row.get("status") == "completed" for row in manifest_rows),
            "npz_failed": sum(row.get("status") != "completed" for row in manifest_rows),
            "prediction_rows": len(prediction_rows),
            "prediction_completed": len(completed_predictions),
            "prediction_failed": len(prediction_rows) - len(completed_predictions),
            "final_positive_cells": len(final_positive),
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live-weather Model A candidate inference with saved NPZ patches.")
    parser.add_argument("--candidates", required=True, help="Model B candidate 1 km cell CSV.")
    parser.add_argument("--feature-config", required=True, help="Model A 25 m feature config JSON.")
    parser.add_argument("--model", required=True, help="Trained Model A .keras file.")
    parser.add_argument("--channel-stats", required=True, help="Model A 25 m channel_stats.json.")
    parser.add_argument("--run-id", default=default_run_id())
    parser.add_argument("--target-time", default=None, help="Optional ISO datetime for deterministic hourly weather.")
    parser.add_argument("--weather-timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--weather-timeout-seconds", type=int, default=20)
    parser.add_argument(
        "--weather-mode",
        choices=[WEATHER_MODE_API, WEATHER_MODE_MODEL_B_MANIFEST],
        default=WEATHER_MODE_API,
        help="Use api for fresh Model A weather calls, or model-b-manifest to reuse Model B weather.",
    )
    parser.add_argument(
        "--model-b-manifest",
        default=None,
        help="Model B manifest CSV containing weather metadata. Defaults to results/runs/<run_id>/model_b_manifest.csv.",
    )
    parser.add_argument("--temperature-noise-scale", type=float, default=DEFAULT_TEMPERATURE_NOISE_SCALE)
    parser.add_argument("--humidity-noise-scale", type=float, default=DEFAULT_HUMIDITY_NOISE_SCALE)
    parser.add_argument("--wind-noise-scale", type=float, default=DEFAULT_WIND_NOISE_SCALE)
    parser.add_argument("--pixel-noise-fraction", type=float, default=DEFAULT_PIXEL_NOISE_FRACTION)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--resolution-m", type=float, default=DEFAULT_RESOLUTION_M)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--min-positive-pixels", type=int, default=DEFAULT_MIN_POSITIVE_PIXELS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-rasters", action="store_true", help="Skip writing GeoTIFF rasters and write CSV only.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--probability-dir", default=None)
    parser.add_argument("--binary-dir", default=None)
    parser.add_argument("--cell-probability-dir", default=None)
    parser.add_argument("--cell-binary-dir", default=None)
    parser.add_argument("--run-metadata-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = apply_default_paths(parse_args())
    if args.batch_size <= 0 or args.patch_size <= 0 or args.resolution_m <= 0:
        raise ValueError("Batch size, patch size, and resolution must be positive.")
    if args.min_positive_pixels < 1:
        raise ValueError("--min-positive-pixels must be >= 1.")

    target_time = parse_iso_time(args.target_time)
    manifest_rows = create_live_weather_model_a_npzs(args, target_time)
    write_manifest(manifest_rows, args.manifest)

    completed_npz = sum(row["status"] == "completed" for row in manifest_rows)
    failed_npz = sum(row["status"] != "completed" for row in manifest_rows)
    print("Live-weather Model A NPZ creation summary")
    print(f"Completed: {completed_npz}")
    print(f"Failed: {failed_npz}")

    stats = load_channel_stats(args.channel_stats)
    records = read_manifest_records(args.manifest, max_files=None)

    print("Loading Model A")
    print(f"Model: {args.model}")
    model = tf.keras.models.load_model(args.model, compile=False)

    print("Running Model A on saved live-weather NPZ patches")
    print(f"Patches selected: {len(records)}")
    print(f"Threshold: {args.threshold}")
    print(f"Min positive pixels: {args.min_positive_pixels}")
    print(f"Weather mode: {args.weather_mode}")
    print(f"Write rasters: {not args.no_rasters}")

    prediction_rows = score_records(
        model=model,
        records=records,
        stats=stats,
        threshold=args.threshold,
        min_positive_pixels=args.min_positive_pixels,
        batch_size=args.batch_size,
        probability_dir=args.probability_dir,
        binary_dir=args.binary_dir,
        cell_probability_dir=args.cell_probability_dir,
        cell_binary_dir=args.cell_binary_dir,
        write_rasters=not args.no_rasters,
    )
    write_output_csv(prediction_rows, args.output_csv)
    write_run_metadata(args.run_metadata_json, args, manifest_rows, prediction_rows)
    print_summary(prediction_rows, args.output_csv)
    print(f"Manifest CSV: {args.manifest}")
    print(f"Run metadata JSON: {args.run_metadata_json}")


if __name__ == "__main__":
    main()
