"""Run Model B geospatial scan with live weather and saved NPZ patches.

This is the validation-mode live-weather runner. It saves the generated Model B
NPZ patches so we can inspect them before switching to a no-save streaming setup.

Example smoke test:
    python -m src.inference.run_live_weather_model_b_scan \
      --grid data/grids/alberta_coarse_grid_epsg3979.geojson \
      --feature-config configs/model_b_1km_features.json \
      --model models/model_B_1km_gatekeeper_hardneg_phase2.keras \
      --channel-stats /mnt/work/wildfire/1km/patches_1km_balanced/channel_stats.json \
      --all \
      --max-patches 10 \
      --run-id live_weather_model_b_smoke \
      --overwrite
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from pyproj import Transformer

from src.geospatial.extract_model_b_patch import (
    build_patch_metadata,
    extract_feature_array,
    load_feature_config,
    load_grid,
    select_grid_rows,
    write_manifest,
)
from src.inference.run_model_b_geospatial import (
    load_channel_stats,
    print_summary,
    score_patches,
    write_candidate_csv,
    write_score_csv,
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


DEFAULT_THRESHOLD = 0.30
DEFAULT_PATCH_SIZE = 32
DEFAULT_RESOLUTION_M = 1000.0
WEATHER_KEYS = {"temperature", "relative_humidity", "wind_speed"}
WEATHER_MODES = {"api", "constant"}


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("live_weather_model_b_%Y%m%d_%H%M%S")


def apply_default_paths(args: argparse.Namespace) -> argparse.Namespace:
    if args.output_dir is None:
        args.output_dir = f"data/runs/{args.run_id}/model_b_npz"
    if args.manifest is None:
        args.manifest = f"results/runs/{args.run_id}/model_b_manifest.csv"
    if args.output_csv is None:
        args.output_csv = f"results/runs/{args.run_id}/model_b_scores.csv"
    if args.candidate_csv is None:
        args.candidate_csv = f"results/runs/{args.run_id}/model_b_candidates.csv"
    if args.run_metadata_json is None:
        args.run_metadata_json = f"results/runs/{args.run_id}/metadata.json"
    return args


def row_bounds(row: Any) -> tuple[float, float, float, float]:
    return (float(row.xmin), float(row.ymin), float(row.xmax), float(row.ymax))


def row_centroid(row: Any) -> tuple[float, float]:
    xmin, ymin, xmax, ymax = row_bounds(row)
    return ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0)


def make_weather_config(args: argparse.Namespace) -> WeatherPatchConfig:
    return WeatherPatchConfig(
        height=args.patch_size,
        width=args.patch_size,
        temperature_noise_scale=args.temperature_noise_scale,
        humidity_noise_scale=args.humidity_noise_scale,
        wind_noise_scale=args.wind_noise_scale,
        pixel_noise_fraction=args.pixel_noise_fraction,
        seed=args.seed,
    )


def validate_weather_args(args: argparse.Namespace) -> None:
    if args.weather_mode not in WEATHER_MODES:
        raise ValueError(f"Unsupported weather mode: {args.weather_mode}. Expected one of {sorted(WEATHER_MODES)}.")

    if args.weather_mode == "constant":
        missing = [
            name
            for name in [
                "constant_temperature",
                "constant_relative_humidity",
                "constant_wind_speed",
            ]
            if getattr(args, name) is None
        ]
        if missing:
            formatted = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            raise ValueError(f"Constant weather mode requires: {formatted}.")


def get_weather_payload(
    *,
    lat: float,
    lon: float,
    args: argparse.Namespace,
    target_time: datetime | None,
) -> dict[str, Any]:
    if args.weather_mode == "constant":
        return {
            "temperature": float(args.constant_temperature),
            "relative_humidity": float(args.constant_relative_humidity),
            "wind_speed": float(args.constant_wind_speed),
            "source": "constant",
            "source_time": args.target_time or "",
            "latitude": float(lat),
            "longitude": float(lon),
            "timezone": args.weather_timezone,
        }

    weather = fetch_live_weather_features(
        lat=float(lat),
        lon=float(lon),
        target_time=target_time,
        timezone=args.weather_timezone,
        timeout_seconds=args.weather_timeout_seconds,
    )
    return asdict(weather)


def save_one_live_weather_npz(
    row: Any,
    feature_specs: list[Any],
    grid_crs: Any,
    transformer: Transformer,
    output_dir: Path,
    args: argparse.Namespace,
    target_time: datetime | None,
    weather_config: WeatherPatchConfig,
) -> tuple[Path, dict[str, Any]]:
    """Create and save one Model B NPZ patch using live or constant weather layers."""
    patch_id = str(row.patch_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{patch_id}.npz"

    metadata = build_patch_metadata(
        row=row,
        feature_specs=feature_specs,
        patch_size=args.patch_size,
        resolution_m=args.resolution_m,
        grid_crs=grid_crs,
    )

    centroid_x, centroid_y = row_centroid(row)
    lon, lat = transformer.transform(centroid_x, centroid_y)
    metadata.update(
        {
            "centroid_x": float(centroid_x),
            "centroid_y": float(centroid_y),
            "centroid_lat": float(lat),
            "centroid_lon": float(lon),
            "weather_source": "open-meteo" if args.weather_mode == "api" else "constant",
            "weather_mode": args.weather_mode,
            "weather_target_time": args.target_time or "",
            "weather_patch_config": asdict(weather_config),
        }
    )

    if output_path.exists() and not args.overwrite:
        return output_path, metadata

    weather = get_weather_payload(
        lat=float(lat),
        lon=float(lon),
        args=args,
        target_time=target_time,
    )
    weather_patch = generate_weather_patch(
        temperature=float(weather["temperature"]),
        relative_humidity=float(weather["relative_humidity"]),
        wind_speed=float(weather["wind_speed"]),
        config=weather_config,
    )
    metadata["weather"] = weather
    metadata["weather_patch_summary"] = asdict(summarize_weather_patch(weather_patch, weather_config))

    bounds = row_bounds(row)
    arrays: dict[str, np.ndarray] = {}
    for spec in feature_specs:
        if spec.key in WEATHER_KEYS:
            arrays[spec.key] = weather_patch[spec.key].astype(np.float32)
        else:
            arrays[spec.key] = extract_feature_array(
                spec=spec,
                bounds=bounds,
                dst_crs=grid_crs,
                patch_size=args.patch_size,
                resolution_m=args.resolution_m,
            )

    np.savez_compressed(output_path, **arrays)
    return output_path, metadata


def create_saved_live_weather_npzs(
    selected: Any,
    feature_specs: list[Any],
    grid_crs: Any,
    args: argparse.Namespace,
    target_time: datetime | None,
    weather_config: WeatherPatchConfig,
) -> list[dict[str, Any]]:
    """Create saved NPZ patches and return extraction manifest rows."""
    transformer = Transformer.from_crs(grid_crs, "EPSG:4326", always_xy=True)
    output_dir = Path(args.output_dir)
    rows: list[dict[str, Any]] = []

    for row in selected.itertuples(index=False):
        patch_id = str(row.patch_id)
        try:
            npz_path, metadata = save_one_live_weather_npz(
                row=row,
                feature_specs=feature_specs,
                grid_crs=grid_crs,
                transformer=transformer,
                output_dir=output_dir,
                args=args,
                target_time=target_time,
                weather_config=weather_config,
            )
            rows.append(
                {
                    "patch_id": patch_id,
                    "status": "completed",
                    "npz_path": str(npz_path),
                    "message": "",
                    "metadata_json": json.dumps(metadata),
                }
            )
        except Exception as exc:  # noqa: BLE001 - continue over province-scale jobs.
            rows.append(
                {
                    "patch_id": patch_id,
                    "status": "failed",
                    "npz_path": "",
                    "message": str(exc),
                    "metadata_json": "",
                }
            )
            print(f"Failed {patch_id}: {exc}")
    return rows


def manifest_rows_to_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert completed manifest rows to score_patches records."""
    records: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "completed" or not row.get("npz_path"):
            continue
        metadata_json = row.get("metadata_json", "")
        metadata = json.loads(metadata_json) if metadata_json else {}
        records.append(
            {
                "patch_id": str(row["patch_id"]),
                "npz_path": Path(row["npz_path"]),
                "metadata": metadata,
            }
        )
    return records


def write_run_metadata(
    path: str | Path,
    args: argparse.Namespace,
    selected_count: int,
    manifest_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": args.run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "validation_saved_npz",
        "weather_source": "open-meteo" if args.weather_mode == "api" else "constant",
        "weather_mode": args.weather_mode,
        "weather_target_time": args.target_time or "",
        "weather_timezone": args.weather_timezone,
        "constant_weather": {
            "temperature": args.constant_temperature,
            "relative_humidity": args.constant_relative_humidity,
            "wind_speed": args.constant_wind_speed,
        }
        if args.weather_mode == "constant"
        else None,
        "weather_noise": {
            "temperature_noise_scale": args.temperature_noise_scale,
            "humidity_noise_scale": args.humidity_noise_scale,
            "wind_noise_scale": args.wind_noise_scale,
            "pixel_noise_fraction": args.pixel_noise_fraction,
            "seed": args.seed,
        },
        "inputs": {
            "grid": args.grid,
            "feature_config": args.feature_config,
            "model": args.model,
            "channel_stats": args.channel_stats,
        },
        "outputs": {
            "npz_dir": args.output_dir,
            "manifest": args.manifest,
            "score_csv": args.output_csv,
            "candidate_csv": args.candidate_csv,
        },
        "counts": {
            "selected_grid_patches": selected_count,
            "npz_completed": sum(row.get("status") == "completed" for row in manifest_rows),
            "npz_failed": sum(row.get("status") != "completed" for row in manifest_rows),
            "score_rows": len(score_rows),
            "candidate_cells": len(candidate_rows),
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live-weather Model B scan and save validation NPZ patches.")
    parser.add_argument("--grid", required=True, help="EPSG:3979 Alberta coarse grid GeoJSON/Parquet.")
    parser.add_argument("--feature-config", required=True, help="Model B feature config JSON.")
    parser.add_argument("--model", required=True, help="Trained Model B .keras file.")
    parser.add_argument("--channel-stats", required=True, help="Model B channel_stats.json.")
    parser.add_argument("--patch-id", default=None, help="Run one grid patch by patch_id.")
    parser.add_argument("--all", action="store_true", help="Run all selected grid patches.")
    parser.add_argument("--max-patches", type=int, default=None, help="Optional cap for smoke tests.")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--resolution-m", type=float, default=DEFAULT_RESOLUTION_M)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--candidate-threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--run-id", default=default_run_id())
    parser.add_argument("--target-time", default=None, help="Optional ISO datetime for deterministic hourly weather.")
    parser.add_argument("--weather-mode", choices=sorted(WEATHER_MODES), default="api")
    parser.add_argument("--constant-temperature", type=float, default=None)
    parser.add_argument("--constant-relative-humidity", type=float, default=None)
    parser.add_argument("--constant-wind-speed", type=float, default=None)
    parser.add_argument("--weather-timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--weather-timeout-seconds", type=int, default=20)
    parser.add_argument("--temperature-noise-scale", type=float, default=DEFAULT_TEMPERATURE_NOISE_SCALE)
    parser.add_argument("--humidity-noise-scale", type=float, default=DEFAULT_HUMIDITY_NOISE_SCALE)
    parser.add_argument("--wind-noise-scale", type=float, default=DEFAULT_WIND_NOISE_SCALE)
    parser.add_argument("--pixel-noise-fraction", type=float, default=DEFAULT_PIXEL_NOISE_FRACTION)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--candidate-csv", default=None)
    parser.add_argument("--run-metadata-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = apply_default_paths(parse_args())
    validate_weather_args(args)
    if args.batch_size <= 0 or args.patch_size <= 0 or args.resolution_m <= 0:
        raise ValueError("Batch size, patch size, and resolution must be positive.")

    candidate_threshold = args.threshold if args.candidate_threshold is None else args.candidate_threshold
    target_time = parse_iso_time(args.target_time)
    weather_config = make_weather_config(args)

    feature_specs = load_feature_config(args.feature_config)
    grid = load_grid(args.grid)
    selected = select_grid_rows(grid, args.patch_id, args.all, args.max_patches)

    print("Live-weather Model B scan")
    print(f"Run ID: {args.run_id}")
    print(f"Grid: {args.grid}")
    print(f"Grid CRS: {grid.crs}")
    print(f"Selected patches: {len(selected)}")
    print(f"Weather mode: {args.weather_mode}")
    print(f"Weather target time: {args.target_time or 'current'}")
    if args.weather_mode == "constant":
        print(
            "Constant weather: "
            f"temperature={args.constant_temperature}, "
            f"relative_humidity={args.constant_relative_humidity}, "
            f"wind_speed={args.constant_wind_speed}"
        )
    print(f"Saving NPZ patches to: {args.output_dir}")

    manifest_rows = create_saved_live_weather_npzs(
        selected=selected,
        feature_specs=feature_specs,
        grid_crs=grid.crs,
        args=args,
        target_time=target_time,
        weather_config=weather_config,
    )
    write_manifest(manifest_rows, Path(args.manifest))
    print("Live-weather NPZ extraction summary")
    print(f"Completed: {sum(row['status'] == 'completed' for row in manifest_rows)}")
    print(f"Failed: {sum(row['status'] != 'completed' for row in manifest_rows)}")

    records = manifest_rows_to_records(manifest_rows)
    if not records:
        raise ValueError("No completed live-weather NPZ patches are available for Model B inference.")

    print("Loading Model B")
    model = tf.keras.models.load_model(args.model, compile=False)
    stats = load_channel_stats(args.channel_stats)

    print("Running Model B on saved live-weather NPZ patches")
    print(f"Batch size: {args.batch_size}")
    print(f"Threshold: {args.threshold}")
    print(f"Candidate threshold: {candidate_threshold}")
    score_rows, candidate_rows = score_patches(
        model=model,
        records=records,
        stats=stats,
        threshold=args.threshold,
        candidate_threshold=candidate_threshold,
        batch_size=args.batch_size,
    )

    write_score_csv(score_rows, args.output_csv)
    write_candidate_csv(candidate_rows, args.candidate_csv)
    write_run_metadata(args.run_metadata_json, args, len(selected), manifest_rows, score_rows, candidate_rows)
    print_summary(score_rows, candidate_rows, args.output_csv, args.candidate_csv)
    print(f"Manifest CSV: {args.manifest}")
    print(f"Run metadata JSON: {args.run_metadata_json}")


if __name__ == "__main__":
    main()
