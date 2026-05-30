"""Generate spatial weather feature layers from point weather observations.

The live weather API returns point values for a patch centroid or candidate-cell
centroid. The ML patch models, however, expect 2D feature arrays. This module
turns one point weather reading into patch-shaped weather layers using a small,
configurable spatial perturbation.

The default noise scales are intentionally conservative because the models were
trained with weather layers that were often spatially constant across a patch.
Use larger values only after validating model behavior.

Model feature keys produced by default:
- temperature
- relative_humidity
- wind_speed

Examples:
    python -m src.weather.weather_patch_generator \
      --temperature 20 \
      --relative-humidity 35 \
      --wind-speed 10 \
      --height 32 \
      --width 32 \
      --seed 42

    python -m src.weather.weather_patch_generator \
      --temperature 20 \
      --relative-humidity 35 \
      --wind-speed 10 \
      --height 64 \
      --width 64 \
      --temperature-noise-scale 0.5 \
      --humidity-noise-scale 2.0 \
      --wind-noise-scale 0.5 \
      --output-npz /tmp/weather_patch.npz
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_TEMPERATURE_NOISE_SCALE = 0.5
DEFAULT_HUMIDITY_NOISE_SCALE = 2.0
DEFAULT_WIND_NOISE_SCALE = 0.5
DEFAULT_PIXEL_NOISE_FRACTION = 0.10


@dataclass(frozen=True)
class WeatherPatchConfig:
    """Configuration for point-to-patch weather extrapolation."""

    height: int
    width: int
    temperature_noise_scale: float = DEFAULT_TEMPERATURE_NOISE_SCALE
    humidity_noise_scale: float = DEFAULT_HUMIDITY_NOISE_SCALE
    wind_noise_scale: float = DEFAULT_WIND_NOISE_SCALE
    pixel_noise_fraction: float = DEFAULT_PIXEL_NOISE_FRACTION
    seed: int | None = None


@dataclass(frozen=True)
class WeatherPatchSummary:
    """Compact numeric summary for logging/debugging."""

    height: int
    width: int
    seed: int | None
    temperature_min: float
    temperature_max: float
    temperature_mean: float
    relative_humidity_min: float
    relative_humidity_max: float
    relative_humidity_mean: float
    wind_speed_min: float
    wind_speed_max: float
    wind_speed_mean: float


def _validate_shape(height: int, width: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError(f"Patch shape must be positive, got {height} x {width}.")


def _validate_noise_scale(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}.")


def _smooth_corner_field(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Create a smooth 2D field by bilinear interpolation between random corners."""
    corners = rng.normal(loc=0.0, scale=1.0, size=(2, 2)).astype(np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

    top = corners[0, 0] * (1.0 - x) + corners[0, 1] * x
    bottom = corners[1, 0] * (1.0 - x) + corners[1, 1] * x
    field = top * (1.0 - y) + bottom * y
    field = field.astype(np.float32)

    std = float(np.std(field))
    if std > 1e-6:
        field = (field - float(np.mean(field))) / std
    else:
        field = field - float(np.mean(field))
    return field.astype(np.float32)


def extrapolate_point_value_to_layer(
    value: float,
    height: int,
    width: int,
    noise_scale: float,
    rng: np.random.Generator,
    pixel_noise_fraction: float = DEFAULT_PIXEL_NOISE_FRACTION,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> np.ndarray:
    """Create one patch-shaped weather layer from a point value.

    The perturbation has two parts:
    1. A smooth low-frequency field, representing gradual spatial variation.
    2. A small pixel-level noise term, controlled by pixel_noise_fraction.

    Setting noise_scale to 0 returns a constant layer.
    """
    _validate_shape(height, width)
    _validate_noise_scale("noise_scale", noise_scale)
    _validate_noise_scale("pixel_noise_fraction", pixel_noise_fraction)

    layer = np.full((height, width), float(value), dtype=np.float32)
    if noise_scale > 0:
        smooth_field = _smooth_corner_field(height, width, rng)
        pixel_noise = rng.normal(
            loc=0.0,
            scale=noise_scale * pixel_noise_fraction,
            size=(height, width),
        ).astype(np.float32)
        layer = layer + (smooth_field * float(noise_scale)).astype(np.float32) + pixel_noise

    if clip_min is not None or clip_max is not None:
        min_value = -np.inf if clip_min is None else float(clip_min)
        max_value = np.inf if clip_max is None else float(clip_max)
        layer = np.clip(layer, min_value, max_value)

    return layer.astype(np.float32)


def generate_weather_patch(
    temperature: float,
    relative_humidity: float,
    wind_speed: float,
    config: WeatherPatchConfig,
) -> dict[str, np.ndarray]:
    """Generate model-ready weather feature arrays for one patch."""
    _validate_shape(config.height, config.width)
    _validate_noise_scale("temperature_noise_scale", config.temperature_noise_scale)
    _validate_noise_scale("humidity_noise_scale", config.humidity_noise_scale)
    _validate_noise_scale("wind_noise_scale", config.wind_noise_scale)
    _validate_noise_scale("pixel_noise_fraction", config.pixel_noise_fraction)

    rng = np.random.default_rng(config.seed)
    return {
        "temperature": extrapolate_point_value_to_layer(
            value=float(temperature),
            height=config.height,
            width=config.width,
            noise_scale=config.temperature_noise_scale,
            rng=rng,
            pixel_noise_fraction=config.pixel_noise_fraction,
            clip_min=-60.0,
            clip_max=60.0,
        ),
        "relative_humidity": extrapolate_point_value_to_layer(
            value=float(relative_humidity),
            height=config.height,
            width=config.width,
            noise_scale=config.humidity_noise_scale,
            rng=rng,
            pixel_noise_fraction=config.pixel_noise_fraction,
            clip_min=0.0,
            clip_max=100.0,
        ),
        "wind_speed": extrapolate_point_value_to_layer(
            value=float(wind_speed),
            height=config.height,
            width=config.width,
            noise_scale=config.wind_noise_scale,
            rng=rng,
            pixel_noise_fraction=config.pixel_noise_fraction,
            clip_min=0.0,
            clip_max=None,
        ),
    }


def summarize_weather_patch(
    patch: dict[str, np.ndarray],
    config: WeatherPatchConfig,
) -> WeatherPatchSummary:
    """Return a compact summary for generated weather arrays."""
    required = ["temperature", "relative_humidity", "wind_speed"]
    missing = [key for key in required if key not in patch]
    if missing:
        raise ValueError(f"Weather patch missing required key(s): {missing}")

    temperature = patch["temperature"]
    humidity = patch["relative_humidity"]
    wind = patch["wind_speed"]

    return WeatherPatchSummary(
        height=int(config.height),
        width=int(config.width),
        seed=config.seed,
        temperature_min=float(np.min(temperature)),
        temperature_max=float(np.max(temperature)),
        temperature_mean=float(np.mean(temperature)),
        relative_humidity_min=float(np.min(humidity)),
        relative_humidity_max=float(np.max(humidity)),
        relative_humidity_mean=float(np.mean(humidity)),
        wind_speed_min=float(np.min(wind)),
        wind_speed_max=float(np.max(wind)),
        wind_speed_mean=float(np.mean(wind)),
    )


def save_weather_patch_npz(patch: dict[str, np.ndarray], output_npz: str | Path) -> None:
    """Save weather arrays to a compressed NPZ for smoke testing."""
    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **patch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate patch-shaped weather layers from point weather values.")
    parser.add_argument("--temperature", type=float, required=True, help="Point temperature value in Celsius.")
    parser.add_argument("--relative-humidity", type=float, required=True, help="Point relative humidity percentage.")
    parser.add_argument("--wind-speed", type=float, required=True, help="Point wind speed in km/h.")
    parser.add_argument("--height", type=int, required=True, help="Output patch height in pixels.")
    parser.add_argument("--width", type=int, required=True, help="Output patch width in pixels.")
    parser.add_argument("--temperature-noise-scale", type=float, default=DEFAULT_TEMPERATURE_NOISE_SCALE)
    parser.add_argument("--humidity-noise-scale", type=float, default=DEFAULT_HUMIDITY_NOISE_SCALE)
    parser.add_argument("--wind-noise-scale", type=float, default=DEFAULT_WIND_NOISE_SCALE)
    parser.add_argument("--pixel-noise-fraction", type=float, default=DEFAULT_PIXEL_NOISE_FRACTION)
    parser.add_argument("--seed", type=int, default=None, help="Optional deterministic seed.")
    parser.add_argument("--output-npz", default=None, help="Optional NPZ output path for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = WeatherPatchConfig(
        height=args.height,
        width=args.width,
        temperature_noise_scale=args.temperature_noise_scale,
        humidity_noise_scale=args.humidity_noise_scale,
        wind_noise_scale=args.wind_noise_scale,
        pixel_noise_fraction=args.pixel_noise_fraction,
        seed=args.seed,
    )
    patch = generate_weather_patch(
        temperature=args.temperature,
        relative_humidity=args.relative_humidity,
        wind_speed=args.wind_speed,
        config=config,
    )
    summary = summarize_weather_patch(patch, config)

    if args.output_npz:
        save_weather_patch_npz(patch, args.output_npz)

    payload: dict[str, Any] = {
        "config": asdict(config),
        "summary": asdict(summary),
        "output_npz": args.output_npz or "",
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
