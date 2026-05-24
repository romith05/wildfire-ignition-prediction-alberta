"""Validate Model A 25 m feature config before fine patch extraction.

This utility checks that a local Model A feature config is compatible with the
25 m channel statistics used when training Model A.

Checks:
- config has unique feature keys
- feature key order matches channel_stats['feature_keys'] when available
- feature count matches channel_stats mean/std lengths
- raster paths exist
- raster band numbers are valid, including multiband TIFFs
- constant weather/month features have numeric values

Example:
    python -m src.geospatial.validate_model_a_feature_config \
      --feature-config configs/model_a_25m_features.json \
      --channel-stats /mnt/work/wildfire/25m/patches_25m_balanced/channel_stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import rasterio


VALID_FEATURE_SOURCES = {"raster", "constant"}


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_config_features(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return and validate the top-level features list."""
    features = config.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("Feature config must contain a non-empty 'features' list.")
    for i, feature in enumerate(features):
        if not isinstance(feature, dict):
            raise ValueError(f"Feature entry {i} must be an object.")
    return features


def validate_config_structure(features: list[dict[str, Any]]) -> list[str]:
    """Validate feature config entries and return keys in config order."""
    keys: list[str] = []
    seen: set[str] = set()

    for i, feature in enumerate(features):
        key = feature.get("key")
        if not key or not isinstance(key, str):
            raise ValueError(f"Feature entry {i} is missing a string 'key'.")
        if key in seen:
            raise ValueError(f"Duplicate feature key: {key}")
        seen.add(key)
        keys.append(key)

        source = str(feature.get("source", "raster")).lower()
        if source not in VALID_FEATURE_SOURCES:
            valid = ", ".join(sorted(VALID_FEATURE_SOURCES))
            raise ValueError(f"Feature '{key}' has invalid source '{source}'. Valid: {valid}")

        if source == "raster":
            validate_raster_feature(key, feature)
        else:
            validate_constant_feature(key, feature)

    return keys


def validate_raster_feature(key: str, feature: dict[str, Any]) -> None:
    """Validate one raster-backed feature config entry."""
    path_value = feature.get("path")
    if not path_value or not isinstance(path_value, str):
        raise ValueError(f"Raster feature '{key}' is missing string path.")

    raster_path = Path(path_value)
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster feature '{key}' path not found: {raster_path}")

    band = int(feature.get("band", 1))
    if band < 1:
        raise ValueError(f"Raster feature '{key}' band must be >= 1.")

    with rasterio.open(raster_path) as src:
        if band > src.count:
            raise ValueError(
                f"Raster feature '{key}' uses band {band}, "
                f"but raster has only {src.count} band(s): {raster_path}"
            )
        if src.crs is None:
            raise ValueError(f"Raster feature '{key}' has no CRS: {raster_path}")


def validate_constant_feature(key: str, feature: dict[str, Any]) -> None:
    """Validate one constant/runtime feature config entry."""
    value = feature.get("value")
    if value is None:
        raise ValueError(f"Constant feature '{key}' is missing numeric value.")
    try:
        float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Constant feature '{key}' value is not numeric: {value}") from exc


def validate_against_channel_stats(config_keys: list[str], channel_stats: dict[str, Any]) -> None:
    """Compare config feature keys against Model A training channel statistics."""
    mean = channel_stats.get("mean")
    std = channel_stats.get("std")
    if not isinstance(mean, list) or not isinstance(std, list):
        raise ValueError("Channel stats must contain list fields 'mean' and 'std'.")
    if len(mean) != len(std):
        raise ValueError(f"Channel stats mean/std length mismatch: {len(mean)} vs {len(std)}")

    if len(config_keys) != len(mean):
        raise ValueError(
            f"Feature count mismatch. Config has {len(config_keys)} feature(s), "
            f"but channel stats have {len(mean)} channel(s)."
        )

    stats_keys = channel_stats.get("feature_keys")
    if stats_keys is None:
        print("Warning: channel stats do not contain feature_keys; checked count only.")
        return
    if not isinstance(stats_keys, list):
        raise ValueError("channel_stats['feature_keys'] must be a list when present.")

    if config_keys != stats_keys:
        missing_from_config = [key for key in stats_keys if key not in config_keys]
        extra_in_config = [key for key in config_keys if key not in stats_keys]
        order_mismatches = [
            (i, config_key, stats_key)
            for i, (config_key, stats_key) in enumerate(zip(config_keys, stats_keys))
            if config_key != stats_key
        ]
        message = [
            "Feature key/order mismatch between config and channel stats.",
            f"Missing from config: {missing_from_config}",
            f"Extra in config: {extra_in_config}",
            f"First order mismatches: {order_mismatches[:10]}",
            "Fix configs/model_a_25m_features.json so keys exactly match Model A training feature_keys.",
        ]
        raise ValueError("\n".join(message))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Model A 25 m feature config.")
    parser.add_argument("--feature-config", required=True, help="Local model_a_25m_features.json path.")
    parser.add_argument("--channel-stats", required=True, help="25 m channel_stats.json used by Model A.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_json(args.feature_config)
    channel_stats = load_json(args.channel_stats)
    features = get_config_features(config)
    config_keys = validate_config_structure(features)
    validate_against_channel_stats(config_keys, channel_stats)

    print("Model A feature config validation passed")
    print(f"Feature count: {len(config_keys)}")
    print("Feature keys:")
    for i, key in enumerate(config_keys):
        print(f"  {i:02d}: {key}")


if __name__ == "__main__":
    main()
