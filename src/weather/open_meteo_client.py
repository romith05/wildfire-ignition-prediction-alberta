"""Open-Meteo live weather client for wildfire inference features.

This module is intentionally small and dependency-light. It fetches the three
weather features currently used by the trained patch models:

- temperature
- relative_humidity
- wind_speed

The output keys match the NPZ feature names expected by the existing geospatial
patch extraction configs.

The client includes conservative request throttling because Open-Meteo's free
API has usage limits. By default, requests are spaced by at least 0.25 seconds,
which is at most 240 requests per minute from this Python process.

Example:
    python -m src.weather.open_meteo_client \
      --lat 53.5461 \
      --lon -113.4938
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_TIMEZONE = "America/Edmonton"
REQUESTED_VARIABLES = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 5.0
ENV_MIN_INTERVAL_SECONDS = "OPEN_METEO_MIN_REQUEST_INTERVAL_SECONDS"


_LAST_REQUEST_MONOTONIC: float | None = None


@dataclass(frozen=True)
class WeatherFeatures:
    """Model-ready weather features for one point and timestamp."""

    temperature: float
    relative_humidity: float
    wind_speed: float
    source: str
    source_time: str
    latitude: float
    longitude: float
    timezone: str


def default_min_request_interval_seconds() -> float:
    """Return conservative Open-Meteo request spacing.

    The default value can be overridden with the environment variable
    OPEN_METEO_MIN_REQUEST_INTERVAL_SECONDS. Invalid environment values fall
    back to 0.25 seconds.
    """
    value = os.environ.get(ENV_MIN_INTERVAL_SECONDS, "0.25")
    try:
        parsed = float(value)
    except ValueError:
        return 0.25
    return max(0.0, parsed)


def parse_iso_time(value: str | None) -> datetime | None:
    """Parse an ISO datetime string if provided."""
    if not value:
        return None
    clean_value = value.replace("Z", "+00:00")
    return datetime.fromisoformat(clean_value)


def build_open_meteo_url(
    lat: float,
    lon: float,
    timezone: str = DEFAULT_TIMEZONE,
    forecast_days: int = 1,
) -> str:
    """Build an Open-Meteo forecast API URL for current and hourly features."""
    params = {
        "latitude": f"{lat:.8f}",
        "longitude": f"{lon:.8f}",
        "current": ",".join(REQUESTED_VARIABLES),
        "hourly": ",".join(REQUESTED_VARIABLES),
        "timezone": timezone,
        "forecast_days": str(forecast_days),
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    return f"{OPEN_METEO_FORECAST_URL}?{urlencode(params)}"


def throttle_open_meteo_requests(min_interval_seconds: float) -> None:
    """Sleep when needed so requests from this process stay below a safe rate."""
    global _LAST_REQUEST_MONOTONIC

    min_interval_seconds = max(0.0, float(min_interval_seconds))
    if min_interval_seconds <= 0:
        _LAST_REQUEST_MONOTONIC = time.monotonic()
        return

    now = time.monotonic()
    if _LAST_REQUEST_MONOTONIC is not None:
        elapsed = now - _LAST_REQUEST_MONOTONIC
        remaining = min_interval_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
    _LAST_REQUEST_MONOTONIC = time.monotonic()


def retry_after_seconds(exc: HTTPError, fallback_seconds: float) -> float:
    """Read HTTP Retry-After header when present."""
    retry_after = exc.headers.get("Retry-After") if exc.headers else None
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            return fallback_seconds
    return fallback_seconds


def fetch_json(
    url: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    min_request_interval_seconds: float | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
) -> dict[str, Any]:
    """Fetch JSON from a URL using the Python standard library.

    Requests are throttled by default to avoid hitting Open-Meteo free-tier
    minute limits. HTTP 429 and transient URL errors are retried with backoff.
    """
    if min_request_interval_seconds is None:
        min_request_interval_seconds = default_min_request_interval_seconds()

    attempts = max(1, int(max_retries) + 1)
    last_error: Exception | None = None

    for attempt in range(attempts):
        throttle_open_meteo_requests(min_request_interval_seconds)
        try:
            with urlopen(url, timeout=timeout_seconds) as response:  # noqa: S310 - fixed HTTPS API endpoint.
                payload = response.read().decode("utf-8")
            data = json.loads(payload)
            if not isinstance(data, dict):
                raise ValueError("Open-Meteo response was not a JSON object.")
            if "error" in data:
                reason = data.get("reason", "unknown error")
                raise ValueError(f"Open-Meteo API error: {reason}")
            return data
        except HTTPError as exc:
            last_error = exc
            if exc.code != 429 or attempt >= attempts - 1:
                raise ValueError(f"Open-Meteo HTTP error {exc.code}: {exc.reason}") from exc
            sleep_seconds = retry_after_seconds(
                exc,
                fallback_seconds=retry_backoff_seconds * float(attempt + 1),
            )
            time.sleep(sleep_seconds)
        except URLError as exc:
            last_error = exc
            if attempt >= attempts - 1:
                raise ValueError(f"Open-Meteo request failed: {exc.reason}") from exc
            time.sleep(retry_backoff_seconds * float(attempt + 1))

    raise ValueError(f"Open-Meteo request failed after retries: {last_error}")


def _get_wind_value(mapping: dict[str, Any]) -> Any:
    """Read wind speed from current or legacy response key names."""
    if "wind_speed_10m" in mapping:
        return mapping.get("wind_speed_10m")
    return mapping.get("windspeed_10m")


def weather_from_current(data: dict[str, Any], lat: float, lon: float) -> WeatherFeatures | None:
    """Extract model-ready weather features from the current conditions block."""
    current = data.get("current")
    if not isinstance(current, dict):
        return None

    temperature = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")
    wind_speed = _get_wind_value(current)
    source_time = current.get("time")

    if temperature is None or humidity is None or wind_speed is None or not source_time:
        return None

    return WeatherFeatures(
        temperature=float(temperature),
        relative_humidity=float(humidity),
        wind_speed=float(wind_speed),
        source="open-meteo-current",
        source_time=str(source_time),
        latitude=float(lat),
        longitude=float(lon),
        timezone=str(data.get("timezone", "")),
    )


def nearest_hour_index(times: list[str], target_time: datetime | None) -> int:
    """Return the nearest hourly index to target_time, or latest hour when target is omitted."""
    if not times:
        raise ValueError("No hourly timestamps available in Open-Meteo response.")

    if target_time is None:
        return 0

    parsed_times = [datetime.fromisoformat(str(value)) for value in times]
    target_naive = target_time.replace(tzinfo=None)
    deltas = [abs((time_value.replace(tzinfo=None) - target_naive).total_seconds()) for time_value in parsed_times]
    return int(min(range(len(deltas)), key=deltas.__getitem__))


def weather_from_hourly(
    data: dict[str, Any],
    lat: float,
    lon: float,
    target_time: datetime | None,
) -> WeatherFeatures:
    """Extract model-ready weather features from the nearest hourly forecast row."""
    hourly = data.get("hourly")
    if not isinstance(hourly, dict):
        raise ValueError("Open-Meteo response does not contain an hourly block.")

    times = [str(value) for value in hourly.get("time", [])]
    index = nearest_hour_index(times, target_time)

    temperature_values = hourly.get("temperature_2m", [])
    humidity_values = hourly.get("relative_humidity_2m", [])
    wind_values = hourly.get("wind_speed_10m", hourly.get("windspeed_10m", []))

    try:
        temperature = temperature_values[index]
        humidity = humidity_values[index]
        wind_speed = wind_values[index]
        source_time = times[index]
    except IndexError as exc:
        raise ValueError("Hourly Open-Meteo arrays have inconsistent lengths.") from exc

    if temperature is None or humidity is None or wind_speed is None:
        raise ValueError("Nearest Open-Meteo hourly row contains missing weather values.")

    return WeatherFeatures(
        temperature=float(temperature),
        relative_humidity=float(humidity),
        wind_speed=float(wind_speed),
        source="open-meteo-hourly",
        source_time=str(source_time),
        latitude=float(lat),
        longitude=float(lon),
        timezone=str(data.get("timezone", "")),
    )


def fetch_live_weather_features(
    lat: float,
    lon: float,
    target_time: datetime | None = None,
    timezone: str = DEFAULT_TIMEZONE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    min_request_interval_seconds: float | None = None,
) -> WeatherFeatures:
    """Fetch live weather and return feature names expected by the ML pipeline.

    When target_time is omitted, the API current-conditions block is preferred.
    When target_time is provided, the nearest hourly forecast row is used so
    province-wide batch jobs can be deterministic for a selected inference time.
    """
    url = build_open_meteo_url(lat=lat, lon=lon, timezone=timezone)
    data = fetch_json(
        url,
        timeout_seconds=timeout_seconds,
        min_request_interval_seconds=min_request_interval_seconds,
    )

    if target_time is None:
        current_features = weather_from_current(data, lat=lat, lon=lon)
        if current_features is not None:
            return current_features

    return weather_from_hourly(data, lat=lat, lon=lon, target_time=target_time)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Open-Meteo weather features for wildfire inference.")
    parser.add_argument("--lat", type=float, required=True, help="Latitude in EPSG:4326.")
    parser.add_argument("--lon", type=float, required=True, help="Longitude in EPSG:4326.")
    parser.add_argument(
        "--target-time",
        default=None,
        help="Optional ISO datetime. Uses nearest hourly forecast row instead of current conditions.",
    )
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE, help="Open-Meteo timezone parameter.")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--min-request-interval-seconds",
        type=float,
        default=None,
        help="Minimum spacing between Open-Meteo calls from this process. Default uses environment/config fallback.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_time = parse_iso_time(args.target_time)
    weather = fetch_live_weather_features(
        lat=args.lat,
        lon=args.lon,
        target_time=target_time,
        timezone=args.timezone,
        timeout_seconds=args.timeout_seconds,
        min_request_interval_seconds=args.min_request_interval_seconds,
    )
    print(json.dumps(asdict(weather), indent=2))


if __name__ == "__main__":
    main()
