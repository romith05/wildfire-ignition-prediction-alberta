"""Download current Alberta active-fire points from the ArcGIS FeatureServer.

The script writes both GeoJSON and CSV outputs so the records can be inspected
manually and passed directly to the live-prediction validation utility.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_SERVICE_URL = (
    "https://services.arcgis.com/wjcPoefzjpzCgffS/ArcGIS/rest/services/"
    "activefires/FeatureServer/0/query"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Alberta active-fire data from the ArcGIS FeatureServer."
    )
    parser.add_argument(
        "--service-url",
        default=DEFAULT_SERVICE_URL,
        help="ArcGIS layer query endpoint.",
    )
    parser.add_argument(
        "--output-geojson",
        default="data/validation/alberta_activefires_current.geojson",
        help="Output GeoJSON path.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/validation/alberta_activefires_current.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--where",
        default="1=1",
        help="ArcGIS SQL where clause. Default downloads all records.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=1000,
        help="Records requested per page.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=60.0,
        help="HTTP timeout per request.",
    )
    return parser.parse_args()


def fetch_page(
    service_url: str,
    where: str,
    offset: int,
    page_size: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    params = {
        "where": where,
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
        "resultOffset": offset,
        "resultRecordCount": page_size,
    }
    url = f"{service_url}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "wildfire-validation/1.0"})

    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if "error" in payload:
        raise RuntimeError(f"ArcGIS query failed: {payload['error']}")

    if payload.get("type") != "FeatureCollection":
        raise RuntimeError("ArcGIS response was not a GeoJSON FeatureCollection.")

    return payload


def fetch_all_features(
    service_url: str,
    where: str,
    page_size: int,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    if page_size <= 0:
        raise ValueError("page_size must be greater than zero")

    features: list[dict[str, Any]] = []
    offset = 0

    while True:
        payload = fetch_page(
            service_url=service_url,
            where=where,
            offset=offset,
            page_size=page_size,
            timeout_seconds=timeout_seconds,
        )
        page_features = payload.get("features", [])
        features.extend(page_features)

        print(f"Fetched offset={offset}: {len(page_features)} records")

        if len(page_features) < page_size:
            break
        offset += page_size

    return features


def feature_to_row(feature: dict[str, Any]) -> dict[str, Any]:
    row = dict(feature.get("properties") or {})
    geometry = feature.get("geometry") or {}
    coordinates = geometry.get("coordinates") or [None, None]

    if len(coordinates) >= 2:
        row["longitude"] = coordinates[0]
        row["latitude"] = coordinates[1]
    else:
        row["longitude"] = None
        row["latitude"] = None

    return row


def write_outputs(
    features: list[dict[str, Any]],
    output_geojson: Path,
    output_csv: Path,
) -> None:
    output_geojson.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    collection = {
        "type": "FeatureCollection",
        "name": "alberta_activefires_current",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }
    output_geojson.write_text(json.dumps(collection, indent=2), encoding="utf-8")

    rows = [feature_to_row(feature) for feature in features]
    if not rows:
        output_csv.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row})
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    output_geojson = Path(args.output_geojson)
    output_csv = Path(args.output_csv)

    print("Downloading Alberta active-fire data")
    print(f"Service: {args.service_url}")
    print(f"Where: {args.where}")

    features = fetch_all_features(
        service_url=args.service_url,
        where=args.where,
        page_size=args.page_size,
        timeout_seconds=args.timeout_seconds,
    )
    write_outputs(features, output_geojson, output_csv)

    print("\nActive-fire download complete")
    print(f"Features: {len(features)}")
    print(f"GeoJSON: {output_geojson}")
    print(f"CSV: {output_csv}")


if __name__ == "__main__":
    main()
