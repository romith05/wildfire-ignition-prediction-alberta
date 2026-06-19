"""Summarize prospective wildfire-validation results.

This read-only utility reports the current performance of the prospective
validation loop. It does not modify raw data, prediction outputs, model files,
rasters, NPZ archives, or validation state.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_LOG_CSV = Path("results/validation/prospective_validation_log.csv")
DEFAULT_LATEST_RUN = Path("results/validation/latest_prediction_run.txt")


def pct(numerator: int, denominator: int) -> str:
    """Return a percentage string, guarding against division by zero."""
    if denominator == 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.1f}%"


def hit_summary(df: pd.DataFrame, prefix: str, label: str) -> list[str]:
    """Build hit-rate summary lines for a prediction source prefix."""
    lines: list[str] = []
    for distance_m in [1000, 5000, 10000, 25000]:
        col = f"{prefix}_hit_within_{distance_m}m"
        if col not in df.columns:
            continue

        hits = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        hit_count = int(hits.sum())
        total = int(len(hits))
        distance_km = int(distance_m / 1000)
        lines.append(
            f"{label} hit <= {distance_km:>2} km: "
            f"{hit_count} / {total} = {pct(hit_count, total)}"
        )
    return lines


def print_distance_summary(df: pd.DataFrame, column: str, label: str) -> None:
    """Print median nearest distance for a distance column when present."""
    if column not in df.columns:
        return

    dist_km = pd.to_numeric(df[column], errors="coerce").dropna() / 1000.0
    if dist_km.empty:
        return

    print(f"{label} nearest distance median: {dist_km.median():.2f} km")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize prospective wildfire-validation results."
    )
    parser.add_argument("--validation-log-csv", type=Path, default=DEFAULT_LOG_CSV)
    parser.add_argument("--latest-run-file", type=Path, default=DEFAULT_LATEST_RUN)
    args = parser.parse_args()

    print("Prospective validation summary")
    print("=" * 40)

    if args.latest_run_file.exists():
        latest_run = args.latest_run_file.read_text(encoding="utf-8").strip()
        print(f"Latest prediction run: {latest_run}")
    else:
        print(f"Latest prediction run: missing ({args.latest_run_file})")

    if not args.validation_log_csv.exists():
        print(f"Validation log missing: {args.validation_log_csv}")
        return

    df = pd.read_csv(args.validation_log_csv)

    if df.empty:
        print("Validated fires: 0")
        return

    if "status" in df.columns:
        status = df["status"].astype(str).str.lower()
        validated = df[status.eq("validated")].copy()
    else:
        validated = df.copy()

    if validated.empty:
        print("Validated fires: 0")
        return

    print(f"Validation log: {args.validation_log_csv}")
    print(f"Validated fires: {len(validated)}")

    if "lead_time_hours" in validated.columns:
        lead = pd.to_numeric(validated["lead_time_hours"], errors="coerce").dropna()
        if not lead.empty:
            print(f"Lead time median: {lead.median():.2f} hours")
            print(f"Lead time range: {lead.min():.2f} to {lead.max():.2f} hours")

    print_distance_summary(
        validated,
        "model_b_candidate_nearest_distance_m",
        "Model B candidate",
    )
    print_distance_summary(
        validated,
        "model_a_candidate_nearest_distance_m",
        "Model A candidate",
    )
    print_distance_summary(
        validated,
        "model_a_positive_nearest_distance_m",
        "Model A positive",
    )

    print("")
    print("Hit rates")
    print("-" * 40)
    for line in hit_summary(validated, "model_b_candidate", "Model B candidate"):
        print(line)
    for line in hit_summary(validated, "model_a_candidate", "Model A candidate"):
        print(line)
    for line in hit_summary(validated, "model_a_positive", "Model A positive"):
        print(line)


if __name__ == "__main__":
    main()
