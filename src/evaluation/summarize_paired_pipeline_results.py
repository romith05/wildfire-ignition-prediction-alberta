"""Summarize test-only paired 1 km -> 25 m pipeline results.

This reads the CSV produced by:

    src.inference.run_paired_patch_pipeline

and reports:
- Model B gatekeeper confusion matrix,
- final cascade confusion matrix,
- Model B pass rate,
- final positive rate,
- Model A removal rate on Model B false positives.

The expected CSV columns include:
- model_b_passed
- final_positive
- label_1km_positive
- label_25m_positive
- status
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_bool_series(series: pd.Series) -> pd.Series:
    """Parse bool-like CSV values into boolean Series."""
    if series.dtype == bool:
        return series

    return series.astype(str).str.lower().map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "1.0": True,
            "0.0": False,
        }
    )


def confusion_counts(y_true: pd.Series, y_pred: pd.Series) -> dict[str, int]:
    """Return binary confusion counts."""
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    tp = int((y_true & y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    fn = int((y_true & ~y_pred).sum())

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    """Compute precision, recall, fp_rate, and accuracy from counts."""
    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]
    total = tp + fp + tn + fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "fp_rate": fp_rate,
        "accuracy": accuracy,
    }


def print_counts_and_metrics(title: str, counts: dict[str, int]) -> None:
    """Print one confusion matrix and derived metrics."""
    metrics = metrics_from_counts(counts)

    print(f"\n{title}")
    print(f"TP: {counts['tp']}")
    print(f"FP: {counts['fp']}")
    print(f"TN: {counts['tn']}")
    print(f"FN: {counts['fn']}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall:    {metrics['recall']:.4f}")
    print(f"fp_rate:   {metrics['fp_rate']:.4f}")
    print(f"accuracy:  {metrics['accuracy']:.4f}")


def summarize(args: argparse.Namespace) -> None:
    """Summarize paired pipeline CSV."""
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {
        "model_b_passed",
        "final_positive",
        "label_1km_positive",
        "label_25m_positive",
        "status",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["label_1km_positive_bool"] = parse_bool_series(df["label_1km_positive"])
    df["label_25m_positive_bool"] = parse_bool_series(df["label_25m_positive"])
    df["model_b_passed_bool"] = df["model_b_passed"].astype(int).astype(bool)
    df["final_positive_bool"] = df["final_positive"].astype(int).astype(bool)

    print("Paired pipeline summary")
    print(f"CSV: {csv_path}")
    print(f"rows: {len(df)}")

    print("\nStatus counts")
    print(df["status"].value_counts(dropna=False).to_string())

    total = len(df)
    model_b_passed = int(df["model_b_passed_bool"].sum())
    final_positive = int(df["final_positive_bool"].sum())
    completed = int((df["status"] == "completed").sum())
    missing_25m = int((df["status"] == "missing_matching_25m_patch").sum())

    print("\nPass rates")
    print(f"Model B passed: {model_b_passed}/{total} = {model_b_passed / total if total else 0:.4f}")
    print(f"Model A ran:    {completed}/{total} = {completed / total if total else 0:.4f}")
    print(f"Final positive: {final_positive}/{total} = {final_positive / total if total else 0:.4f}")
    print(f"Missing 25 m:   {missing_25m}")

    model_b_counts = confusion_counts(
        y_true=df["label_1km_positive_bool"],
        y_pred=df["model_b_passed_bool"],
    )
    print_counts_and_metrics("Model B gatekeeper vs 1 km labels", model_b_counts)

    final_counts = confusion_counts(
        y_true=df["label_25m_positive_bool"],
        y_pred=df["final_positive_bool"],
    )
    print_counts_and_metrics("Final cascade vs 25 m labels", final_counts)

    passed = df[df["model_b_passed_bool"]].copy()
    if len(passed) > 0:
        passed_counts = confusion_counts(
            y_true=passed["label_25m_positive_bool"],
            y_pred=passed["final_positive_bool"],
        )
        print_counts_and_metrics("Model A on Model-B-passed patches", passed_counts)

        model_b_false_positives = passed[~passed["label_25m_positive_bool"]]
        if len(model_b_false_positives) > 0:
            removed = int((~model_b_false_positives["final_positive_bool"]).sum())
            total_fp_candidates = len(model_b_false_positives)
            kept = total_fp_candidates - removed
            print("\nModel A removal of Model B false positives")
            print(f"Model B false-positive candidates: {total_fp_candidates}")
            print(f"Removed by Model A: {removed}")
            print(f"Kept as final positives: {kept}")
            print(f"Removal rate: {removed / total_fp_candidates:.4f}")
        else:
            print("\nModel A removal of Model B false positives")
            print("No Model B false-positive candidates in this CSV.")

    print("\nNote")
    print("This summarizes the test-only filename-paired pipeline. It does not replace final geospatial prototype evaluation.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize paired 1 km -> 25 m pipeline CSV results.")
    parser.add_argument("--csv", required=True, help="CSV produced by run_paired_patch_pipeline.py")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summarize(args)


if __name__ == "__main__":
    main()
