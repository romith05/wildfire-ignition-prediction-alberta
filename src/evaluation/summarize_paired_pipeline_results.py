"""Summarize test-only paired 1 km -> 25 m pipeline results.

This reads the CSV produced by:

    src.inference.run_paired_patch_pipeline

and reports patch-level cascade behavior:
- Model B gatekeeper confusion matrix,
- final patch-level cascade confusion matrix on rows with available 25 m labels,
- Model B pass rate,
- final positive patch rate,
- Model A removal rate on Model B false-positive patches.

Important:
This script summarizes patch-level pass/fail behavior only. It does not compute
Model A pixel-level Dice/IoU/precision/recall because the paired-pipeline CSV
contains scalar patch outcomes, not full prediction masks. Use
``src.evaluation.diagnose_model_a_validation`` for pixel-wise Model A metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_bool_series(series: pd.Series, *, allow_missing: bool = False) -> pd.Series:
    """Parse bool-like CSV values.

    When allow_missing=True, blank/NaN values remain as pandas.NA. This is needed
    for 25 m labels, because rows blocked by Model B never look up a 25 m patch.
    """
    parsed = series.astype("string").str.strip().str.lower().map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "1.0": True,
            "0.0": False,
        }
    )

    if not allow_missing and parsed.isna().any():
        bad_values = sorted(series[parsed.isna()].astype(str).unique().tolist())
        raise ValueError(f"Could not parse boolean values: {bad_values}")

    return parsed.astype("boolean")


def confusion_counts(y_true: pd.Series, y_pred: pd.Series) -> dict[str, int]:
    """Return binary confusion counts after dropping missing labels."""
    valid = y_true.notna() & y_pred.notna()
    y_true = y_true[valid].astype(bool)
    y_pred = y_pred[valid].astype(bool)

    tp = int(((y_true == True) & (y_pred == True)).sum())
    fp = int(((y_true == False) & (y_pred == True)).sum())
    tn = int(((y_true == False) & (y_pred == False)).sum())
    fn = int(((y_true == True) & (y_pred == False)).sum())

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    """Compute patch-level precision, recall, fp_rate, and accuracy from counts."""
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
    """Print one patch-level confusion matrix and derived metrics."""
    metrics = metrics_from_counts(counts)

    print(f"\n{title}")
    print(f"TP: {counts['tp']}")
    print(f"FP: {counts['fp']}")
    print(f"TN: {counts['tn']}")
    print(f"FN: {counts['fn']}")
    print(f"patch_precision: {metrics['precision']:.4f}")
    print(f"patch_recall:    {metrics['recall']:.4f}")
    print(f"patch_fp_rate:   {metrics['fp_rate']:.4f}")
    print(f"patch_accuracy:  {metrics['accuracy']:.4f}")


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
    df["label_25m_positive_bool"] = parse_bool_series(df["label_25m_positive"], allow_missing=True)
    df["model_b_passed_bool"] = df["model_b_passed"].astype(int).astype(bool).astype("boolean")
    df["final_positive_bool"] = df["final_positive"].astype(int).astype(bool).astype("boolean")

    print("Paired pipeline summary")
    print(f"CSV: {csv_path}")
    print(f"rows: {len(df)}")
    print("Metric type: patch-level cascade metrics")

    print("\nStatus counts")
    print(df["status"].value_counts(dropna=False).to_string())

    total = len(df)
    model_b_passed = int((df["model_b_passed_bool"] == True).sum())
    final_positive = int((df["final_positive_bool"] == True).sum())
    completed = int((df["status"] == "completed").sum())
    missing_25m = int((df["status"] == "missing_matching_25m_patch").sum())
    rows_with_25m_label = int(df["label_25m_positive_bool"].notna().sum())

    print("\nPass rates")
    print(f"Model B passed patches: {model_b_passed}/{total} = {model_b_passed / total if total else 0:.4f}")
    print(f"Model A ran patches:    {completed}/{total} = {completed / total if total else 0:.4f}")
    print(f"Final positive patches: {final_positive}/{total} = {final_positive / total if total else 0:.4f}")
    print(f"Rows with 25 m labels:  {rows_with_25m_label}/{total}")
    print(f"Missing 25 m patches:   {missing_25m}")

    model_b_counts = confusion_counts(
        y_true=df["label_1km_positive_bool"],
        y_pred=df["model_b_passed_bool"],
    )
    print_counts_and_metrics("Model B gatekeeper vs 1 km patch labels", model_b_counts)

    rows_with_25m = df[df["label_25m_positive_bool"].notna()].copy()
    if len(rows_with_25m) > 0:
        final_counts = confusion_counts(
            y_true=rows_with_25m["label_25m_positive_bool"],
            y_pred=rows_with_25m["final_positive_bool"],
        )
        print_counts_and_metrics("Final cascade vs available 25 m patch labels", final_counts)
    else:
        print("\nFinal cascade vs available 25 m patch labels")
        print("No rows have 25 m labels. This usually means Model B blocked all patches or no 25 m matches were found.")

    passed = df[df["model_b_passed_bool"] == True].copy()
    passed_with_25m = passed[passed["label_25m_positive_bool"].notna()].copy()
    if len(passed_with_25m) > 0:
        passed_counts = confusion_counts(
            y_true=passed_with_25m["label_25m_positive_bool"],
            y_pred=passed_with_25m["final_positive_bool"],
        )
        print_counts_and_metrics("Model A patch outcome on Model-B-passed patches", passed_counts)

        model_b_false_positives = passed_with_25m[passed_with_25m["label_25m_positive_bool"] == False]
        if len(model_b_false_positives) > 0:
            removed = int((model_b_false_positives["final_positive_bool"] == False).sum())
            total_fp_candidates = len(model_b_false_positives)
            kept = total_fp_candidates - removed
            print("\nModel A removal of Model B false-positive patches")
            print(f"Model B false-positive patch candidates: {total_fp_candidates}")
            print(f"Removed by Model A: {removed}")
            print(f"Kept as final positives: {kept}")
            print(f"Patch removal rate: {removed / total_fp_candidates:.4f}")
        else:
            print("\nModel A removal of Model B false-positive patches")
            print("No Model B false-positive patch candidates with available 25 m labels in this CSV.")

    print("\nPixel-wise Model A metrics")
    print("Not computed here. This CSV stores patch-level scalar outcomes only.")
    print("Use src.evaluation.diagnose_model_a_validation for pixel-wise Dice/IoU/precision/recall.")

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
