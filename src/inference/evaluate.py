import argparse
import numpy as np
from tqdm import tqdm

from src.data.dataset import NPZPatchDataset
from src.models.loaders import load_keras_model
from pipeline.two_stage_inference import TwoStageWildfirePipeline


def evaluate(args):
    dataset = NPZPatchDataset(
        folder=args.test_dir,
        label_key="class",
        channel_stats=None,
    )

    model_a = load_keras_model(args.model_a)
    model_b = load_keras_model(args.model_b)

    pipeline = TwoStageWildfirePipeline(
        model_a=model_a,
        model_b=model_b,
        threshold_a=args.threshold_a,
        threshold_b=args.threshold_b,
    )

    # Pixel-level counters
    TP = FP = TN = FN = 0

    # Patch-level counters
    patch_tp = patch_fp = patch_fn = patch_tn = 0

    for i in tqdm(range(len(dataset))):
        X, y = dataset[i]

        result = pipeline.predict_patch(X)
        pred_mask = result["mask"]

        # --- Pixel-level ---
        y_flat = y.squeeze()
        pred_flat = pred_mask

        TP += np.sum((y_flat == 1) & (pred_flat == 1))
        FP += np.sum((y_flat == 0) & (pred_flat == 1))
        TN += np.sum((y_flat == 0) & (pred_flat == 0))
        FN += np.sum((y_flat == 1) & (pred_flat == 0))

        # --- Patch-level ---
        gt_patch = 1 if y_flat.sum() > 0 else 0
        pred_patch = 1 if result["passed_gate"] else 0

        if gt_patch == 1 and pred_patch == 1:
            patch_tp += 1
        elif gt_patch == 0 and pred_patch == 1:
            patch_fp += 1
        elif gt_patch == 1 and pred_patch == 0:
            patch_fn += 1
        else:
            patch_tn += 1

    # --- Metrics ---
    pixel_precision = TP / (TP + FP + 1e-7)
    pixel_recall = TP / (TP + FN + 1e-7)
    pixel_fp_rate = FP / (FP + TN + 1e-7)

    patch_recall = patch_tp / (patch_tp + patch_fn + 1e-7)
    patch_fp_rate = patch_fp / (patch_fp + patch_tn + 1e-7)

    print("\n===== FINAL METRICS =====")
    print(f"Pixel Precision: {pixel_precision:.5f}")
    print(f"Pixel Recall: {pixel_recall:.5f}")
    print(f"Pixel FP Rate: {pixel_fp_rate:.6f}")
    print(f"Patch Recall: {patch_recall:.5f}")
    print(f"Patch FP Rate: {patch_fp_rate:.5f}")

    return {
        "pixel_precision": pixel_precision,
        "pixel_recall": pixel_recall,
        "pixel_fp_rate": pixel_fp_rate,
        "patch_recall": patch_recall,
        "patch_fp_rate": patch_fp_rate,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--threshold-a", type=float, default=0.30)
    parser.add_argument("--threshold-b", type=float, default=0.25)

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
