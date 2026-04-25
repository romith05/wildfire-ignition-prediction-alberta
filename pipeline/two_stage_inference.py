import argparse
import json
import numpy as np

from src.data.npz_loader import load_npz_patch
from src.models.loaders import load_keras_model


class TwoStageWildfirePipeline:
    def __init__(self, model_a, model_b, threshold_a=0.30, threshold_b=0.25):
        self.model_a = model_a
        self.model_b = model_b
        self.threshold_a = threshold_a
        self.threshold_b = threshold_b

    def predict_patch(self, patch):
        patch = np.asarray(patch).astype("float32")

        if patch.shape != (64, 64, 17):
            raise ValueError(f"Expected patch shape (64, 64, 17), got {patch.shape}")

        b_pred = self.model_b.predict(patch[np.newaxis, ...], verbose=0)[0]
        b_conf = float(np.max(b_pred))

        if b_conf < self.threshold_b:
            return {
                "passed_gate": False,
                "stage_b_confidence": b_conf,
                "stage_a_confidence": 0.0,
                "predicted_pixels": 0,
                "mask": np.zeros((64, 64), dtype=np.uint8),
            }

        a_pred = self.model_a.predict(patch[np.newaxis, ...], verbose=0)[0]

        if a_pred.ndim == 3:
            prob_map = a_pred[:, :, 0]
        else:
            prob_map = a_pred

        mask = (prob_map >= self.threshold_a).astype(np.uint8)

        return {
            "passed_gate": True,
            "stage_b_confidence": b_conf,
            "stage_a_confidence": float(np.max(prob_map)),
            "predicted_pixels": int(mask.sum()),
            "mask": mask,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--patch", required=True)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--threshold-a", type=float, default=0.30)
    parser.add_argument("--threshold-b", type=float, default=0.25)

    args = parser.parse_args()

    channel_stats = None
    if args.stats:
        with open(args.stats, "r") as f:
            channel_stats = json.load(f)

    model_a = load_keras_model(args.model_a)
    model_b = load_keras_model(args.model_b)

    patch, label, feature_keys = load_npz_patch(
        args.patch,
        label_key="class",
        channel_stats=channel_stats,
    )

    pipeline = TwoStageWildfirePipeline(
        model_a=model_a,
        model_b=model_b,
        threshold_a=args.threshold_a,
        threshold_b=args.threshold_b,
    )

    result = pipeline.predict_patch(patch)

    print("Feature channels:", len(feature_keys))
    print("Passed gate:", result["passed_gate"])
    print("Stage B confidence:", result["stage_b_confidence"])
    print("Stage A confidence:", result["stage_a_confidence"])
    print("Predicted ignition pixels:", result["predicted_pixels"])

    if label is not None:
        print("True ignition pixels:", int(label.sum()))


if __name__ == "__main__":
    main()
