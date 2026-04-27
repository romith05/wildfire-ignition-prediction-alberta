import os
import numpy as np
import matplotlib.pyplot as plt

from src.data.dataset import NPZPatchDataset
from src.models.loaders import load_keras_model
from pipeline.two_stage_inference import TwoStageWildfirePipeline


def visualize_patch(patch, mask, prob_map=None, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Show first channel as base (e.g., DEM or landcover proxy)
    base = patch[:, :, 0]

    axes[0].imshow(base, cmap="gray")
    axes[0].set_title("Input Patch")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="hot")
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    if prob_map is not None:
        axes[2].imshow(prob_map, cmap="hot")
        axes[2].set_title("Probability Map")
        axes[2].axis("off")
    else:
        axes[2].axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def run_visualization(args):
    dataset = NPZPatchDataset(args.data_dir)

    model_a = load_keras_model(args.model_a)
    model_b = load_keras_model(args.model_b)

    pipeline = TwoStageWildfirePipeline(
        model_a=model_a,
        model_b=model_b,
        threshold_a=args.threshold_a,
        threshold_b=args.threshold_b,
    )

    for i in range(min(args.num_samples, len(dataset))):
        X, y = dataset[i]

        result = pipeline.predict_patch(X)

        if result["passed_gate"]:
            prob_map = model_a.predict(X[np.newaxis, ...], verbose=0)[0][:, :, 0]
        else:
            prob_map = None

        save_path = f"{args.output_dir}/sample_{i}.png"

        visualize_patch(
            patch=X,
            mask=result["mask"],
            prob_map=prob_map,
            save_path=save_path,
        )

        print(f"Saved: {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--output-dir", default="results/sample_outputs")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--threshold-a", type=float, default=0.30)
    parser.add_argument("--threshold-b", type=float, default=0.25)

    args = parser.parse_args()

    run_visualization(args)
