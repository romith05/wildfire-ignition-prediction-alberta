"""Train an ignition-only base U-Net before Model B phased training.

This script captures the base-model step from the original notebook workflow:
1. train a U-Net on ignition-only patches,
2. save that base model,
3. later initialize Model B from this saved model before mixed/phased training.

For the new coarse-to-fine architecture, use this first on 1 km ignition patches
before training the 1 km Model B gatekeeper.
"""

from __future__ import annotations

import argparse
import json

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from src.data.dataset import NPZPatchDataset
from src.data.keras_generator import KerasNPZGenerator
from src.models.unet import unet_deep
from src.training.losses import focal_loss
from src.training.metrics import pixel_precision, pixel_recall, pixel_fp_rate


def load_channel_stats(path: str | None):
    """Load channel normalization statistics from JSON, if provided."""
    if path is None:
        return None

    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    if "mean" not in stats or "std" not in stats:
        raise ValueError(f"Channel stats file must contain 'mean' and 'std': {path}")

    if len(stats["mean"]) != len(stats["std"]):
        raise ValueError(f"Mean/std length mismatch in channel stats file: {path}")

    print(f"Loaded channel stats from {path}")
    print(f"Channels in stats: {len(stats['mean'])}")
    return stats


def infer_input_shape(dataset: NPZPatchDataset) -> tuple[int, int, int]:
    """Infer model input shape directly from the first dataset sample."""
    sample_x, _ = dataset[0]
    if sample_x.ndim != 3:
        raise ValueError(f"Expected sample X shape (H, W, C), got {sample_x.shape}")
    return tuple(sample_x.shape)


def build_optimizer(learning_rate: float, clipnorm: float | None):
    """Build Adam optimizer matching the base notebook, with optional clipping."""
    if clipnorm is None or clipnorm <= 0:
        return Adam(learning_rate=learning_rate)
    return Adam(learning_rate=learning_rate, clipnorm=clipnorm)


def build_callbacks(output_model: str):
    """Callbacks for stable base-model training."""
    return [
        ModelCheckpoint(
            output_model,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=1,
        ),
    ]


def train_model_a_base(args: argparse.Namespace) -> None:
    """Train and save the ignition-only base U-Net."""
    channel_stats = load_channel_stats(args.channel_stats)

    train_dataset = NPZPatchDataset(
        folder=args.train_dir,
        label_key=args.label_key,
        channel_stats=channel_stats,
    )
    val_dataset = NPZPatchDataset(
        folder=args.val_dir,
        label_key=args.label_key,
        channel_stats=channel_stats,
    )

    input_shape = infer_input_shape(train_dataset)
    val_input_shape = infer_input_shape(val_dataset)
    if val_input_shape != input_shape:
        raise ValueError(
            f"Train/val input shape mismatch. Train: {input_shape}, Val: {val_input_shape}"
        )

    if args.patch_size is not None and args.patch_size != input_shape[0]:
        print(
            f"Warning: --patch-size={args.patch_size} but detected patch height={input_shape[0]}. "
            "Using detected input shape."
        )

    print(f"Base Model A input shape: {input_shape}")
    print(f"Base Model A patch resolution: {args.resolution_m} m")
    print("Training mode: ignition-only base U-Net")

    train_gen = KerasNPZGenerator(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_gen = KerasNPZGenerator(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = unet_deep(input_shape=input_shape)
    model.compile(
        optimizer=build_optimizer(args.learning_rate, args.clipnorm),
        loss=focal_loss(alpha=args.focal_alpha, gamma=args.focal_gamma),
        metrics=[
            pixel_precision(0.5),
            pixel_recall(0.5),
            pixel_fp_rate(0.5),
        ],
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=build_callbacks(args.output_model),
    )

    model.save(args.output_model)
    print(f"Saved ignition-only base Model A to {args.output_model}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ignition-only base U-Net for later Model B initialization."
    )
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-model", default="models/model_A_1km_base_unet.keras")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--channel-stats", default=None, help="Optional channel_stats.json path.")
    parser.add_argument("--label-key", default="class", help="NPZ label/mask key.")
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Optional expected patch height/width. The actual model shape is inferred from data.",
    )
    parser.add_argument("--resolution-m", type=int, default=1000, help="Patch pixel resolution in metres.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--focal-alpha", type=float, default=0.80)
    parser.add_argument("--focal-gamma", type=float, default=2.00)
    parser.add_argument(
        "--clipnorm",
        type=float,
        default=0.0,
        help="Optional Adam clipnorm. Default 0 disables clipping to match the base notebook.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model_a_base(args)


if __name__ == "__main__":
    main()
