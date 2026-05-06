import argparse
import json

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.data.dataset import NPZPatchDataset
from src.data.keras_generator import KerasNPZGenerator
from src.models.unet import unet_deep
from src.training.losses import focal_loss
from src.training.metrics import pixel_precision, pixel_recall, pixel_fp_rate


def load_channel_stats(path):
    """Load channel normalization statistics from JSON.

    The JSON should contain at least:
    - mean: list of per-channel means
    - std: list of per-channel standard deviations
    """
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


def train_model_b(args):
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

    train_gen = KerasNPZGenerator(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_gen = KerasNPZGenerator(val_dataset, batch_size=args.batch_size, shuffle=False)

    input_shape = (args.patch_size, args.patch_size, train_dataset.C)
    print(f"Model B input shape: {input_shape}")
    print(f"Model B patch resolution: {args.resolution_m} m")

    model = unet_deep(input_shape=input_shape)

    model.compile(
        optimizer="adam",
        loss=focal_loss(alpha=0.85, gamma=2.00),
        metrics=[
            pixel_precision(0.5),
            pixel_recall(0.5),
            pixel_fp_rate(0.5),
        ],
    )

    callbacks = [
        ModelCheckpoint(
            args.output_model,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(args.output_model)
    print(f"Saved Model B to {args.output_model}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-model", default="models/model_B_1km_gatekeeper.keras")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--channel-stats", default=None, help="Optional channel_stats.json path.")
    parser.add_argument("--label-key", default="class", help="NPZ label/mask key.")
    parser.add_argument("--patch-size", type=int, default=64, help="Patch height/width in pixels.")
    parser.add_argument("--resolution-m", type=int, default=1000, help="Patch pixel resolution in metres.")

    args = parser.parse_args()
    train_model_b(args)


if __name__ == "__main__":
    main()
