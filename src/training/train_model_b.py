import argparse
import json
import math

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from src.data.dataset import NPZPatchDataset
from src.data.keras_generator import KerasNPZGenerator
from src.models.unet import unet_deep
from src.training.losses import combined_loss, focal_loss
from src.training.metrics import pixel_fp_rate, patch_fp_rate


def load_channel_stats(path):
    """Load channel normalization statistics from JSON."""
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


def infer_input_shape(dataset):
    """Infer model input shape directly from the first dataset sample."""
    sample_x, _ = dataset[0]
    if sample_x.ndim != 3:
        raise ValueError(f"Expected sample X shape (H, W, C), got {sample_x.shape}")
    return tuple(sample_x.shape)


def build_model(input_shape, base_model=None):
    """Load the ignition-only base model or create a new U-Net."""
    if base_model is None:
        print("No --base-model provided. Creating a new U-Net.")
        return unet_deep(input_shape=input_shape)

    print(f"Loading ignition-only base model: {base_model}")
    model = load_model(base_model, compile=False)
    loaded_shape = tuple(model.input_shape[1:])
    if loaded_shape != tuple(input_shape):
        raise ValueError(
            f"Base model input shape {loaded_shape} does not match dataset input shape {input_shape}."
        )
    return model


def checkpoint_callback(output_model):
    """Save the best model for the current phase."""
    return ModelCheckpoint(
        output_model,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )


def compile_for_phase(model, loss_fn, learning_rate):
    """Compile exactly for a notebook-style phase."""
    opt = Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(
        optimizer=opt,
        loss=loss_fn,
        metrics=[
            "precision",
            "recall",
            pixel_fp_rate(0.5),
            patch_fp_rate(0.5),
        ],
    )


def fit_phase(model, train_gen, val_gen, args, phase_name, epochs):
    """Run one fixed Model B training phase."""
    print(f"\n{phase_name}")
    model.fit(
        train_gen,
        steps_per_epoch=math.ceil(len(train_gen.dataset) / args.batch_size),
        validation_data=val_gen,
        validation_steps=math.ceil(len(val_gen.dataset) / args.batch_size),
        epochs=epochs,
        callbacks=[checkpoint_callback(args.output_model)],
        verbose=1,
    )


def train_phased_model_b(model, train_gen, val_gen, args):
    """Run the four-phase Model B regimen from training.ipynb."""
    compile_for_phase(
        model,
        loss_fn=focal_loss(alpha=0.90, gamma=2.50),
        learning_rate=args.learning_rate,
    )
    fit_phase(model, train_gen, val_gen, args, "Phase 1: focal only", args.phase1_epochs)

    compile_for_phase(
        model,
        loss_fn=combined_loss(alpha=0.90, gamma=2.50, lambda_patch=0.10),
        learning_rate=args.learning_rate,
    )
    fit_phase(
        model,
        train_gen,
        val_gen,
        args,
        "Phase 2: focal + weak patch suppression",
        args.phase2_epochs,
    )

    compile_for_phase(
        model,
        loss_fn=combined_loss(alpha=0.90, gamma=2.50, lambda_patch=0.50),
        learning_rate=args.learning_rate,
    )
    fit_phase(
        model,
        train_gen,
        val_gen,
        args,
        "Phase 3: focal + strong patch suppression",
        args.phase3_epochs,
    )

    compile_for_phase(
        model,
        loss_fn=combined_loss(alpha=0.90, gamma=2.50, lambda_patch=0.30),
        learning_rate=args.learning_rate,
    )
    fit_phase(
        model,
        train_gen,
        val_gen,
        args,
        "Phase 4: recall-optimized suppression",
        args.phase4_epochs,
    )


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

    train_gen = KerasNPZGenerator(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_gen = KerasNPZGenerator(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Model B input shape: {input_shape}")
    print(f"Model B patch resolution: {args.resolution_m} m")
    print("Model B training regimen: phased only")

    model = build_model(input_shape=input_shape, base_model=args.base_model)
    train_phased_model_b(model, train_gen, val_gen, args)

    model.save(args.output_model)
    print(f"Saved Model B to {args.output_model}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-model", default="models/model_B_1km_gatekeeper.keras")
    parser.add_argument("--base-model", required=True, help="Saved ignition-only base Model A used to initialize Model B.")
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
    parser.add_argument("--phase1-epochs", type=int, default=3)
    parser.add_argument("--phase2-epochs", type=int, default=5)
    parser.add_argument("--phase3-epochs", type=int, default=5)
    parser.add_argument("--phase4-epochs", type=int, default=3)

    args = parser.parse_args()
    train_model_b(args)


if __name__ == "__main__":
    main()
