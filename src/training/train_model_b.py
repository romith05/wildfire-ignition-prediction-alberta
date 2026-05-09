import argparse
import json

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from src.data.dataset import NPZPatchDataset
from src.data.keras_generator import KerasNPZGenerator
from src.models.unet import unet_deep
from src.training.losses import combined_loss, focal_loss
from src.training.metrics import pixel_precision, pixel_recall, pixel_fp_rate, patch_fp_rate


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
    """Create Model B, optionally loading weights/architecture from a saved base model."""
    if base_model is None:
        print("No --base-model provided. Creating a new U-Net.")
        return unet_deep(input_shape=input_shape)

    print(f"Loading base model for Model B initialization: {base_model}")
    model = load_model(base_model, compile=False)
    loaded_shape = tuple(model.input_shape[1:])
    if loaded_shape != tuple(input_shape):
        raise ValueError(
            f"Base model input shape {loaded_shape} does not match dataset input shape {input_shape}."
        )
    return model


def build_callbacks(output_model):
    """Create checkpoint/early-stop/LR callbacks for one training phase."""
    return [
        ModelCheckpoint(output_model, monitor="val_loss", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]


def model_b_metrics(threshold=0.5):
    """Metrics used to monitor Model B gatekeeper behavior."""
    return [
        pixel_precision(threshold),
        pixel_recall(threshold),
        pixel_fp_rate(threshold),
        patch_fp_rate(threshold),
    ]


def compile_model(model, loss_fn, learning_rate):
    """Compile Model B with notebook-style Adam settings."""
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=loss_fn,
        metrics=model_b_metrics(0.5),
    )


def fit_phase(model, train_gen, val_gen, output_model, phase_name, epochs):
    """Run one named training phase."""
    if epochs <= 0:
        print(f"Skipping {phase_name}: epochs={epochs}")
        return

    print(f"\n=== {phase_name} | epochs={epochs} ===")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=build_callbacks(output_model),
    )


def train_simple_regimen(model, train_gen, val_gen, args):
    """Run one simple focal-loss training phase for quick smoke tests."""
    compile_model(
        model=model,
        loss_fn=focal_loss(alpha=args.simple_alpha, gamma=args.simple_gamma),
        learning_rate=args.learning_rate,
    )
    fit_phase(
        model=model,
        train_gen=train_gen,
        val_gen=val_gen,
        output_model=args.output_model,
        phase_name="Model B simple focal training",
        epochs=args.epochs,
    )


def train_phased_regimen(model, train_gen, val_gen, args):
    """Run the four-phase Model B regimen."""
    phases = [
        ("Phase 1: focal only", args.phase1_epochs, focal_loss(alpha=0.90, gamma=2.50)),
        (
            "Phase 2: focal + weak patch suppression",
            args.phase2_epochs,
            combined_loss(alpha=0.90, gamma=2.50, lambda_patch=0.10),
        ),
        (
            "Phase 3: focal + strong patch suppression",
            args.phase3_epochs,
            combined_loss(alpha=0.90, gamma=2.50, lambda_patch=0.50),
        ),
        (
            "Phase 4: recall-optimized suppression",
            args.phase4_epochs,
            combined_loss(alpha=0.90, gamma=2.50, lambda_patch=0.30),
        ),
    ]

    for phase_name, epochs, loss_fn in phases:
        compile_model(model=model, loss_fn=loss_fn, learning_rate=args.learning_rate)
        fit_phase(
            model=model,
            train_gen=train_gen,
            val_gen=val_gen,
            output_model=args.output_model,
            phase_name=phase_name,
            epochs=epochs,
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
    print(f"Model B training regimen: {args.training_regimen}")

    model = build_model(input_shape=input_shape, base_model=args.base_model)

    if args.training_regimen == "simple":
        train_simple_regimen(model, train_gen, val_gen, args)
    elif args.training_regimen == "phased":
        train_phased_regimen(model, train_gen, val_gen, args)
    else:
        raise ValueError(f"Unknown training regimen: {args.training_regimen}")

    model.save(args.output_model)
    print(f"Saved Model B to {args.output_model}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-model", default="models/model_B_1km_gatekeeper.keras")
    parser.add_argument("--base-model", default=None, help="Optional saved base Model A used to initialize Model B.")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for --training-regimen simple.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--channel-stats", default=None, help="Optional channel_stats.json path.")
    parser.add_argument("--label-key", default="class", help="NPZ label/mask key.")
    parser.add_argument(
        "--patch-size",
        type=int,
        default=None,
        help="Optional expected patch height/width. The actual model shape is inferred from data.",
    )
    parser.add_argument("--resolution-m", type=int, default=1000, help="Patch pixel resolution in metres.")
    parser.add_argument(
        "--training-regimen",
        choices=["simple", "phased"],
        default="phased",
        help="Use simple focal training or phased Model B training.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--simple-alpha", type=float, default=0.85)
    parser.add_argument("--simple-gamma", type=float, default=2.00)
    parser.add_argument("--phase1-epochs", type=int, default=3)
    parser.add_argument("--phase2-epochs", type=int, default=5)
    parser.add_argument("--phase3-epochs", type=int, default=5)
    parser.add_argument("--phase4-epochs", type=int, default=3)

    args = parser.parse_args()
    train_model_b(args)


if __name__ == "__main__":
    main()
