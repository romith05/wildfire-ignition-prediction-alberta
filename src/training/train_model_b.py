import argparse
import json
import math
from pathlib import Path

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from src.data.dataset import NPZPatchDataset
from src.data.keras_generator import KerasNPZGenerator
from src.models.unet import unet_deep
from src.training.losses import combined_loss, focal_loss
from src.training.metrics import pixel_fp_rate, patch_fp_rate


class MixedNPZPatchDataset:
    """Dataset wrapper that mixes balanced data with repeated hard negatives.

    The hard-negative ratio is approximate and implemented by repeating hard
    negative samples until they make up the requested fraction of the mixed
    training list.

    Example:
        hard_negative_ratio=0.10 means roughly 10% of one epoch comes from
        hard negatives and 90% comes from the base balanced dataset.
    """

    def __init__(self, base_dataset, hard_negative_datasets=None, hard_negative_ratio=0.0):
        self.base_dataset = base_dataset
        self.hard_negative_datasets = hard_negative_datasets or []
        self.hard_negative_ratio = float(hard_negative_ratio)
        self.C = base_dataset.C
        self.feature_keys = base_dataset.feature_keys

        self.samples = [("base", 0, idx) for idx in range(len(base_dataset))]

        if self.hard_negative_ratio > 0 and self.hard_negative_datasets:
            self._validate_hard_negative_datasets()
            self.samples.extend(self._build_hard_negative_samples())

        self.valid_files = self._build_valid_file_list()

        print("Mixed Model B training dataset")
        print(f"Base samples: {len(base_dataset)}")
        print(f"Hard-negative folders: {len(self.hard_negative_datasets)}")
        print(f"Requested hard-negative ratio: {self.hard_negative_ratio:.3f}")
        print(f"Total mixed samples: {len(self.samples)}")
        print(f"Effective hard-negative samples: {self._count_hard_samples()}")

    def _validate_hard_negative_datasets(self):
        for dataset in self.hard_negative_datasets:
            if dataset.C != self.C:
                raise ValueError(
                    f"Hard-negative channel count {dataset.C} does not match base channel count {self.C}."
                )
            if dataset.feature_keys != self.feature_keys:
                raise ValueError(
                    "Hard-negative feature keys do not match base dataset feature keys."
                )

    def _build_hard_negative_samples(self):
        hard_refs = []
        for dataset_idx, dataset in enumerate(self.hard_negative_datasets):
            for sample_idx in range(len(dataset)):
                hard_refs.append(("hard", dataset_idx, sample_idx))

        if not hard_refs:
            return []

        if self.hard_negative_ratio <= 0:
            return []
        if self.hard_negative_ratio >= 1:
            raise ValueError("hard_negative_ratio must be < 1.0")

        base_count = len(self.base_dataset)
        target_hard_count = math.ceil((self.hard_negative_ratio * base_count) / (1.0 - self.hard_negative_ratio))
        repeats = math.ceil(target_hard_count / len(hard_refs))
        expanded = (hard_refs * repeats)[:target_hard_count]
        return expanded

    def _build_valid_file_list(self):
        files = list(self.base_dataset.valid_files)
        for dataset in self.hard_negative_datasets:
            files.extend(dataset.valid_files)
        return files

    def _count_hard_samples(self):
        return sum(1 for sample in self.samples if sample[0] == "hard")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        source, dataset_idx, sample_idx = self.samples[idx]
        if source == "base":
            return self.base_dataset[sample_idx]
        return self.hard_negative_datasets[dataset_idx][sample_idx]


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


def phase_model_path(output_model, phase_number):
    """Return a stable file path for a phase-specific checkpoint."""
    output_path = Path(output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.with_name(f"{output_path.stem}_phase{phase_number}{output_path.suffix}")


def checkpoint_callback(output_model):
    """Save the best model for the current phase."""
    return ModelCheckpoint(
        output_model,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )


def compile_for_phase(model, loss_fn, learning_rate):
    """Compile one phase with an explicit objective and learning rate."""
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


def fit_phase(model, train_gen, val_gen, args, phase_number, phase_name, phase_goal, epochs, output_path):
    """Run one fixed Model B training phase and save its own model."""
    print(f"\nPhase {phase_number}: {phase_name}")
    print(f"Goal: {phase_goal}")
    print(f"Training samples this phase: {len(train_gen.dataset)}")
    print(f"Saving best phase model to: {output_path}")

    model.fit(
        train_gen,
        steps_per_epoch=math.ceil(len(train_gen.dataset) / args.batch_size),
        validation_data=val_gen,
        validation_steps=math.ceil(len(val_gen.dataset) / args.batch_size),
        epochs=epochs,
        callbacks=[checkpoint_callback(output_path)],
        verbose=1,
    )

    model.save(output_path)
    print(f"Saved phase {phase_number} model to {output_path}")


def make_phase_train_generator(base_dataset, hard_negative_datasets, batch_size, hard_negative_ratio):
    """Create a phase-specific training generator."""
    if hard_negative_ratio > 0 and hard_negative_datasets:
        dataset = MixedNPZPatchDataset(
            base_dataset=base_dataset,
            hard_negative_datasets=hard_negative_datasets,
            hard_negative_ratio=hard_negative_ratio,
        )
    else:
        dataset = base_dataset
        print("Mixed Model B training dataset")
        print(f"Base samples: {len(base_dataset)}")
        print("Hard-negative folders: 0 used in this phase")
        print("Requested hard-negative ratio: 0.000")
        print(f"Total mixed samples: {len(dataset)}")

    return KerasNPZGenerator(dataset, batch_size=batch_size, shuffle=True)


def train_phased_model_b(model, base_train_dataset, hard_negative_datasets, val_gen, args):
    """Run the four-phase patch-gatekeeper regimen.

    Phase goals:
    1. preserve ignition sensitivity from the base U-Net,
    2. introduce weak patch-level suppression with light hard-negative exposure,
    3. aggressively reduce no-ignition false activations with more hard negatives,
    4. recover recall while keeping suppression on the balanced base distribution.
    """
    phases = [
        {
            "number": 1,
            "name": "focal-only sensitivity warm start",
            "goal": "Preserve base ignition recall and adapt the base model to mixed 1 km data.",
            "epochs": args.phase1_epochs,
            "learning_rate": args.phase1_lr,
            "hard_negative_ratio": 0.0,
            "loss": focal_loss(alpha=0.90, gamma=2.50),
        },
        {
            "number": 2,
            "name": "weak patch suppression with light hard negatives",
            "goal": "Begin reducing no-ignition patch activations while introducing a small hard-negative fraction.",
            "epochs": args.phase2_epochs,
            "learning_rate": args.phase2_lr,
            "hard_negative_ratio": args.phase2_hard_negative_ratio,
            "loss": combined_loss(alpha=0.90, gamma=2.50, lambda_patch=args.phase2_lambda_patch),
        },
        {
            "number": 3,
            "name": "strong patch suppression with hard negatives",
            "goal": "Push down false positive no-ignition patches using stronger suppression and more hard negatives.",
            "epochs": args.phase3_epochs,
            "learning_rate": args.phase3_lr,
            "hard_negative_ratio": args.phase3_hard_negative_ratio,
            "loss": combined_loss(alpha=0.90, gamma=2.50, lambda_patch=args.phase3_lambda_patch),
        },
        {
            "number": 4,
            "name": "recall recovery and calibration",
            "goal": "Return to the balanced base distribution to recover recall while retaining patch-level discipline.",
            "epochs": args.phase4_epochs,
            "learning_rate": args.phase4_lr,
            "hard_negative_ratio": args.phase4_hard_negative_ratio,
            "loss": combined_loss(alpha=0.90, gamma=2.50, lambda_patch=args.phase4_lambda_patch),
        },
    ]

    for phase in phases:
        output_path = phase_model_path(args.output_model, phase["number"])
        train_gen = make_phase_train_generator(
            base_dataset=base_train_dataset,
            hard_negative_datasets=hard_negative_datasets,
            batch_size=args.batch_size,
            hard_negative_ratio=phase["hard_negative_ratio"],
        )
        compile_for_phase(
            model,
            loss_fn=phase["loss"],
            learning_rate=phase["learning_rate"],
        )
        fit_phase(
            model=model,
            train_gen=train_gen,
            val_gen=val_gen,
            args=args,
            phase_number=phase["number"],
            phase_name=phase["name"],
            phase_goal=phase["goal"],
            epochs=phase["epochs"],
            output_path=output_path,
        )


def load_hard_negative_datasets(paths, label_key, channel_stats):
    """Load zero or more hard-negative folders."""
    datasets = []
    if not paths:
        return datasets

    for path in paths:
        print(f"Loading Model B hard negatives from: {path}")
        datasets.append(
            NPZPatchDataset(
                folder=path,
                label_key=label_key,
                channel_stats=channel_stats,
            )
        )
    return datasets


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
    hard_negative_datasets = load_hard_negative_datasets(
        paths=args.hard_negative_dir,
        label_key=args.label_key,
        channel_stats=channel_stats,
    )

    input_shape = infer_input_shape(train_dataset)
    val_input_shape = infer_input_shape(val_dataset)
    if val_input_shape != input_shape:
        raise ValueError(
            f"Train/val input shape mismatch. Train: {input_shape}, Val: {val_input_shape}"
        )

    for dataset in hard_negative_datasets:
        hard_input_shape = infer_input_shape(dataset)
        if hard_input_shape != input_shape:
            raise ValueError(
                f"Hard-negative input shape mismatch. Base: {input_shape}, Hard negatives: {hard_input_shape}"
            )

    if args.patch_size is not None and args.patch_size != input_shape[0]:
        print(
            f"Warning: --patch-size={args.patch_size} but detected patch height={input_shape[0]}. "
            "Using detected input shape."
        )

    val_gen = KerasNPZGenerator(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Model B input shape: {input_shape}")
    print(f"Model B patch resolution: {args.resolution_m} m")
    print("Model B training regimen: phased patch gatekeeper with optional hard-negative mixing")
    print(f"Hard-negative datasets loaded: {len(hard_negative_datasets)}")

    model = build_model(input_shape=input_shape, base_model=args.base_model)
    train_phased_model_b(model, train_dataset, hard_negative_datasets, val_gen, args)

    model.save(args.output_model)
    print(f"Saved final Model B to {args.output_model}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--output-model", default="models/model_B_1km_gatekeeper.keras")
    parser.add_argument("--base-model", required=True, help="Saved ignition-only base Model A used to initialize Model B.")
    parser.add_argument(
        "--hard-negative-dir",
        action="append",
        default=None,
        help="Optional hard-negative folder. Can be passed multiple times.",
    )
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

    parser.add_argument("--phase1-epochs", type=int, default=3)
    parser.add_argument("--phase2-epochs", type=int, default=5)
    parser.add_argument("--phase3-epochs", type=int, default=5)
    parser.add_argument("--phase4-epochs", type=int, default=3)

    parser.add_argument("--phase1-lr", type=float, default=1e-4)
    parser.add_argument("--phase2-lr", type=float, default=7.5e-5)
    parser.add_argument("--phase3-lr", type=float, default=5e-5)
    parser.add_argument("--phase4-lr", type=float, default=3e-5)

    parser.add_argument("--phase2-lambda-patch", type=float, default=0.10)
    parser.add_argument("--phase3-lambda-patch", type=float, default=0.50)
    parser.add_argument("--phase4-lambda-patch", type=float, default=0.30)

    parser.add_argument("--phase2-hard-negative-ratio", type=float, default=0.10)
    parser.add_argument("--phase3-hard-negative-ratio", type=float, default=0.25)
    parser.add_argument("--phase4-hard-negative-ratio", type=float, default=0.00)

    args = parser.parse_args()
    train_model_b(args)


if __name__ == "__main__":
    main()
