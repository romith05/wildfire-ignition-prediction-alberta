import argparse

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.data.dataset import NPZPatchDataset
from src.data.keras_generator import KerasNPZGenerator
from src.models.unet import unet_deep
from src.training.losses import focal_loss
from src.training.metrics import pixel_precision, pixel_recall, pixel_fp_rate


def train_model_b(args):
    train_dataset = NPZPatchDataset(
        folder=args.train_dir,
        label_key="class",
        channel_stats=None,
    )

    val_dataset = NPZPatchDataset(
        folder=args.val_dir,
        label_key="class",
        channel_stats=None,
    )

    train_gen = KerasNPZGenerator(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_gen = KerasNPZGenerator(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = unet_deep(input_shape=(64, 64, 17))

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
    parser.add_argument("--output-model", default="models/model_B_gatekeeper.keras")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)

    args = parser.parse_args()
    train_model_b(args)


if __name__ == "__main__":
    main()
