"""Model loading helpers for wildfire inference scripts."""

from __future__ import annotations

from tensorflow.keras.models import load_model

from src.training.losses import focal_loss
from src.training.metrics import pixel_fp_rate, pixel_precision, pixel_recall


def load_keras_model(model_path: str):
    """Load a Keras model saved by the wildfire training scripts.

    The training scripts compile models with custom focal loss and custom pixel
    metrics, so those objects are registered here for reliable loading.
    """
    custom_objects = {
        "loss": focal_loss(),
        "metric": None,
        "pixel_precision": pixel_precision,
        "pixel_recall": pixel_recall,
        "pixel_fp_rate": pixel_fp_rate,
    }

    return load_model(model_path, custom_objects=custom_objects, compile=False)
