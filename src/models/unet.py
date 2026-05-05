"""U-Net model definitions for wildfire ignition segmentation.

This module preserves the U-Net architecture used in the original training
notebook while making it importable by the modular training scripts.
"""

from __future__ import annotations

from tensorflow.keras import layers, models


def unet_deep(input_shape: tuple[int, int, int]):
    """Build the deep U-Net used for 64x64 wildfire ignition patches.

    Args:
        input_shape: Input tensor shape as ``(height, width, channels)``.
            The current wildfire patch setup uses ``(64, 64, 17)``.

    Returns:
        A compiled-ready ``tf.keras.Model`` with a single sigmoid mask output.
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    c1 = layers.Conv2D(32, 3, padding="same", activation="relu")(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, padding="same", activation="relu")(p1)
    c2 = layers.Conv2D(64, 3, padding="same", activation="relu")(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(128, 3, padding="same", activation="relu")(p2)
    c3 = layers.Conv2D(128, 3, padding="same", activation="relu")(c3)
    p3 = layers.MaxPooling2D()(c3)

    # Bottleneck
    b = layers.Conv2D(256, 3, padding="same", activation="relu")(p3)
    b = layers.Conv2D(256, 3, padding="same", activation="relu")(b)

    # Decoder
    u3 = layers.UpSampling2D()(b)
    u3 = layers.Concatenate()([u3, c3])
    c4 = layers.Conv2D(128, 3, padding="same", activation="relu")(u3)
    c4 = layers.Conv2D(128, 3, padding="same", activation="relu")(c4)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = layers.Conv2D(64, 3, padding="same", activation="relu")(u2)
    c5 = layers.Conv2D(64, 3, padding="same", activation="relu")(c5)

    u1 = layers.UpSampling2D()(c5)
    u1 = layers.Concatenate()([u1, c1])
    c6 = layers.Conv2D(32, 3, padding="same", activation="relu")(u1)
    c6 = layers.Conv2D(32, 3, padding="same", activation="relu")(c6)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c6)
    return models.Model(inputs, outputs)
