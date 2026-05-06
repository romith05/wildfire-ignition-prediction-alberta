import tensorflow as tf


def focal_loss(alpha=0.90, gamma=2.50):
    """Pixel-wise focal loss for sparse wildfire ignition masks."""

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)

        ce = -(
            alpha * y_true * tf.math.log(y_pred)
            + (1.0 - alpha) * (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        )
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        fl = tf.pow(1.0 - pt, gamma) * ce
        return tf.reduce_mean(fl)

    loss.__name__ = f"focal_loss_alpha_{alpha}_gamma_{gamma}"
    return loss


def patch_level_fp_loss(y_true, y_pred):
    """Binary cross-entropy on patch-level ignition presence.

    This suppresses patches that activate anywhere when the ground-truth mask has
    no ignition pixels. It matches the notebook's Model B gatekeeper objective.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)

    y_pred_any = tf.reduce_max(y_pred, axis=[1, 2, 3])
    y_true_any = tf.reduce_max(y_true, axis=[1, 2, 3])

    bce = -(
        y_true_any * tf.math.log(y_pred_any)
        + (1.0 - y_true_any) * tf.math.log(1.0 - y_pred_any)
    )
    return tf.reduce_mean(bce)


def combined_loss(alpha=0.90, gamma=2.50, lambda_patch=0.50):
    """Focal loss plus weighted patch-level suppression loss."""
    focal = focal_loss(alpha=alpha, gamma=gamma)

    def loss(y_true, y_pred):
        return focal(y_true, y_pred) + lambda_patch * patch_level_fp_loss(y_true, y_pred)

    loss.__name__ = f"combined_loss_alpha_{alpha}_gamma_{gamma}_lambda_{lambda_patch}"
    return loss
