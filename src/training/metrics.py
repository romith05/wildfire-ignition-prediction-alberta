import tensorflow as tf


def pixel_precision(threshold=0.5):
    """Pixel-level precision at a fixed probability threshold."""

    def metric(y_true, y_pred):
        y_true = tf.cast(y_true > 0.5, tf.float32)
        y_pred = tf.cast(y_pred >= threshold, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1.0 - y_true) * y_pred)

        return tp / (tp + fp + 1e-7)

    metric.__name__ = f"pixel_precision_{threshold}"
    return metric


def pixel_recall(threshold=0.5):
    """Pixel-level recall at a fixed probability threshold."""

    def metric(y_true, y_pred):
        y_true = tf.cast(y_true > 0.5, tf.float32)
        y_pred = tf.cast(y_pred >= threshold, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred)
        fn = tf.reduce_sum(y_true * (1.0 - y_pred))

        return tp / (tp + fn + 1e-7)

    metric.__name__ = f"pixel_recall_{threshold}"
    return metric


def pixel_fp_rate(threshold=0.5):
    """Pixel-level false positive rate at a fixed probability threshold."""

    def metric(y_true, y_pred):
        y_true = tf.cast(y_true > 0.5, tf.float32)
        y_pred = tf.cast(y_pred >= threshold, tf.float32)

        fp = tf.reduce_sum((1.0 - y_true) * y_pred)
        tn = tf.reduce_sum((1.0 - y_true) * (1.0 - y_pred))

        return fp / (fp + tn + 1e-7)

    metric.__name__ = f"pixel_fp_rate_{threshold}"
    return metric


def patch_fp_rate(threshold=0.5):
    """Patch-level false positive rate at a fixed probability threshold.

    A negative patch is counted as a false positive if any pixel in the predicted
    probability mask exceeds the threshold while the ground-truth patch contains
    no ignition pixels.
    """

    def metric(y_true, y_pred):
        y_true_any = tf.cast(tf.reduce_max(y_true, axis=[1, 2, 3]) > 0.5, tf.float32)
        y_pred_any = tf.cast(tf.reduce_max(y_pred, axis=[1, 2, 3]) >= threshold, tf.float32)

        negative = 1.0 - y_true_any
        false_positive = negative * y_pred_any

        return tf.reduce_sum(false_positive) / (tf.reduce_sum(negative) + 1e-7)

    metric.__name__ = f"patch_fp_rate_{threshold}"
    return metric
