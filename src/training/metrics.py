import tensorflow as tf


def pixel_precision(threshold=0.5):
    def metric(y_true, y_pred):
        y_true = tf.cast(y_true > 0.5, tf.float32)
        y_pred = tf.cast(y_pred >= threshold, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)

        return tp / (tp + fp + 1e-7)

    return metric


def pixel_recall(threshold=0.5):
    def metric(y_true, y_pred):
        y_true = tf.cast(y_true > 0.5, tf.float32)
        y_pred = tf.cast(y_pred >= threshold, tf.float32)

        tp = tf.reduce_sum(y_true * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))

        return tp / (tp + fn + 1e-7)

    return metric


def pixel_fp_rate(threshold=0.5):
    def metric(y_true, y_pred):
        y_true = tf.cast(y_true > 0.5, tf.float32)
        y_pred = tf.cast(y_pred >= threshold, tf.float32)

        fp = tf.reduce_sum((1 - y_true) * y_pred)
        tn = tf.reduce_sum((1 - y_true) * (1 - y_pred))

        return fp / (fp + tn + 1e-7)

    return metric
