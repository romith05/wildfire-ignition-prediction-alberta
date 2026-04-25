import tensorflow as tf


def focal_loss(alpha=0.90, gamma=2.50):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        weight = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)

        return tf.reduce_mean(weight * tf.pow(1 - pt, gamma) * bce)

    return loss
