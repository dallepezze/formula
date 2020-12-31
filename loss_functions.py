"""Loss functions."""
import numpy as np
import tensorflow as tf

EPS = tf.keras.backend.epsilon()

def cast_labels(y_true, y_pred):
    return tf.cast(y_true, dtype=tf.float32), tf.cast(y_pred, dtype=tf.float32) + EPS


def get_WFL(alpha, gamma):
    """
    Returns the Weighted Focal Loss (WFL)

    Args:
        alpha: weight parameter to compensate class imbalance.
        gamma: defines the rate at which easy examples are down-weighted: 
               the higher ,the wider the range in which an example receives low loss.

    Example:
        alpha = 0.8
        gamma = 2.0
        loss_function = get_WFL(alpha, gamma)
    """
    def WFL(y_true, y_pred):
        y_true, y_pred = cast_labels(y_true, y_pred)
        loss = -alpha * tf.math.pow(1 - y_pred, gamma) * \
            y_true * tf.math.log(y_pred) - \
            (1 - alpha) * tf.math.pow(1 - (1 - y_pred), gamma) * \
            (1 - y_true) * tf.math.log(1 - y_pred)
        return loss
    return WFL


