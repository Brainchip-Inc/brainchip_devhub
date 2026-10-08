# Copyright 2026 Brainchip Holdings Ltd.  Apache 2.0 License
"""Training helpers shared across model zoo examples."""
import tensorflow as tf
import tf_keras
from tf_keras.regularizers import Regularizer


@tf_keras.utils.register_keras_serializable(package="Custom", name="HoyerSquare")
class HoyerSquare(Regularizer):
    """Hoyer-square activity regularizer: strength * ||x||_1^2 / ||x||_2^2.

    Promotes sparse activations by minimizing the ratio of the squared L1 norm to the
    squared L2 norm, which is scale-invariant: it rewards having few non-zero values
    rather than small values. Applied per activation tensor; on Akida, sparser
    activations mean less computation and lower latency and energy.

    Args:
        strength (float): weight of the penalty in the training loss.
    """

    def __init__(self, strength=1e-4):
        self.strength = float(strength)

    def __call__(self, x):
        l1 = tf.reduce_sum(tf.abs(x))
        l2sq = tf.reduce_sum(tf.square(x)) + 1e-8  # epsilon avoids div-by-zero on zero tensors
        return self.strength * (l1 ** 2) / l2sq

    def get_config(self):
        return {"strength": self.strength}
