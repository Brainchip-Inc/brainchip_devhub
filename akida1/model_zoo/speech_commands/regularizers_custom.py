import tensorflow as tf
import tf_keras
from tf_keras.regularizers import Regularizer


@tf_keras.utils.register_keras_serializable(package="Custom", name="HoyerSquare")
class HoyerSquare(Regularizer):
    """Hoyer-square activity regularizer: strength * ||x||_1^2 / ||x||_2^2.

    Promotes sparse binary-like activations by minimizing the ratio of L1 to L2 norm.
    Applied per activation tensor.
    """

    def __init__(self, strength=1e-4):
        self.strength = float(strength)

    def __call__(self, x):
        l1 = tf.reduce_sum(tf.abs(x))
        l2sq = tf.reduce_sum(tf.square(x)) + 1e-8  # epsilon avoids div-by-zero on zero tensors
        return self.strength * (l1 ** 2) / l2sq

    def get_config(self):
        return {"strength": self.strength}
