import tensorflow as tf
import tf_keras
from tf_keras.regularizers import Regularizer


@tf_keras.utils.register_keras_serializable(package="Custom", name="HoyerSquare")
class HoyerSquare(Regularizer):
    """Hoyer-square activity regularizer: strength * ||x||_1^2 / ||x||_2^2.

    Promotes sparse binary-like activations by minimizing the ratio of L1 to L2 norm.
    Applied per activation tensor.

    Raw form is unbounded: for a tensor of N elements the ratio ranges up to N
    itself (dense/uniform case), so the same `strength` exerts more pressure on
    layers with more elements. With `normalize=True`, divides by N so the
    penalty is bounded in (0, 1] regardless of tensor size -- see
    ../vww/SPARSITY_EXPERIMENT.md, ../plant_village/SPARSITY_EXPERIMENT.md and
    ../speech_commands/SPARSITY_EXPERIMENT.md for why this matters in practice.
    """

    def __init__(self, strength=1e-4, normalize=False):
        self.strength = float(strength)
        self.normalize = bool(normalize)

    def __call__(self, x):
        l1 = tf.reduce_sum(tf.abs(x))
        l2sq = tf.reduce_sum(tf.square(x)) + 1e-8  # epsilon avoids div-by-zero on zero tensors
        hoyer_sq = (l1 ** 2) / l2sq
        if self.normalize:
            n = tf.cast(tf.size(x), tf.float32)
            hoyer_sq = hoyer_sq / n
        return self.strength * hoyer_sq

    def get_config(self):
        return {"strength": self.strength, "normalize": self.normalize}


@tf_keras.utils.register_keras_serializable(package="Custom", name="L1L2Activity")
class L1L2Activity(Regularizer):
    """L1L2 activity regularizer: strength * (sum|x| + sum(x^2)).

    This is the pre-existing regularizer this project shipped with (applied via
    `apply_activity_regularizer` in model.py using `regularizers.L1L2`); pulled out
    here alongside HoyerSquare so both are selectable through the same `reg_type`
    switch, for comparability with the VWW/PlantVillage/Speech-Commands sweeps.
    """

    def __init__(self, strength=1e-6):
        self.strength = float(strength)

    def __call__(self, x):
        l1 = tf.reduce_sum(tf.abs(x))
        l2sq = tf.reduce_sum(tf.square(x))
        return self.strength * (l1 + l2sq)

    def get_config(self):
        return {"strength": self.strength}
