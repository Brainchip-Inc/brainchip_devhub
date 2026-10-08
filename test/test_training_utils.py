"""Unit tests for brainchip_utils.training_utils (no hardware required)."""
import numpy as np
import pytest
import tf_keras

from brainchip_utils.training_utils import HoyerSquare


@pytest.mark.parametrize("x, expected", [
    ([3.0, 0.0, 0.0, 0.0], 1.0),     # one non-zero value: the minimum, ||x||_1^2 / ||x||_2^2 = 1
    ([1.0, 1.0, 1.0, 1.0], 4.0),     # all equal: the maximum, the number of elements
    ([2.0, -2.0, 0.0, 0.0], 2.0),    # sign doesn't matter
])
def test_penalty_counts_effective_non_zeros(x, expected):
    assert float(HoyerSquare(1.0)(np.array(x, dtype=np.float32))) == pytest.approx(expected, rel=1e-5)


def test_penalty_is_scale_invariant_and_scaled_by_strength():
    x = np.array([0.5, 1.0, 0.0, 2.0], dtype=np.float32)
    assert float(HoyerSquare(1.0)(10 * x)) == pytest.approx(float(HoyerSquare(1.0)(x)), rel=1e-5)
    assert float(HoyerSquare(0.01)(x)) == pytest.approx(0.01 * float(HoyerSquare(1.0)(x)), rel=1e-5)


def test_zero_tensor_gives_zero_not_nan():
    assert float(HoyerSquare(1.0)(np.zeros(8, dtype=np.float32))) == 0.0


def test_serialization_round_trip():
    reg = HoyerSquare(2.5e-5)
    config = tf_keras.regularizers.serialize(reg)
    restored = tf_keras.regularizers.deserialize(config)
    assert isinstance(restored, HoyerSquare)
    assert restored.strength == pytest.approx(2.5e-5)
