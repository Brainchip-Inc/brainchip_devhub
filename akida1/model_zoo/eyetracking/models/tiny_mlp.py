"""Tiny MLP: secondary baseline (MCU-style, no temporal convolution).

Input (64, 8) -> Flatten -> Dense(64) -> ReLU -> Dense(32) -> ReLU ->
Dense(16) -> ReLU -> Dense(2)

Target: ~10K-20K parameters. See README.md for rationale.
"""

from tensorflow import keras
from keras import layers


def build_model(window=64, channels=8):
    inputs = keras.Input(shape=(window, channels), name="photodiode_signal")

    x = layers.Flatten()(inputs)
    x = layers.Dense(64)(x)
    x = layers.ReLU()(x)
    x = layers.Dense(32)(x)
    x = layers.ReLU()(x)
    x = layers.Dense(16)(x)
    x = layers.ReLU()(x)
    outputs = layers.Dense(2, name="gaze_xy")(x)

    return keras.Model(inputs, outputs, name="tiny_mlp")


if __name__ == "__main__":
    model = build_model()
    model.summary()
    print(f"total params: {model.count_params():,}")
