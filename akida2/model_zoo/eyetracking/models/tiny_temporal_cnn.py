"""Tiny Temporal CNN: the project's primary reference model.

Input (64, 8) -> Conv1D(16, k=5) -> BN -> ReLU -> Conv1D(32, k=3) -> BN -> ReLU
-> GlobalAveragePooling1D -> Dense(32) -> ReLU -> Dense(2)

Target: < 50K parameters. See README.md for rationale.
"""

from tensorflow import keras
from keras import layers


def build_model(window=64, channels=8):
    inputs = keras.Input(shape=(window, channels), name="photodiode_signal")

    x = layers.Conv1D(16, kernel_size=5, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(32, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32)(x)
    x = layers.ReLU()(x)
    outputs = layers.Dense(2, name="gaze_xy")(x)

    return keras.Model(inputs, outputs, name="tiny_temporal_cnn")


if __name__ == "__main__":
    model = build_model()
    model.summary()
    print(f"total params: {model.count_params():,}")
