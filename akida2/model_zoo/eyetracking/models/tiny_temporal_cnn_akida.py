"""Tiny Temporal CNN, re-expressed for Akida deployment.

quantizeml/cnn2snn only quantize/convert Conv2D-family, Dense, and (for
genuinely causal streaming models) BufferTempConv layers - plain Conv1D and
GlobalAveragePooling1D are not supported. Conv1D over a (T, C) sequence is
mathematically identical to Conv2D over a (T, 1, C) "tall, 1-pixel-wide
image" with a (k, 1) kernel, so this reproduces the exact README
architecture (same math, same param count) using only Conv2D-family layers:

Input (64, 1, 8) -> Conv2D(16, (5,1)) -> BN -> ReLU -> Conv2D(32, (3,1)) ->
BN -> ReLU -> GlobalAveragePooling2D -> Dense(32) -> ReLU -> Dense(2)

Built with tf_keras (Keras 2), not tf.keras/Keras 3 - required by
quantizeml/cnn2snn, matching BrainChip's own akida_models reference code.
Input is uint8 (0-255), matching a real ADC-digitized sensor reading, with a
Rescaling layer to map back to float for the conv/dense math.
"""

import tf_keras as keras
from tf_keras import layers


def build_model(window=64, channels=8):
    inputs = keras.Input(shape=(window, 1, channels), name="photodiode_signal", dtype="uint8")
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    x = layers.Conv2D(16, kernel_size=(5, 1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(32, kernel_size=(3, 1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(32)(x)
    x = layers.ReLU()(x)
    outputs = layers.Dense(2, name="gaze_xy")(x)

    return keras.Model(inputs, outputs, name="tiny_temporal_cnn_akida")


if __name__ == "__main__":
    model = build_model()
    model.summary()
    print(f"total params: {model.count_params():,}")
