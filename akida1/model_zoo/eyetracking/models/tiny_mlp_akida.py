"""Tiny MLP, tf_keras version for Akida deployment.

Unlike the Tiny Temporal CNN, this needed no architecture rework: quantizeml
supports Flatten immediately followed by Dense (folded together during
sanitize()'s remove_reshape transform), Dense has a direct QuantizedDense
mapping, and there's no temporal/buffered state to worry about - a plain
feedforward network. Same architecture as models/tiny_mlp.py, just
authored in tf_keras (Keras 2, required by quantizeml/cnn2snn) with a
uint8 input matching an ADC-digitized sensor reading.

Input (1, 1, 512) -> Rescaling -> Dense(64) -> ReLU -> Dense(32) -> ReLU ->
Dense(16) -> ReLU -> Dense(2)

The window is pre-flattened to 512 (=64*8) OUTSIDE the model (in numpy, see
training/train_mlp_akida.py) rather than via a Keras Flatten layer inside
the graph: quantizeml's QuantizedFlatten only accepts FixedPoint inputs and
rejected both a raw Input tensor and a QuantizedRescaling output (QFloat) -
since Flatten has no learnable weights, reshaping in data prep sidesteps
the issue entirely with identical math.
"""

import tf_keras as keras
from tf_keras import layers


def build_model(window=64, channels=8):
    # cnn2snn.convert() raises "Only signed inputs are supported" for a
    # uint8 InputData feeding QuantizedDense too (not just BufferTempConv,
    # per the streaming CNN's earlier fix) - use signed int8, recentering
    # the [0, 255] ADC-style reading to [-128, 127] (data prep subtracts
    # 128; Rescaling's offset undoes it for the float math).
    inputs = keras.Input(shape=(1, 1, window * channels), name="photodiode_signal", dtype="int8")
    x = layers.Rescaling(1.0 / 255.0, offset=128.0 / 255.0, name="rescaling")(inputs)
    x = layers.Dense(64)(x)
    x = layers.ReLU()(x)
    x = layers.Dense(32)(x)
    x = layers.ReLU()(x)
    x = layers.Dense(16)(x)
    x = layers.ReLU()(x)
    outputs = layers.Dense(2, name="gaze_xy")(x)

    return keras.Model(inputs, outputs, name="tiny_mlp_akida")


if __name__ == "__main__":
    model = build_model()
    model.summary()
    print(f"total params: {model.count_params():,}")
