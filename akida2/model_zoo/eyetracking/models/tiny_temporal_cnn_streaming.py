"""Tiny Temporal CNN, causal/streaming form for genuine Akida deployment.

The Conv2D reshape (tiny_temporal_cnn_akida.py) hit a real hardware limit:
Akida's 2D conv engine only supports SQUARE kernels, so a (5,1)-style
"height-only" kernel (our Conv1D-as-Conv2D trick) is rejected by
cnn2snn.convert(). The actual supported primitive for 1D temporal
convolution on Akida is BufferTempConv - a causal, FIFO-buffered conv used
by BrainChip's own eye-tracking reference model
(akida_models/tenn_spatiotemporal/eye_train.py). It requires:
  - a 5D (batch, T, H, W, C) input - here H=W=1, since we have no spatial
    structure, just C=8 channels over time
  - Conv3D layers with kernel_size=(k,1,1), causal-padded via ZeroPadding3D
    (pad only the past: (k-1, 0)), stride/dilation 1
  - no temporal pooling or reshape (both are explicitly unsupported by the
    Conv3D->BufferTempConv transform)

This makes the model inherently CAUSAL and STREAMING: instead of predicting
one gaze value from a whole symmetric 64-sample window (the original
GlobalAveragePooling1D design), it predicts gaze continuously at EVERY
timestep using only past context up to that point - the same shape output a
real low-power streaming deployment would actually produce. Early timesteps
in a window have little receptive field (warm-up) and will be less
accurate; see training/train_akida_streaming.py's per-position error curve.

Input (64, 1, 1, 8) -> causal Conv3D(16, k=5) -> BN -> ReLU -> causal
Conv3D(32, k=3) -> BN -> ReLU -> Dense(32) -> ReLU -> Dense(2), Dense layers
applied per-timestep (Keras Dense broadcasts over all leading dims).
"""

import tf_keras as keras
from tf_keras import layers


def build_model(window=64, channels=8):
    # Akida's buffered temporal-conv path (BufferTempConv, used for genuine
    # 1D streaming conv) only accepts SIGNED int8 input - cnn2snn.convert()
    # raises "Only signed inputs are supported" for a uint8 InputData feeding
    # a BufferTempConv. The ADC-style [0, 255] reading is recentered to
    # int8's [-128, 127] range (data prep subtracts 128; Rescaling's offset
    # undoes that shift for the float math).
    inputs = keras.Input(shape=(window, 1, 1, channels), name="photodiode_signal", dtype="int8")
    x = layers.Rescaling(1.0 / 255.0, offset=128.0 / 255.0, name="rescaling")(inputs)

    x = layers.ZeroPadding3D(padding=((4, 0), (0, 0), (0, 0)))(x)
    x = layers.Conv3D(16, kernel_size=(5, 1, 1), padding="valid", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.ZeroPadding3D(padding=((2, 0), (0, 0), (0, 0)))(x)
    x = layers.Conv3D(32, kernel_size=(3, 1, 1), padding="valid", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Dense(32)(x)
    x = layers.ReLU()(x)
    outputs = layers.Dense(2, name="gaze_xy")(x)
    # outputs shape: (batch, window, 1, 1, 2). A trailing Reshape to squeeze
    # the trivial (1, 1) spatial dims trips a bug in quantizeml's
    # replace_conv3d (_verify_reshape assumes every Reshape layer's config
    # has 'batch_input_shape', true only for a model's first layer) - so the
    # (1, 1) dims are left in place here and squeezed in numpy post-processing
    # instead (see training/train_akida_streaming.py).

    return keras.Model(inputs, outputs, name="tiny_temporal_cnn_streaming")


if __name__ == "__main__":
    model = build_model()
    model.summary()
    print(f"total params: {model.count_params():,}")
