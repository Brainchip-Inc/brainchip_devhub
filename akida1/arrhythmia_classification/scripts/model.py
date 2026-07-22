import tensorflow as tf
import tf_keras
from tf_keras import layers, models, regularizers
from tf_keras.layers import ReLU

from cnn2snn import quantize

def ds_block(x, filters, regularizer, name):
    """Standard Depthwise Separable Layer with targeted pointwise penalty constraints."""
    x = layers.SeparableConv2D(
        filters=filters, kernel_size=(5, 5), pointwise_regularizer=regularizer,
        padding="same", use_bias=False, name=f"{name}_sepconv"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(max_value=6, name=f"{name}_relu6")(x)
    return x


def final_ds_block(x, filters, regularizer, name):
    """Terminal Depthwise Feature Mapping Block omitting immediate localized activations."""
    x = layers.SeparableConv2D(
        filters=filters, kernel_size=(5, 5), pointwise_regularizer=regularizer,
        padding="same", use_bias=False, name=f"{name}_sepconv"
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return x


def build_akida_model(input_shape, num_classes,L1L2_reg_value):
    """Assembles the base FP32 architectural topology."""
    regularizer = regularizers.L2(L1L2_reg_value)
    inputs = layers.Input(shape=input_shape)
    
    # Stem Layer
    x = layers.Conv2D(16, (5, 5), padding="same", use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)
    
    # Feature Blocks with Downsampling Operations
    x = ds_block(x, 32, regularizer, "block1")
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)
    
    x = ds_block(x, 64, regularizer, "block2")
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)
    
    x = final_ds_block(x, 128, regularizer, "block3")
    
    # Classification Projection Block
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.ReLU(max_value=6)(x)
    x = layers.Dense(64, activation="linear")(x)
    x = layers.ReLU(max_value=6)(x)
    
    outputs = layers.Dense(num_classes, activation="linear")(x)
    return models.Model(inputs, outputs, name="Akida_ECG_Sparsity_Net")

def apply_activity_regularizer(model, reg_val):
    """Injects L1L2 Activity Penalties dynamically into the model's ReLU layers."""
    regularizer = regularizers.L1L2(l1=reg_val, l2=reg_val)
    for layer in model.layers:
        if isinstance(layer, ReLU):
            layer.activity_regularizer = regularizer
    print(f"Applied Activity Regularization (L1: {reg_val}, L2: {reg_val}) across network ReLU nodes.")


def prepare_qat_model(float_model_path):
    float_model = models.load_model(float_model_path) if isinstance(float_model_path, str) else float_model_path
    q_model = quantize(
        float_model,
        weight_quantization=4,
        activ_quantization=4,
        input_weight_quantization=8
    )
    return q_model