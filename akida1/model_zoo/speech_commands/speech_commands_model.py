import argparse

import tensorflow as tf
from tf_keras import regularizers
from tf_keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Reshape, Rescaling
from tf_keras.models import Model
from tf_keras.utils import set_random_seed
from akida_models.layer_blocks import conv_block, separable_conv_block
import yaml

tf.config.experimental.enable_op_determinism()


def build_ds_cnn(
    filters=64,
    dropout_initial=0.2,
    dropout_final=0.4,
    weight_decay=1e-4,
    input_shape=(49, 10, 1),
    num_classes=12,
    num_sep_conv_blocks=4,
    classifier_head="dense",
):
    """Build Akida 1.0-compatible DS-CNN for keyword spotting.

    Expects uint8 inputs [0, 255]; Rescaling layer normalises on-chip.
    Output is raw logits (no softmax) — use from_logits=True in loss.

    num_sep_conv_blocks counts the plain sep-conv blocks plus the trailing
    GAP block that is always present and always last (default 4 = today's
    3 plain blocks + 1 GAP block).
    """
    regularizer = regularizers.L2(weight_decay)

    inputs = Input(shape=input_shape)
    x = Rescaling(scale=1.0 / 255.0, offset=0.0, name="rescaling")(inputs)

    x = conv_block(
        x, filters, kernel_size=(5, 5), padding="same", strides=(2, 2),
        kernel_regularizer=regularizer, add_batchnorm=True, relu_activation="ReLU6",
    )
    x = Dropout(rate=dropout_initial)(x)

    for _ in range(num_sep_conv_blocks - 1):
        x = separable_conv_block(
            x, filters, kernel_size=(3, 3),
            pointwise_regularizer=regularizer,
            fused=True, add_batchnorm=True, relu_activation="ReLU6",
        )

    # 4th block: global average pooling before ReLU (Akida 1.0 requirement)
    x = separable_conv_block(
        x, filters, kernel_size=(3, 3),
        pointwise_regularizer=regularizer,
        fused=True, add_batchnorm=True, relu_activation="ReLU6",
        pooling="global_avg", post_relu_gap=False,
    )
    x = Dropout(rate=dropout_final)(x)

    if classifier_head == "pointwise_conv":
        # GAP flattened spatial dims to (batch, filters); restore (1, 1, filters)
        # before a 1x1 Conv2D — same pattern akida_models.kws.model_ds_cnn uses for
        # its include_top=False branch. No kernel_regularizer, matching Dense below.
        x = Reshape((1, 1, filters))(x)
        x = Conv2D(num_classes, (1, 1), padding="same", use_bias=True)(x)
        outputs = Flatten()(x)
    else:
        outputs = Dense(num_classes)(x)

    return Model(inputs=inputs, outputs=outputs, name="ds_cnn")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Build the KWS model for Akida 1')
    parser.add_argument("--config", default="configs/training_cfg.yml",
                        help='Model training configuration file')
    parser.add_argument("-s",
                        "--savepath",
                        type=str,
                        default='./models/kws_untrained.h5',
                        help="Save model with the specified path + name")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    set_random_seed(cfg['seed'])

    model = build_ds_cnn(
        filters=cfg["filters"],
        dropout_initial=cfg["dropout_initial"],
        dropout_final=cfg["dropout_final"],
        weight_decay=cfg["weight_decay"],
        num_sep_conv_blocks=cfg.get("num_sep_conv_blocks", 4),
        classifier_head=cfg.get("classifier_head", "dense"),
    )
    model.summary()
    model.save(args.savepath, include_optimizer=False) # dropout and weight_decay are correctly restored when loading back the model
    print(f'Model saved to {args.savepath}')
