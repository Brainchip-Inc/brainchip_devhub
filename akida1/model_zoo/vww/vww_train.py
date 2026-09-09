#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
VWW training

Example
-------
    python vww_train.py -d /data/vww_coco2014_96/ -e 50  \\
        -l akidanet_vww_untrained.h5 -s akidanet_vww.h5
"""
import argparse
import tensorflow as tf

from tf_keras.losses import SparseCategoricalCrossentropy
from tf_keras.optimizers.legacy import Adam
from tf_keras.optimizers.schedules import CosineDecay
from tf_keras import regularizers
from tf_keras.layers import ReLU
from tf_keras.utils import set_random_seed

from cnn2snn import load_quantized_model

from vww_data import get_data

# Must be called before any TF ops to make GPU ops (conv backward passes,
# bilinear resize, etc.) deterministic. Has a small throughput cost.
tf.config.experimental.enable_op_determinism()


class HoyerSquare(regularizers.Regularizer):
    """Hoyer-Square activity regularizer: factor * (sum|x|)^2 / sum(x^2).

    Scale-invariant sparsity-inducing penalty (Yang, Wen & Li, "DeepHoyer",
    ICLR 2020). Unlike L1/L2, minimizing this ratio pushes activations toward a
    sparse (few large, rest ~zero) configuration rather than uniformly shrinking
    every value -- a large-but-sparse activation and a small-but-dense one can
    have the same L1 norm, but the sparse one has a lower Hoyer-Square value.

    Raw form is unbounded: for a tensor of N elements the ratio ranges up to N
    itself (dense/uniform case), so the same `factor` exerts more pressure on
    layers with more elements. With `normalize=True`, divides by N so the
    penalty is bounded in (0, 1] regardless of tensor size -- a prior
    unrelated project (DPLS) found this normalization is what prevents
    DeepHoyer from collapsing training (raw penalty on one activation batch
    measured ~797,000 vs ~0.5 normalized).
    """

    def __init__(self, factor, normalize=False):
        self.factor = float(factor)
        self.normalize = bool(normalize)

    def __call__(self, x):
        l1 = tf.reduce_sum(tf.abs(x))
        l2_sq = tf.reduce_sum(tf.square(x))
        # eps avoids 0/0 on an all-zero activation (e.g. early in training)
        hoyer_sq = tf.square(l1) / (l2_sq + 1e-12)
        if self.normalize:
            n = tf.cast(tf.size(x), tf.float32)
            hoyer_sq = hoyer_sq / n
        return self.factor * hoyer_sq

    def get_config(self):
        return {'factor': float(self.factor), 'normalize': self.normalize}


def train_vww(model, train_ds, val_ds, epochs, learning_rate, regularization=None,
              reg_type='l1l2', seed=42):
    set_random_seed(seed)
    
    steps_per_epoch = len(train_ds)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup

    lr_scheduler = CosineDecay(
        initial_learning_rate=0.0,
        decay_steps=total_steps - warmup_steps,
        warmup_target=learning_rate,
        warmup_steps=warmup_steps,
    )
    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    if regularization is not None:
        print(f'Adding {reg_type} Activity Regularization to ReLU layers')
        if reg_type == 'hoyer_square':
            regularizer = HoyerSquare(regularization)
        elif reg_type == 'hoyer_square_norm':
            regularizer = HoyerSquare(regularization, normalize=True)
        else:
            regularizer = regularizers.L1L2(regularization, regularization)
        for layer in model.layers:
            if isinstance(layer, ReLU):
                layer.activity_regularizer = regularizer

    model.compile(optimizer=Adam(learning_rate=lr_scheduler),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    
    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        # train_ds/val_ds are Keras Sequence-based generators (ImageDataGenerator);
        # without parallel workers, single-threaded augmentation CPU-starves the GPU
        # (measured ~30min/epoch on this dataset vs. ~4min/epoch with this enabled).
        workers=8,
        use_multiprocessing=True,
        max_queue_size=32,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras or .fbz akida model)')
    parser.add_argument('-s', '--savemodel', required=True,
                        help='Model save path')

    parser.add_argument('-d', '--data', default='./data/vw_coco2014_96',
                        help='VWW dataset root (contains train/ and val/ subdirs)')

    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('-reg', '--regularization', type=float, default=None,
                        help='Activity Regularization to increase sparsity')
    parser.add_argument('--reg-type', choices=['l1l2', 'hoyer_square', 'hoyer_square_norm'],
                        default='l1l2',
                        help='Type of activity regularizer to use with -reg')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    model = load_quantized_model(args.loadmodel)

    # ---------------------------------------------------------------------------
    # Data loading
    # ---------------------------------------------------------------------------
    train_ds, val_ds = get_data(args.data, model.input_shape[1:], args.batch_size, seed=args.seed)

    train_vww(model=model,
              train_ds=train_ds,
              val_ds=val_ds,
              epochs=args.epochs,
              learning_rate=args.learning_rate,
              regularization=args.regularization,
              reg_type=args.reg_type,
              seed=args.seed)
    
    model.save(args.savemodel, include_optimizer=False)
    print(f'Model saved as {args.savemodel}.')
