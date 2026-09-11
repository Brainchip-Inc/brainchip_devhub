#!/usr/bin/env python
# Copyright 2025 Brainchip Holdings Ltd.  Apache 2.0 License
"""
UORED-VAFCLS bearing fault training.

Used for both stages of the pipeline: full-precision training from the untrained
model, and quantization-aware tuning of the quantized one. Only the loaded
model, the epoch count and the learning rate differ between the two.

The loss is binary cross-entropy over four independent fault labels, not a
softmax over classes: a bearing can in principle show more than one fault, and
a healthy bearing is the all-zero label vector rather than a fifth class. The
model emits raw logits, so `from_logits=True` throughout.

There is no validation set, and consequently no early stopping, no
learning-rate-on-plateau and no best-weight restore. This is deliberate. The
only data not used for training is the held-out fold, and selecting weights on
that would be exactly the leakage the bearing-disjoint split exists to prevent.
The recipe is instead fixed - a set epoch count, evaluated at the final epoch -
and folds 0-4 are reserved as the budget for choosing it. Please do not "fix"
this by adding a callback that monitors the held-out fold.

Example
-------
    python uored_vafcls_train.py -l models/akdcnn_uored_vafcls_untrained.h5 \\
        -s models/akdcnn_uored_vafcls.h5 -e 30 -lr 2e-4 --fold 5
"""

import argparse

import tensorflow as tf
from cnn2snn import load_quantized_model
from tf_keras import regularizers
from tf_keras.layers import ReLU
from tf_keras.losses import BinaryCrossentropy
from tf_keras.metrics import AUC
from tf_keras.optimizers.legacy import Adam
from tf_keras.optimizers.schedules import CosineDecay
from tf_keras.utils import set_random_seed

from uored_vafcls_data import FIXED_FOLD, NUM_LABELS, BATCH_SIZE, get_data

# Must be called before any TF ops to make GPU ops deterministic. It matters
# more here than in most examples: without it, re-running one unchanged fold
# moved its AUROC by up to 0.024, which is as large as the architecture
# differences this example exists to compare. Costs some throughput.
tf.config.experimental.enable_op_determinism()


def get_lr_schedule(initial_lr, total_steps, warmup_fraction=0.05):
    """Cosine decay with a short linear warmup, over the real step count.

    The schedule has to span the steps actually taken. The source implementation
    hard-coded the step count and undershot it by a sixth, which left the last
    few epochs running at the schedule's floor of zero - training for fewer
    epochs than it appeared to. Deriving total_steps from the dataset avoids
    that, and avoids it silently recurring if the batch size or the number of
    windows per recording ever changes.

    Args:
        initial_lr (float): the peak learning rate, reached after warmup.
        total_steps (int): optimizer steps over the whole run.
        warmup_fraction (float): fraction of total_steps spent warming up.

    Returns:
        tf_keras.optimizers.schedules.LearningRateSchedule: the schedule.
    """
    warmup_steps = max(1, int(warmup_fraction * total_steps))
    return CosineDecay(initial_learning_rate=initial_lr * 0.01,
                       warmup_target=initial_lr,
                       warmup_steps=warmup_steps,
                       decay_steps=max(1, total_steps - warmup_steps))


def train_uored_vafcls(model, train_ds, val_ds, epochs, learning_rate,
                       regularization=None, seed=0):
    """Train or quantization-aware tune the model on one fold.

    Args:
        model (tf_keras.Model): the model to train, float or quantized.
        train_ds (tf.data.Dataset): training dataset.
        val_ds (tf.data.Dataset): optional dataset to report per-epoch metrics
            on. Pass None, which is the default in the pipeline: the only
            candidate is the held-out fold, and nothing may be decided from it.
        epochs (int): number of epochs.
        learning_rate (float): peak learning rate for the cosine schedule.
        regularization (float, optional): activity regularization on the ReLU
            layers, to increase activation sparsity.
        seed (int): random seed.

    Returns:
        tf_keras.callbacks.History: the fit history.
    """
    set_random_seed(seed)

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------
    if regularization is not None:
        print('Adding Activity Regularization to ReLU layers')
        regularizer = regularizers.L1L2(regularization, regularization)
        for layer in model.layers:
            if isinstance(layer, ReLU):
                layer.activity_regularizer = regularizer

    steps_per_epoch = len(train_ds)
    lr_schedule = get_lr_schedule(learning_rate, steps_per_epoch * epochs)

    model.compile(optimizer=Adam(learning_rate=lr_schedule),
                  loss=BinaryCrossentropy(from_logits=True),
                  metrics=[AUC(name='auroc', multi_label=True,
                               num_labels=NUM_LABELS)])

    # ---------------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------------
    print(f'Training for {epochs} epochs, {steps_per_epoch} steps per epoch '
          f'({steps_per_epoch * epochs} steps), peak LR {learning_rate}')

    # No callbacks: the recipe is a fixed number of epochs and the final-epoch
    # weights are what gets evaluated. See the module docstring.
    return model.fit(train_ds, epochs=epochs, validation_data=val_ds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loadmodel', required=True,
                        help='Model to load (.h5 tf_keras or quantized model)')
    parser.add_argument('-s', '--savemodel', required=True,
                        help='Model save path')
    parser.add_argument('-d', '--data', default='./data/uored_vafcls',
                        help='Directory holding the prepared .npz cache')
    parser.add_argument('-b', '--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size. The default divides the 720 training '
                             'windows exactly, giving 6 steps per epoch')
    parser.add_argument('-e', '--epochs', type=int, default=30)
    parser.add_argument('-lr', '--learning_rate', type=float, default=2e-4,
                        help='Peak learning rate for the cosine schedule')
    parser.add_argument('-reg', '--regularization', type=float, default=None,
                        help='Activity Regularization to increase sparsity')
    parser.add_argument('--fold', type=int, default=FIXED_FOLD,
                        help='Bearing-disjoint fold index. Folds 0-4 are the '
                             'tuning folds, the only folds any hyperparameter '
                             'may be chosen on. Folds 5-104 are the 100 '
                             'evaluation folds; 5 is the fixed fold the '
                             'pretrained models and hardware benchmarks use')
    parser.add_argument('--monitor-heldout', action='store_true',
                        help='Report per-epoch metrics on the held-out fold. '
                             'For eyeballing the training curve only - no '
                             'decision may be taken from it')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    model = load_quantized_model(args.loadmodel)
    train_ds, test_ds = get_data(args.data, model.input_shape[1:],
                                 args.batch_size, fold=args.fold,
                                 seed=args.seed)

    val_ds = None
    if args.monitor_heldout:
        print('WARNING: --monitor-heldout reports metrics on the held-out '
              'bearings. It is for watching the curve only; choosing anything '
              'on the basis of it - epochs, learning rate, weights - leaks the '
              'test fold. Use folds 0-4 for that.')
        val_ds = test_ds

    train_uored_vafcls(model=model, train_ds=train_ds, val_ds=val_ds,
                       epochs=args.epochs, learning_rate=args.learning_rate,
                       regularization=args.regularization, seed=args.seed)
    model.save(args.savemodel, include_optimizer=False)
    print(f'Model saved as {args.savemodel}.')
